"""Monocular SVO state machine.

The implementation is populated from the same public state/result vocabulary
as SVO Pro while keeping ROS and backend concerns outside the core package.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from .alignment import PyramidalPatchTracker, SparseImageAligner
from .camera import Camera
from .config import SVOConfig
from .depth_filter import SynchronousDepthFilter
from .features import GridFeatureDetector
from .frame import EDGELET, INVALID_ID, FeatureSet, Frame, Landmark, SparseMap
from .geometry import invert_transform, transform_points
from .image import build_image_pyramid, image_gradients, prepare_image, sample_image
from .initialization import estimate_two_view_geometry
from .optimization import PoseOptimizer, optimize_point


class Stage(StrEnum):
    PAUSED = "paused"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    RELOCALIZING = "relocalizing"


class TrackingQuality(StrEnum):
    INSUFFICIENT = "insufficient"
    BAD = "bad"
    GOOD = "good"


class UpdateResult(StrEnum):
    DEFAULT = "default"
    KEYFRAME = "keyframe"
    FAILURE = "failure"


@dataclass(slots=True)
class OdometryResult:
    timestamp_ns: int
    stage: Stage
    quality: TrackingQuality
    update: UpdateResult
    T_world_cam: Tensor | None
    is_keyframe: bool
    num_observations: int
    sparse_points: Tensor | None = None
    message: str = ""


class MonoSVO:
    """Tensor-native visual-only monocular semi-direct odometry."""

    def __init__(self, camera: Camera, config: SVOConfig | None = None) -> None:
        self.config = SVOConfig() if config is None else config
        self.config.validate()
        self.device = self.config.torch_device()
        self.dtype = self.config.torch_dtype()
        self.camera = camera.to(device=self.device, dtype=self.dtype)
        self.detector = GridFeatureDetector(
            max_features=self.config.max_features,
            grid_size=self.config.grid_size,
            quality_level=self.config.detector_threshold,
            border=self.config.feature_border,
            edgelet_ratio=self.config.detector_edgelet_ratio,
            max_level=min(self.config.n_pyr_levels - 1, 2),
        )
        # SVO Pro's bootstrap tracker asks the detector for the complete
        # occupancy grid, independently of the ``max_fts`` map budget.  At
        # 752x480 with 30-pixel cells this permits 416 initial tracks, which
        # gives the five/eight-point estimator substantially more support than
        # the normal 180-feature tracking budget.
        initializer_capacity = (
            (self.camera.width + self.config.grid_size - 1) // self.config.grid_size
        ) * ((self.camera.height + self.config.grid_size - 1) // self.config.grid_size)
        self.initializer_detector = GridFeatureDetector(
            max_features=initializer_capacity,
            grid_size=self.config.grid_size,
            quality_level=self.config.detector_threshold,
            border=self.config.feature_border,
            edgelet_ratio=self.config.detector_edgelet_ratio,
            max_level=min(self.config.n_pyr_levels - 1, 2),
        )
        self.tracker = PyramidalPatchTracker(
            patch_size=self.config.patch_size,
            max_level=self.config.alignment_max_level,
            min_level=0,
            max_iterations=self.config.alignment_iterations,
            min_update=self.config.alignment_min_update,
        )
        self.aligner = SparseImageAligner(
            self.camera,
            patch_size=min(self.config.patch_size, 8),
            max_level=self.config.alignment_max_level,
            min_level=self.config.alignment_min_level,
            max_iterations=self.config.alignment_iterations,
        )
        self.pose_optimizer = PoseOptimizer(
            self.camera,
            max_iterations=self.config.pose_iterations,
            huber_delta=self.config.pose_huber_delta,
            outlier_threshold=self.config.pose_reprojection_threshold,
        )
        self.depth_filter = SynchronousDepthFilter(
            self.camera,
            convergence_threshold=self.config.seed_convergence_sigma2_thresh,
            max_updates=self.config.seed_max_updates,
            reprojection_threshold=self.config.pose_reprojection_threshold,
        )
        self.map = SparseMap(self.config.max_keyframes)
        self.stage = Stage.PAUSED
        self.quality = TrackingQuality.INSUFFICIENT
        self.last_frame: Frame | None = None
        self.previous_frame: Frame | None = None
        self.last_keyframe: Frame | None = None
        self._initial_reference: Frame | None = None
        self._initial_tracked_pixels: Tensor | None = None
        self._last_timestamp_ns: int | None = None
        self._last_motion: Tensor | None = None
        self._next_track_id = 0
        self._frames_since_keyframe = 0
        self._last_observation_count = 0
        self._relocalization_trials = 0

    def start(self) -> None:
        if self.stage != Stage.PAUSED:
            return
        self._clear_runtime()
        self.stage = Stage.INITIALIZING

    def reset(self) -> None:
        self._clear_runtime()
        self.stage = Stage.PAUSED

    def process(self, image: Tensor, timestamp_ns: int) -> OdometryResult:
        """Process one grayscale image and return the current frontend state."""

        timestamp_ns = int(timestamp_ns)
        # A paused frontend does not consume input.  This lets a caller probe
        # the state and then submit the same first frame after start().
        if self.stage == Stage.PAUSED:
            return self._result(timestamp_ns, UpdateResult.DEFAULT, None, "call start() first")
        if timestamp_ns < 0:
            raise ValueError("frame timestamps must be non-negative")
        if self._last_timestamp_ns is not None and timestamp_ns <= self._last_timestamp_ns:
            raise ValueError("frame timestamps must be strictly increasing")

        tensor = prepare_image(image, device=self.device, dtype=self.dtype)
        height, width = tensor.shape[-2:]
        if (width, height) != (self.camera.width, self.camera.height):
            raise ValueError(
                f"image is {width}x{height}, calibration expects "
                f"{self.camera.width}x{self.camera.height}"
            )
        pyramid = build_image_pyramid(tensor, self.config.image_pyramid_levels)
        # Input validation is transactional: malformed images do not prevent
        # retrying a corrected frame with the same timestamp.
        self._last_timestamp_ns = timestamp_ns
        initial_pose = self._predicted_pose()
        frame = Frame(
            camera=self.camera,
            image=tensor,
            timestamp_ns=timestamp_ns,
            T_world_cam=initial_pose,
            pyramid=pyramid,
        )

        if self.stage == Stage.INITIALIZING:
            return self._process_initialization(frame)
        return self._process_tracking(frame)

    def _clear_runtime(self) -> None:
        self.map.clear()
        self.depth_filter.clear()
        self.quality = TrackingQuality.INSUFFICIENT
        self.last_frame = None
        self.previous_frame = None
        self.last_keyframe = None
        self._initial_reference = None
        self._initial_tracked_pixels = None
        self._last_timestamp_ns = None
        self._last_motion = None
        self._next_track_id = 0
        self._frames_since_keyframe = 0
        self._last_observation_count = 0
        self._relocalization_trials = 0

    def _restart_initialization(self) -> None:
        last_timestamp = self._last_timestamp_ns
        self._clear_runtime()
        self._last_timestamp_ns = last_timestamp
        self.stage = Stage.INITIALIZING

    def _predicted_pose(self) -> Tensor:
        if self.last_frame is None:
            return torch.eye(4, device=self.device, dtype=self.dtype)
        # Match SVO Pro's visual-only path: without an IMU or an explicitly
        # weighted pose/image-alignment prior, the new frame starts at the
        # previous frame pose. Sparse image alignment supplies the inter-frame
        # motion estimate. Unconditionally extrapolating the last full SE(3)
        # update is materially different from the C++ frontend and amplifies
        # monocular depth/translation uncertainty into a poor patch-search
        # initialization.
        return self.last_frame.T_world_cam.clone()

    def _detect(
        self,
        frame: Frame,
        *,
        detector: GridFeatureDetector | None = None,
    ) -> FeatureSet:
        features = (self.detector if detector is None else detector).detect(
            frame.pyramid, self.camera.mask
        )
        count = len(features)
        features.track_ids = torch.arange(
            self._next_track_id,
            self._next_track_id + count,
            device=self.device,
            dtype=torch.long,
        )
        self._next_track_id += count
        frame.features = features
        return features

    @staticmethod
    def _current_gradients(
        pyramid: list[Tensor],
        pixels: Tensor,
        levels: Tensor,
        fallback: Tensor,
    ) -> Tensor:
        """Sample normalized feature directions in the current image pyramid."""

        if not pyramid:
            raise ValueError("pyramid must contain at least one image")
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("pixels must have shape [N, 2]")
        if levels.shape != (pixels.shape[0],) or fallback.shape != pixels.shape:
            raise ValueError("feature metadata must match the pixel count")

        eps = torch.finfo(pixels.dtype).eps
        fallback_norm = torch.linalg.vector_norm(fallback, dim=-1, keepdim=True)
        fallback_valid = torch.isfinite(fallback).all(dim=-1) & (fallback_norm.squeeze(-1) > eps)
        default = torch.zeros_like(fallback)
        default[:, 0] = 1.0
        refreshed = torch.where(
            fallback_valid[:, None], fallback / fallback_norm.clamp_min(eps), default
        ).clone()

        for level, image in enumerate(pyramid):
            at_level = levels == level
            if not bool(at_level.any()):
                continue
            dx, dy = image_gradients(image)
            level_pixels = pixels[at_level] / float(1 << level)
            sampled = torch.stack(
                (sample_image(dx, level_pixels), sample_image(dy, level_pixels)), dim=-1
            )
            norm = torch.linalg.vector_norm(sampled, dim=-1, keepdim=True)
            # Very weak current gradients are dominated by interpolation and
            # quantization noise. Retain the normalized prior direction there.
            usable = torch.isfinite(sampled).all(dim=-1) & (norm.squeeze(-1) > 32.0 * eps)
            current = sampled / norm.clamp_min(eps)
            refreshed[at_level] = torch.where(usable[:, None], current, refreshed[at_level])
        return refreshed.detach()

    @staticmethod
    def _tracked_features(
        source: FeatureSet,
        pixels: Tensor,
        status: Tensor,
        current_pyramid: list[Tensor],
    ) -> FeatureSet:
        kept = source[status]
        current_pixels = pixels[status].detach()
        return FeatureSet(
            # The frontend uses autograd internally to form alignment and pose
            # Jacobians, but frames are runtime state rather than a training
            # graph.  Detaching here prevents every tracked frame retaining the
            # complete optimizer graph.
            pixels=current_pixels,
            scores=kept.scores.detach(),
            levels=kept.levels.detach(),
            gradients=MonoSVO._current_gradients(
                current_pyramid,
                current_pixels,
                kept.levels,
                kept.gradients,
            ),
            kinds=kept.kinds.detach(),
            track_ids=kept.track_ids.detach(),
            landmark_ids=kept.landmark_ids.detach(),
        )

    def _accept_initial_reference(self, frame: Frame) -> OdometryResult:
        features = self._detect(frame, detector=self.initializer_detector)
        self._initial_reference = frame
        self._initial_tracked_pixels = features.pixels.detach().clone()
        if len(features) < self.config.init_min_features:
            return self._result(
                frame.timestamp_ns,
                UpdateResult.FAILURE,
                frame.T_world_cam,
                f"only {len(features)} initialization features",
            )
        return self._result(
            frame.timestamp_ns,
            UpdateResult.DEFAULT,
            frame.T_world_cam,
            "reference frame selected",
        )

    def _process_initialization(self, frame: Frame) -> OdometryResult:
        reference = self._initial_reference
        if reference is None:
            return self._accept_initial_reference(frame)
        assert reference.features is not None
        assert self._initial_tracked_pixels is not None
        tracked = self.tracker.track(
            reference.pyramid,
            frame.pyramid,
            reference.features.pixels,
            self._initial_tracked_pixels,
        )
        count = int(tracked.status.sum())
        # This intentionally follows the released C++ implementation, which
        # resets against init_min_features here (despite also parsing the
        # older init_min_tracked option).
        if count < self.config.init_min_features:
            return self._accept_initial_reference(frame)
        ref_tracked = reference.features[tracked.status]
        pixels_cur = tracked.pixels[tracked.status].detach()
        # Keep the first observation as the photometric template, but carry
        # the previous-frame estimate into the next KLT call and discard
        # terminated tracks.  This is FeatureTracker's default behavior in
        # rpg_svo_pro_open and is important once displacement grows beyond a
        # single coarse-level convergence basin.
        reference.features = ref_tracked
        self._initial_tracked_pixels = pixels_cur
        disparity = torch.linalg.vector_norm(pixels_cur - ref_tracked.pixels, dim=-1).median()
        if float(disparity) < self.config.init_min_disparity:
            return self._result(
                frame.timestamp_ns,
                UpdateResult.DEFAULT,
                reference.T_world_cam,
                f"initialization disparity {float(disparity):.2f}px",
            )

        geometry = estimate_two_view_geometry(
            self.camera,
            ref_tracked.pixels,
            pixels_cur,
            pixel_threshold=self.config.pose_reprojection_threshold,
            probability=self.config.init_ransac_probability,
            max_iterations=self.config.init_ransac_iterations,
            min_inliers=self.config.init_min_inliers,
            scene_depth=self.config.init_map_scale,
            min_parallax_degrees=self.config.init_min_parallax,
            random_seed=self.config.random_seed if self.config.deterministic else None,
        )
        if geometry is None:
            return self._accept_initial_reference(frame)

        ref_inliers = ref_tracked[geometry.inliers]
        current_pixels = pixels_cur[geometry.inliers]
        frame.features = FeatureSet(
            pixels=current_pixels,
            scores=ref_inliers.scores.clone(),
            levels=ref_inliers.levels.clone(),
            gradients=self._current_gradients(
                frame.pyramid,
                current_pixels,
                ref_inliers.levels,
                ref_inliers.gradients,
            ),
            kinds=ref_inliers.kinds.clone(),
            track_ids=ref_inliers.track_ids.clone(),
            landmark_ids=torch.full_like(ref_inliers.landmark_ids, INVALID_ID),
        )
        reference.features = ref_inliers
        frame.T_world_cam = reference.T_world_cam @ invert_transform(geometry.T_cur_ref)
        points_world = transform_points(reference.T_world_cam, geometry.points_ref)
        for index, point in enumerate(points_world):
            landmark = self.map.create_landmark(
                point,
                observations={reference.id: index, frame.id: index},
            )
            reference.features.landmark_ids[index] = landmark.id
            frame.features.landmark_ids[index] = landmark.id

        self.map.add_keyframe(reference)
        self._augment_keyframe_features(frame)
        self._initialize_keyframe_seeds(frame)
        self._record_keyframe_observations(frame)
        self.map.add_keyframe(frame)
        self.previous_frame = reference
        self.last_frame = frame
        self.last_keyframe = frame
        self._last_motion = geometry.T_cur_ref.detach()
        self._frames_since_keyframe = 0
        self._last_observation_count = int(geometry.inliers.sum())
        self._initial_reference = None
        self._initial_tracked_pixels = None
        self.stage = Stage.TRACKING
        self.quality = TrackingQuality.GOOD
        return self._result(
            frame.timestamp_ns,
            UpdateResult.KEYFRAME,
            frame.T_world_cam,
            f"initialized {int(geometry.inliers.sum())} landmarks",
            is_keyframe=True,
            observations=int(geometry.inliers.sum()),
        )

    def _landmark_points(self, features: FeatureSet) -> tuple[Tensor, Tensor]:
        indices: list[int] = []
        points: list[Tensor] = []
        landmark_ids = features.landmark_ids.detach().cpu().tolist()
        track_ids = features.track_ids.detach().cpu().tolist()
        for index, (landmark_id, track_id) in enumerate(zip(landmark_ids, track_ids, strict=True)):
            landmark = self.map.landmarks.get(int(landmark_id))
            if landmark is not None:
                indices.append(index)
                points.append(landmark.position_world.to(device=self.device, dtype=self.dtype))
                continue
            seed_point = self.depth_filter.point_for_track(int(track_id))
            if seed_point is not None:
                indices.append(index)
                points.append(seed_point.to(device=self.device, dtype=self.dtype))
        if not indices:
            return (
                torch.empty(0, device=self.device, dtype=torch.long),
                torch.empty((0, 3), device=self.device, dtype=self.dtype),
            )
        return torch.tensor(indices, device=self.device), torch.stack(points)

    def _num_tracked_landmarks(self, features: FeatureSet) -> int:
        """Count map-backed tracks, excluding inverse-depth seed references."""

        return sum(
            int(landmark_id) in self.map.landmarks
            for landmark_id in features.landmark_ids.detach().cpu().tolist()
        )

    def _direct_pose_prior(self, reference: Frame, frame: Frame) -> None:
        if not self.config.use_sparse_image_alignment or reference.features is None:
            return
        indices, points_world = self._landmark_points(reference.features)
        if indices.numel() < 6:
            return
        points_ref = reference.camera_from_world(points_world)
        depths = torch.linalg.vector_norm(points_ref, dim=-1)
        # The supported visual-only path has no external motion prior, so the
        # original frontend initializes sparse alignment with the identity
        # relative pose. Construct it exactly: forming R.T @ R from float32
        # pose matrices would amplify harmless SO(3) round-off every frame.
        T_cur_ref = torch.eye(4, device=self.device, dtype=self.dtype)
        aligned = self.aligner.align(
            reference.pyramid,
            frame.pyramid,
            reference.features.pixels[indices],
            depths,
            T_cur_ref,
        )
        alignment_is_usable = (
            aligned.converged
            and int(aligned.valid.sum()) >= 6
            and torch.isfinite(aligned.T_cur_ref).all()
            and torch.isfinite(torch.as_tensor(aligned.error))
            and aligned.error <= self.aligner.huber_delta
        )
        if alignment_is_usable:
            frame.T_world_cam = (
                reference.T_world_cam @ invert_transform(aligned.T_cur_ref)
            ).detach()

    def _tracking_initial_pixels(self, reference: Frame, frame: Frame) -> Tensor:
        assert reference.features is not None
        pixels = reference.features.pixels.clone()
        indices, points_world = self._landmark_points(reference.features)
        if indices.numel():
            projected, valid = self.camera.project(frame.camera_from_world(points_world))
            chosen = indices[valid]
            pixels[chosen] = projected[valid]
        return pixels

    def _recover_map_landmarks(self, frame: Frame, current: FeatureSet) -> FeatureSet:
        """Patch-match visible, currently unobserved landmarks from keyframes.

        Every landmark is sourced from its newest retained keyframe observation
        whose feature index is still valid.  The motion/direct-alignment prior
        supplies the initial current-frame projection; patch tracking performs
        the photometric verification.  Existing landmark and track identifiers
        are excluded before matching so merging cannot duplicate observations.
        """

        observed_landmarks = {
            landmark_id
            for landmark_id in current.landmark_ids.detach().cpu().tolist()
            if landmark_id in self.map.landmarks
        }
        observed_tracks = set(current.track_ids.detach().cpu().tolist())
        capacity = self.config.max_features - len(observed_landmarks)
        if capacity <= 0 or not self.map.landmarks or not self.map.keyframes:
            return FeatureSet.empty(device=self.device, dtype=self.dtype)

        keyframes = list(self.map.keyframes.values())[-self.config.reprojector_max_keyframes :]
        source_indices: dict[int, list[int]] = {}
        claimed_landmarks = set(observed_landmarks)
        claimed_tracks = set(observed_tracks)

        # Iterating newest first selects the latest valid observation without
        # depending on landmark-dictionary insertion order.
        for source in reversed(keyframes):
            if source.features is None:
                continue
            landmark_ids = source.features.landmark_ids.detach().cpu().tolist()
            track_ids = source.features.track_ids.detach().cpu().tolist()
            for feature_index, (landmark_id, track_id) in enumerate(
                zip(landmark_ids, track_ids, strict=True)
            ):
                landmark = self.map.landmarks.get(int(landmark_id))
                if (
                    landmark is None
                    or landmark.id in claimed_landmarks
                    or int(track_id) in claimed_tracks
                    or landmark.observations.get(source.id) != feature_index
                ):
                    continue
                source_indices.setdefault(source.id, []).append(feature_index)
                claimed_landmarks.add(landmark.id)
                claimed_tracks.add(int(track_id))

        if not source_indices:
            return FeatureSet.empty(device=self.device, dtype=self.dtype)

        recovered_parts: list[FeatureSet] = []
        recovered_landmarks = set(observed_landmarks)
        recovered_tracks = set(observed_tracks)
        recovered_count = 0

        for source in reversed(keyframes):
            indices = source_indices.get(source.id)
            if not indices or source.features is None:
                continue

            landmark_ids = source.features.landmark_ids[indices].detach().cpu().tolist()
            points_world = torch.stack(
                [
                    self.map.landmarks[int(landmark_id)].position_world.to(
                        device=self.device, dtype=self.dtype
                    )
                    for landmark_id in landmark_ids
                ]
            )
            projected, visible = self.camera.project(frame.camera_from_world(points_world))
            visible_positions = torch.nonzero(visible, as_tuple=False).squeeze(-1)
            if not visible_positions.numel():
                continue
            source_index_tensor = torch.tensor(indices, device=self.device, dtype=torch.long)
            usable_source_indices = source_index_tensor[visible_positions]
            usable_initial_pixels = projected[visible_positions]

            cursor = 0
            while cursor < usable_source_indices.numel() and recovered_count < capacity:
                # Failed matches do not consume capacity. Processing bounded
                # chunks lets later candidates from the same keyframe fill it.
                remaining = capacity - recovered_count
                stop = min(cursor + remaining, usable_source_indices.numel())
                batch_indices = usable_source_indices[cursor:stop]
                initial_pixels = usable_initial_pixels[cursor:stop]
                source_features = source.features[batch_indices]
                tracked = self.tracker.track(
                    source.pyramid,
                    frame.pyramid,
                    source_features.pixels,
                    initial_pixels,
                )
                part = self._tracked_features(
                    source_features,
                    tracked.pixels,
                    tracked.status,
                    frame.pyramid,
                )
                cursor = stop
                if not len(part):
                    continue

                keep: list[int] = []
                for index, (landmark_id, track_id) in enumerate(
                    zip(
                        part.landmark_ids.detach().cpu().tolist(),
                        part.track_ids.detach().cpu().tolist(),
                        strict=True,
                    )
                ):
                    if (
                        landmark_id not in self.map.landmarks
                        or landmark_id in recovered_landmarks
                        or track_id in recovered_tracks
                    ):
                        continue
                    keep.append(index)
                    recovered_landmarks.add(int(landmark_id))
                    recovered_tracks.add(int(track_id))
                    if recovered_count + len(keep) >= capacity:
                        break
                if keep:
                    kept = part[torch.tensor(keep, device=self.device, dtype=torch.long)]
                    recovered_parts.append(kept)
                    recovered_count += len(kept)

        if not recovered_parts:
            return FeatureSet.empty(device=self.device, dtype=self.dtype)
        return FeatureSet.concatenate(recovered_parts)

    def _merge_recovered_features(
        self,
        current: FeatureSet,
        recovered: FeatureSet,
    ) -> FeatureSet:
        """Merge unique map matches, preferring landmarks over seedless tracks."""

        if not len(recovered):
            return current
        remaining_current = max(0, self.config.max_features - len(recovered))
        landmark_indices: list[int] = []
        other_indices: list[int] = []
        seen_landmarks: set[int] = set()
        seen_tracks: set[int] = set()
        for index, (landmark_id, track_id) in enumerate(
            zip(
                current.landmark_ids.detach().cpu().tolist(),
                current.track_ids.detach().cpu().tolist(),
                strict=True,
            )
        ):
            if int(track_id) in seen_tracks:
                continue
            if landmark_id in self.map.landmarks:
                if int(landmark_id) in seen_landmarks:
                    continue
                landmark_indices.append(index)
                seen_landmarks.add(int(landmark_id))
            else:
                other_indices.append(index)
            seen_tracks.add(int(track_id))

        keep = landmark_indices[:remaining_current]
        keep.extend(other_indices[: max(0, remaining_current - len(keep))])
        keep.sort()
        kept_current = (
            current[torch.tensor(keep, device=self.device, dtype=torch.long)]
            if keep
            else FeatureSet.empty(device=self.device, dtype=self.dtype)
        )
        return FeatureSet.concatenate((kept_current, recovered))

    def _process_tracking(self, frame: Frame) -> OdometryResult:
        reference = self.last_frame
        if reference is None or reference.features is None:
            self._restart_initialization()
            return self._process_initialization(frame)

        self._direct_pose_prior(reference, frame)
        tracked = self.tracker.track(
            reference.pyramid,
            frame.pyramid,
            reference.features.pixels,
            self._tracking_initial_pixels(reference, frame),
        )
        frame.features = self._tracked_features(
            reference.features,
            tracked.pixels,
            tracked.status,
            frame.pyramid,
        )
        recovered = self._recover_map_landmarks(frame, frame.features)
        frame.features = self._merge_recovered_features(frame.features, recovered)
        landmark_indices, points_world = self._landmark_points(frame.features)
        if landmark_indices.numel() < 6:
            return self._tracking_failure(
                frame,
                "too few landmark tracks after local-map recovery",
            )

        landmark_features = frame.features[landmark_indices]
        optimized = self.pose_optimizer.optimize(
            frame.T_world_cam,
            points_world,
            landmark_features.pixels,
            kinds=landmark_features.kinds,
            levels=landmark_features.levels,
            gradients=landmark_features.gradients,
        )
        optimization_is_usable = (
            optimized.converged or optimized.final_error <= self.config.pose_reprojection_threshold
        ) and torch.isfinite(torch.as_tensor(optimized.final_error))
        if not optimization_is_usable:
            return self._tracking_failure(frame, "pose optimization did not improve")
        frame.T_world_cam = optimized.T_world_cam
        inlier_count = int(optimized.inliers.sum())
        if inlier_count < self.config.quality_min_features:
            return self._tracking_failure(
                frame,
                f"only {inlier_count} pose inliers",
                observations=inlier_count,
            )

        # Remove geometrically rejected landmark observations while retaining
        # untriangulated tracks that may become landmarks at the next keyframe.
        keep = torch.ones(len(frame.features), dtype=torch.bool, device=self.device)
        keep[landmark_indices[~optimized.inliers]] = False
        frame.features = frame.features[keep]
        tracked_landmarks = self._num_tracked_landmarks(frame.features)
        seed_updates = self.depth_filter.update_observed(frame)
        self._optimize_structure(frame)
        feature_drop = self._last_observation_count - inlier_count
        self.quality = (
            TrackingQuality.BAD
            if feature_drop > self.config.quality_max_feature_drop
            else TrackingQuality.GOOD
        )
        was_relocalizing = self.stage == Stage.RELOCALIZING
        self.stage = Stage.TRACKING
        self._relocalization_trials = 0
        self._frames_since_keyframe += 1

        update = UpdateResult.DEFAULT
        is_keyframe = False
        promoted_seeds = 0
        initialized_seeds = 0
        if self._needs_keyframe(frame, tracked_landmarks):
            promoted_seeds = self.depth_filter.promote_observed(frame, self.map)
            self._promote_new_landmarks(frame)
            self._augment_keyframe_features(frame)
            initialized_seeds = self._initialize_keyframe_seeds(frame)
            self._record_keyframe_observations(frame)
            self.map.add_keyframe(frame)
            self.depth_filter.discard_missing_keyframes(set(self.map.keyframes))
            self.last_keyframe = frame
            self._frames_since_keyframe = 0
            update = UpdateResult.KEYFRAME
            is_keyframe = True

        T_cur_ref = invert_transform(frame.T_world_cam) @ reference.T_world_cam
        self.previous_frame = reference
        self.last_frame = frame
        self._last_motion = T_cur_ref.detach()
        self._last_observation_count = inlier_count
        message = "relocalized" if was_relocalizing else "tracking"
        if len(recovered):
            message = f"{message}; recovered {len(recovered)} map landmarks"
        if seed_updates:
            message = f"{message}; updated {seed_updates} depth seeds"
        if promoted_seeds or initialized_seeds:
            message = (
                f"{message}; promoted {promoted_seeds} and initialized "
                f"{initialized_seeds} depth seeds"
            )
        return self._result(
            frame.timestamp_ns,
            update,
            frame.T_world_cam,
            message,
            is_keyframe=is_keyframe,
            observations=inlier_count,
        )

    def _tracking_failure(
        self,
        frame: Frame,
        message: str,
        *,
        observations: int = 0,
    ) -> OdometryResult:
        self.quality = TrackingQuality.INSUFFICIENT
        self.stage = Stage.RELOCALIZING
        self._relocalization_trials += 1
        result = self._result(
            frame.timestamp_ns,
            UpdateResult.FAILURE,
            frame.T_world_cam,
            message,
            observations=observations,
        )
        if self._relocalization_trials >= self.config.relocalization_max_trials:
            self._restart_initialization()
            result.stage = self.stage
            result.sparse_points = self.map.points_tensor(dtype=self.dtype, device=self.device)
            result.message = f"{message}; restarting initialization"
        return result

    def _structure_observations(
        self,
        landmark: Landmark,
    ) -> tuple[list[Tensor], list[Tensor], list[Camera]] | None:
        """Collect retained keyframe measurements for one map landmark."""

        poses: list[Tensor] = []
        pixels: list[Tensor] = []
        cameras: list[Camera] = []
        for frame_id, feature_index in landmark.observations.items():
            keyframe = self.map.keyframes.get(frame_id)
            if (
                keyframe is None
                or keyframe.features is None
                or not 0 <= feature_index < len(keyframe.features)
            ):
                continue
            poses.append(keyframe.T_world_cam)
            pixels.append(keyframe.features.pixels[feature_index])
            cameras.append(keyframe.camera)
        if len(poses) < 2:
            return None
        return poses, pixels, cameras

    def _optimize_structure(self, frame: Frame) -> int:
        """Refine a bounded, least-recently-optimized set of corner landmarks.

        This mirrors ``FrameHandlerBase::optimizeStructure``: points must be
        visible as corner observations in the current frame, but their stored
        keyframe observations define the structure objective. The explicit
        feature-order tie break makes the C++ ``nth_element`` policy
        deterministic when several points have the same last-update frame.
        """

        limit = self.config.structure_optimization_max_points
        if limit == 0 or frame.features is None:
            return 0

        candidates: list[tuple[int, int, Landmark, list[Tensor], list[Tensor], list[Camera]]] = []
        seen: set[int] = set()
        for feature_index, (kind, landmark_id) in enumerate(
            zip(
                frame.features.kinds.detach().cpu().tolist(),
                frame.features.landmark_ids.detach().cpu().tolist(),
                strict=True,
            )
        ):
            landmark = self.map.landmarks.get(int(landmark_id))
            if int(kind) == EDGELET or landmark is None or landmark.id in seen:
                continue
            observations = self._structure_observations(landmark)
            if observations is None:
                continue
            poses, pixels, cameras = observations
            candidates.append(
                (
                    landmark.last_structure_optimization,
                    feature_index,
                    landmark,
                    poses,
                    pixels,
                    cameras,
                )
            )
            seen.add(landmark.id)

        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        if limit > 0:
            candidates = candidates[:limit]

        optimized = 0
        for _, _, landmark, poses, pixels, cameras in candidates:
            refined = optimize_point(
                landmark.position_world,
                torch.stack(poses),
                torch.stack(pixels),
                cameras,
                max_iterations=5,
                huber_delta=self.config.pose_huber_delta,
            )
            if bool(torch.isfinite(refined).all()):
                landmark.position_world = refined.detach()
            landmark.last_structure_optimization = frame.id
            optimized += 1
        return optimized

    def _common_track_disparity(self, first: Frame, second: Frame) -> float:
        assert first.features is not None and second.features is not None
        second_lookup = {
            int(track): index
            for index, track in enumerate(second.features.track_ids.detach().cpu().tolist())
        }
        first_indices: list[int] = []
        second_indices: list[int] = []
        for index, track in enumerate(first.features.track_ids.detach().cpu().tolist()):
            match = second_lookup.get(int(track))
            if match is not None:
                first_indices.append(index)
                second_indices.append(match)
        if not first_indices:
            return float("inf")
        displacement = first.features.pixels[first_indices] - second.features.pixels[second_indices]
        return float(torch.linalg.vector_norm(displacement, dim=-1).median().detach())

    def _needs_keyframe(self, frame: Frame, observations: int) -> bool:
        if self.last_keyframe is None:
            return True
        if self.config.keyframe_criterion == "DOWNLOOKING":
            # SVO Pro normalizes this test by scene depth. Landmark depth is
            # directly available here, so use its median just as the C++
            # frontend's frame_utils::getSceneDepth does.
            indices, points_world = self._landmark_points(frame.features)
            if indices.numel():
                depths = torch.linalg.vector_norm(frame.camera_from_world(points_world), dim=-1)
                finite_depths = depths[torch.isfinite(depths) & (depths > 0)]
                median_depth = float(finite_depths.median()) if finite_depths.numel() else 1.0
            else:
                median_depth = 1.0
            for keyframe in self.map.keyframes.values():
                relative = frame.camera_from_world(keyframe.position)
                limit = self.config.keyframe_min_distance * median_depth
                if (
                    abs(float(relative[0])) < limit
                    and abs(float(relative[1])) < limit * 0.8
                    and abs(float(relative[2])) < limit * 1.3
                ):
                    return False
            return True

        # SVO Pro's FORWARD selector intentionally checks feature-count bounds
        # before disparity and pose overlap. Critically low tracking therefore
        # forces a keyframe while there are still enough landmarks to recover.
        if observations > self.config.keyframe_num_features_upper:
            return False
        if self._frames_since_keyframe < self.config.keyframe_min_frames:
            return False
        if observations < self.config.keyframe_num_features_lower:
            return True
        disparity = self._common_track_disparity(self.last_keyframe, frame)
        if disparity < self.config.keyframe_min_disparity:
            return False

        for keyframe in self.map.keyframes.values():
            rotation = frame.T_world_cam[:3, :3].transpose(-1, -2) @ keyframe.T_world_cam[:3, :3]
            cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
            angle = float(torch.rad2deg(torch.acos(cosine)))
            distance = float(torch.linalg.vector_norm(frame.position - keyframe.position))
            if (
                angle < self.config.keyframe_min_angle_degrees
                and distance < self.config.keyframe_min_distance_metric
            ):
                return False
        return True

    @staticmethod
    def _triangulate_rays(
        bearings_ref: Tensor,
        bearings_cur: Tensor,
        T_cur_ref: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        rotation = T_cur_ref[:3, :3]
        translation = T_cur_ref[:3, 3]
        rotated_ref = torch.einsum("ij,nj->ni", rotation, bearings_ref)
        system = torch.stack((rotated_ref, -bearings_cur), dim=-1)
        rhs = -translation[None, :, None].expand(bearings_ref.shape[0], 3, 1)
        depths = torch.linalg.lstsq(system, rhs).solution.squeeze(-1)
        points_ref = bearings_ref * depths[:, :1]
        points_cur = transform_points(T_cur_ref, points_ref)
        return points_ref, points_cur, depths

    def _promote_new_landmarks(self, frame: Frame) -> None:
        reference = self.last_keyframe
        if reference is None or reference.features is None or frame.features is None:
            return
        current_lookup = {
            int(track): index
            for index, track in enumerate(frame.features.track_ids.detach().cpu().tolist())
        }
        ref_indices: list[int] = []
        cur_indices: list[int] = []
        for ref_index, track in enumerate(reference.features.track_ids.detach().cpu().tolist()):
            cur_index = current_lookup.get(int(track))
            if (
                cur_index is not None
                and int(reference.features.landmark_ids[ref_index]) == INVALID_ID
                and int(frame.features.landmark_ids[cur_index]) == INVALID_ID
            ):
                ref_indices.append(ref_index)
                cur_indices.append(cur_index)
        if len(ref_indices) < 2:
            return
        T_cur_ref = invert_transform(frame.T_world_cam) @ reference.T_world_cam
        if torch.linalg.vector_norm(T_cur_ref[:3, 3]) < 1e-3:
            return
        pixels_ref = reference.features.pixels[ref_indices]
        pixels_cur = frame.features.pixels[cur_indices]
        bearings_ref = self.camera.unproject(pixels_ref)
        bearings_cur = self.camera.unproject(pixels_cur)
        points_ref, points_cur, depths = self._triangulate_rays(
            bearings_ref, bearings_cur, T_cur_ref
        )
        reproj_ref, valid_ref = self.camera.project(points_ref)
        reproj_cur, valid_cur = self.camera.project(points_cur)
        errors = torch.maximum(
            torch.linalg.vector_norm(reproj_ref - pixels_ref, dim=-1),
            torch.linalg.vector_norm(reproj_cur - pixels_cur, dim=-1),
        )
        valid = (
            valid_ref
            & valid_cur
            & (depths[:, 0] > 0)
            & (depths[:, 1] > 0)
            & (errors <= self.config.pose_reprojection_threshold)
        )
        points_world = transform_points(reference.T_world_cam, points_ref)
        for local_index in torch.nonzero(valid, as_tuple=False).squeeze(-1).detach().cpu().tolist():
            ref_index = ref_indices[local_index]
            cur_index = cur_indices[local_index]
            landmark = self.map.create_landmark(
                points_world[local_index],
                observations={reference.id: ref_index, frame.id: cur_index},
            )
            reference.features.landmark_ids[ref_index] = landmark.id
            frame.features.landmark_ids[cur_index] = landmark.id

    def _initialize_keyframe_seeds(self, frame: Frame) -> int:
        """Add inverse-depth priors after existing observations occupy the frame."""

        assert frame.features is not None
        _, points_world = self._landmark_points(frame.features)
        if points_world.numel():
            scene_depths = torch.linalg.vector_norm(frame.camera_from_world(points_world), dim=-1)
        else:
            scene_depths = torch.empty(0, device=self.device, dtype=self.dtype)
        return self.depth_filter.add_keyframe(
            frame,
            scene_depths,
            fallback_depth=self.config.init_map_scale,
        )

    def _augment_keyframe_features(self, frame: Frame) -> None:
        assert frame.features is not None
        capacity = self.config.max_features - len(frame.features)
        if capacity <= 0:
            return
        detected = self.detector.detect(frame.pyramid, self.camera.mask)
        if not len(detected):
            return
        if len(frame.features):
            distances = torch.cdist(detected.pixels, frame.features.pixels)
            available = distances.min(dim=1).values >= self.config.grid_size * 0.5
            detected = detected[available]
        if not len(detected):
            return
        detected = detected[torch.arange(min(capacity, len(detected)), device=self.device)]
        detected.track_ids = torch.arange(
            self._next_track_id,
            self._next_track_id + len(detected),
            device=self.device,
            dtype=torch.long,
        )
        self._next_track_id += len(detected)
        frame.features = FeatureSet.concatenate((frame.features, detected))

    def _record_keyframe_observations(self, frame: Frame) -> None:
        assert frame.features is not None
        for index, landmark_id in enumerate(frame.features.landmark_ids.detach().cpu().tolist()):
            landmark = self.map.landmarks.get(int(landmark_id))
            if landmark is not None:
                landmark.add_observation(frame.id, index)

    def _result(
        self,
        timestamp_ns: int,
        update: UpdateResult,
        pose: Tensor | None,
        message: str,
        *,
        is_keyframe: bool = False,
        observations: int = 0,
    ) -> OdometryResult:
        return OdometryResult(
            timestamp_ns=timestamp_ns,
            stage=self.stage,
            quality=self.quality,
            update=update,
            T_world_cam=None if pose is None else pose.detach().clone(),
            is_keyframe=is_keyframe,
            num_observations=observations,
            sparse_points=self.map.points_tensor(dtype=self.dtype, device=self.device),
            message=message,
        )
