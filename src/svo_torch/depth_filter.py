"""Bounded synchronous inverse-depth filtering for the monocular frontend.

SVO Pro stores newly detected keyframe features as inverse-depth seeds.  A
seed observation can constrain the current pose before it has converged, and
an observed seed is upgraded to a landmark when the current frame becomes a
keyframe.  This module keeps the same lifecycle without the original ROS/C++
worker thread: all state is tensor-native and updates run synchronously in the
frontend call that owns the frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .camera import Camera
from .depth import (
    DepthSeeds,
    compute_tau,
    inverse_depth_variance,
    triangulate_depth,
    update_filter_vogiatzis,
)
from .frame import INVALID_ID, Frame, SparseMap
from .geometry import invert_transform, transform_points


@dataclass(slots=True)
class KeyframeDepthSeeds:
    """Inverse-depth seed arrays owned by one keyframe."""

    frame: Frame
    feature_indices: Tensor
    track_ids: Tensor
    state: DepthSeeds
    mu_range: float
    updates: Tensor

    def __len__(self) -> int:
        return int(self.feature_indices.numel())


class SynchronousDepthFilter:
    """A deterministic, bounded counterpart of SVO Pro's depth-filter worker."""

    def __init__(
        self,
        camera: Camera,
        *,
        convergence_threshold: float = 200.0,
        max_updates: int = 30,
        reprojection_threshold: float = 2.0,
    ) -> None:
        if convergence_threshold <= 0 or max_updates < 1 or reprojection_threshold <= 0:
            raise ValueError("invalid synchronous depth-filter options")
        self.camera = camera
        self.convergence_threshold = float(convergence_threshold)
        self.max_updates = int(max_updates)
        self.reprojection_threshold = float(reprojection_threshold)
        self.keyframes: dict[int, KeyframeDepthSeeds] = {}
        self._track_lookup: dict[int, tuple[int, int]] = {}
        self._pixel_error_angle = self._compute_pixel_error_angle()

    def _compute_pixel_error_angle(self) -> Tensor:
        dtype = self.camera.dtype
        device = self.camera.device
        center = torch.tensor(
            [[0.5 * self.camera.width, 0.5 * self.camera.height]],
            dtype=dtype,
            device=device,
        )
        adjacent = center + center.new_tensor([[1.0, 0.0]])
        first = self.camera.unproject(center)
        second = self.camera.unproject(adjacent)
        cosine = (first * second).sum(dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cosine)[0].detach()

    def clear(self) -> None:
        self.keyframes.clear()
        self._track_lookup.clear()

    @property
    def active_count(self) -> int:
        return len(self._track_lookup)

    def add_keyframe(
        self,
        frame: Frame,
        scene_depths: Tensor,
        *,
        fallback_depth: float = 1.0,
    ) -> int:
        """Initialize seeds for the keyframe's feature slots without landmarks.

        The prior follows ``FrameHandlerMono``: median scene depth, with the
        inverse-depth range initialized from half the minimum scene depth.
        """

        if frame.features is None:
            return 0
        invalid = frame.features.landmark_ids == INVALID_ID
        feature_indices = torch.nonzero(invalid, as_tuple=False).squeeze(-1)
        if not feature_indices.numel():
            return 0

        depths = scene_depths.to(device=self.camera.device, dtype=self.camera.dtype).reshape(-1)
        depths = depths[torch.isfinite(depths) & (depths > 0)]
        if depths.numel():
            depth_mean = float(depths.median())
            depth_min = 0.5 * float(depths.min())
        else:
            depth_mean = float(fallback_depth)
            depth_min = 0.5 * depth_mean
        tiny = float(torch.finfo(self.camera.dtype).eps)
        depth_mean = max(depth_mean, tiny)
        depth_min = max(depth_min, depth_mean * 1e-3, tiny)

        count = int(feature_indices.numel())
        state = DepthSeeds.initialize(
            count,
            depth_mean=depth_mean,
            depth_min=depth_min,
            dtype=self.camera.dtype,
            device=self.camera.device,
        )
        track_ids = frame.features.track_ids[feature_indices].detach().clone()
        record = KeyframeDepthSeeds(
            frame=frame,
            feature_indices=feature_indices.detach().clone(),
            track_ids=track_ids,
            state=state,
            mu_range=1.0 / depth_min,
            updates=torch.zeros(count, dtype=torch.long, device=self.camera.device),
        )

        # A track can only be owned by its newest seed source.  In normal SVO
        # ordering inherited seeds are promoted before this method is called;
        # replacing here also keeps the state safe for custom callers.
        for slot, track_id in enumerate(track_ids.detach().cpu().tolist()):
            previous = self._track_lookup.pop(int(track_id), None)
            if previous is not None:
                previous_record = self.keyframes.get(previous[0])
                if previous_record is not None:
                    previous_record.updates[previous[1]] = self.max_updates
            self._track_lookup[int(track_id)] = (frame.id, slot)
        self.keyframes[frame.id] = record
        return count

    def discard_missing_keyframes(self, retained_frame_ids: set[int]) -> None:
        """Drop seed sources removed from the bounded local map."""

        for frame_id in list(self.keyframes):
            if frame_id in retained_frame_ids:
                continue
            record = self.keyframes.pop(frame_id)
            for track_id in record.track_ids.detach().cpu().tolist():
                location = self._track_lookup.get(int(track_id))
                if location is not None and location[0] == frame_id:
                    self._track_lookup.pop(int(track_id), None)

    def _location(self, track_id: int) -> tuple[KeyframeDepthSeeds, int] | None:
        location = self._track_lookup.get(int(track_id))
        if location is None:
            return None
        record = self.keyframes.get(location[0])
        if record is None:
            self._track_lookup.pop(int(track_id), None)
            return None
        return record, location[1]

    def point_for_track(self, track_id: int) -> Tensor | None:
        """Return the current seed mean as a detached world-space point."""

        location = self._location(track_id)
        if location is None:
            return None
        record, slot = location
        feature_index = int(record.feature_indices[slot])
        assert record.frame.features is not None
        inverse_depth = record.state.mu[slot]
        if not bool(torch.isfinite(inverse_depth)) or float(inverse_depth) <= 0:
            return None
        pixel = record.frame.features.pixels[feature_index : feature_index + 1]
        bearing = self.camera.unproject(pixel)[0]
        point_source = bearing / inverse_depth
        return record.frame.world_from_camera(point_source).detach()

    def update_observed(self, frame: Frame) -> int:
        """Fuse tracked two-view measurements after the current pose is optimized."""

        if frame.features is None or not self._track_lookup:
            return 0
        grouped: dict[int, list[tuple[int, int]]] = {}
        for current_index, track_id in enumerate(frame.features.track_ids.detach().cpu().tolist()):
            location = self._track_lookup.get(int(track_id))
            if location is None or location[0] == frame.id:
                continue
            record = self.keyframes.get(location[0])
            if record is None or int(record.updates[location[1]]) >= self.max_updates:
                continue
            grouped.setdefault(location[0], []).append((current_index, location[1]))

        successes = 0
        for source_id, pairs in grouped.items():
            record = self.keyframes[source_id]
            source = record.frame
            assert source.features is not None
            current_indices = torch.tensor(
                [pair[0] for pair in pairs], dtype=torch.long, device=self.camera.device
            )
            slots = torch.tensor(
                [pair[1] for pair in pairs], dtype=torch.long, device=self.camera.device
            )
            source_indices = record.feature_indices[slots]
            T_cur_ref = invert_transform(frame.T_world_cam) @ source.T_world_cam
            if float(torch.linalg.vector_norm(T_cur_ref[:3, 3])) <= 1e-5:
                continue

            bearing_ref = self.camera.unproject(source.features.pixels[source_indices])
            bearing_cur = self.camera.unproject(frame.features.pixels[current_indices])
            depth, triangulated = triangulate_depth(
                T_cur_ref.expand(len(pairs), -1, -1), bearing_ref, bearing_cur
            )
            points_ref = bearing_ref * depth[:, None]
            points_cur = transform_points(T_cur_ref, points_ref)
            reproj_ref, visible_ref = self.camera.project(points_ref)
            reproj_cur, visible_cur = self.camera.project(points_cur)
            error = torch.maximum(
                torch.linalg.vector_norm(
                    reproj_ref - source.features.pixels[source_indices], dim=-1
                ),
                torch.linalg.vector_norm(
                    reproj_cur - frame.features.pixels[current_indices], dim=-1
                ),
            )

            selected = record.state[slots]
            inverse_depth = depth.reciprocal()
            sigma = torch.sqrt(selected.sigma2.clamp_min(0))
            in_prior = (inverse_depth >= (selected.mu - sigma).clamp_min(1e-8)) & (
                inverse_depth <= selected.mu + sigma
            )
            depth_sigma = compute_tau(
                invert_transform(T_cur_ref).expand(len(pairs), -1, -1),
                bearing_ref,
                depth,
                self._pixel_error_angle,
            ).abs()
            tau2 = inverse_depth_variance(depth, depth_sigma)
            measurement_valid = (
                triangulated
                & visible_ref
                & visible_cur
                & in_prior
                & torch.isfinite(tau2)
                & (tau2 > 0)
                & (error <= self.reprojection_threshold)
            )
            safe_measurement = torch.where(measurement_valid, inverse_depth, selected.mu)
            safe_tau2 = torch.where(measurement_valid, tau2, torch.ones_like(tau2))
            updated, filter_valid = update_filter_vogiatzis(
                selected,
                safe_measurement,
                safe_tau2,
                record.mu_range,
            )
            accepted = measurement_valid & filter_valid
            record.state.mu[slots] = torch.where(accepted, updated.mu, selected.mu)
            record.state.sigma2[slots] = torch.where(accepted, updated.sigma2, selected.sigma2)
            record.state.a[slots] = torch.where(accepted, updated.a, selected.a)
            record.state.b[slots] = torch.where(accepted, updated.b, selected.b)
            record.updates[slots] += accepted.to(torch.long)
            successes += int(accepted.sum())
        return successes

    def is_converged(self, track_id: int) -> bool:
        location = self._location(track_id)
        if location is None:
            return False
        record, slot = location
        return bool(record.state[slot].converged(record.mu_range, self.convergence_threshold))

    def promote_observed(self, frame: Frame, sparse_map: SparseMap) -> int:
        """Upgrade successfully observed seeds before adding a new keyframe."""

        if frame.features is None:
            return 0
        promoted = 0
        for current_index, track_id in enumerate(frame.features.track_ids.detach().cpu().tolist()):
            if int(frame.features.landmark_ids[current_index]) != INVALID_ID:
                continue
            location = self._location(int(track_id))
            if location is None:
                continue
            record, slot = location
            if record.frame.id == frame.id or int(record.updates[slot]) < 1:
                continue
            source_index = int(record.feature_indices[slot])
            assert record.frame.features is not None
            existing_id = int(record.frame.features.landmark_ids[source_index])
            if existing_id in sparse_map.landmarks:
                landmark = sparse_map.landmarks[existing_id]
                landmark.add_observation(frame.id, current_index)
            else:
                position = self.point_for_track(int(track_id))
                if position is None:
                    continue
                landmark = sparse_map.create_landmark(
                    position,
                    observations={record.frame.id: source_index, frame.id: current_index},
                )
                record.frame.features.landmark_ids[source_index] = landmark.id
            frame.features.landmark_ids[current_index] = landmark.id
            self._track_lookup.pop(int(track_id), None)
            promoted += 1
        return promoted


__all__ = ["KeyframeDepthSeeds", "SynchronousDepthFilter"]
