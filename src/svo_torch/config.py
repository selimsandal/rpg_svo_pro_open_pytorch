"""Configuration for the tensor-native SVO front end."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass(slots=True)
class SVOConfig:
    """Runtime options with conservative defaults for 640-752 px images.

    ``from_yaml`` accepts this port's flat files as well as the overlapping
    keys from SVO Pro's ROS parameter files.
    """

    max_features: int = 180
    grid_size: int = 30
    n_pyr_levels: int = 5
    detector_threshold: float = 0.01
    detector_edgelet_ratio: float = 0.25
    feature_border: int = 10

    quality_min_features: int = 50
    quality_max_feature_drop: int = 40

    init_min_features: int = 100
    init_min_tracked: int = 80
    init_min_inliers: int = 70
    init_min_disparity: float = 40.0
    init_min_parallax: float = 0.5
    init_map_scale: float = 1.0
    init_ransac_iterations: int = 256
    init_ransac_probability: float = 0.999

    pose_reprojection_threshold: float = 2.0
    pose_iterations: int = 10
    pose_huber_delta: float = 2.0
    structure_optimization_max_points: int = 20
    relocalization_max_trials: int = 50

    max_keyframes: int = 5
    reprojector_max_keyframes: int = 5
    keyframe_criterion: str = "FORWARD"
    keyframe_num_features_upper: int = 120
    keyframe_num_features_lower: int = 70
    keyframe_min_distance: float = 0.12
    keyframe_min_distance_metric: float = 0.5
    keyframe_min_angle_degrees: float = 5.0
    keyframe_min_disparity: float = 40.0
    keyframe_min_frames: int = 2
    # Retained for source compatibility with the first port. SVO Pro's
    # FORWARD selector uses the absolute upper/lower feature thresholds above.
    keyframe_min_tracked_ratio: float = 0.65

    patch_size: int = 8
    alignment_max_level: int = 4
    alignment_min_level: int = 2
    alignment_iterations: int = 10
    alignment_min_update: float = 0.03
    # Parsed from original parameter files for compatibility. The current
    # fixed-brightness aligner does not add affine illumination variables.
    alignment_estimate_gain: bool = False
    alignment_estimate_offset: bool = False
    use_sparse_image_alignment: bool = True

    # SVO Pro runs this filter in a worker by default. MonoSVO preserves the
    # seed lifecycle synchronously so frame ownership and tensor work stay bounded.
    epipolar_max_steps: int = 128
    epipolar_step: float = 0.7
    epipolar_max_score: float = 0.20
    seed_convergence_sigma2_thresh: float = 200.0
    seed_max_updates: int = 30

    device: str = "auto"
    dtype: str = "float32"
    deterministic: bool = True
    random_seed: int = 7

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SVOConfig:
        """Build options from native or original SVO parameter names."""

        aliases = {
            "max_fts": "max_features",
            "max_n_kfs": "max_keyframes",
            "reprojector_max_n_kfs": "reprojector_max_keyframes",
            "map_scale": "init_map_scale",
            "quality_min_fts": "quality_min_features",
            "quality_max_drop_fts": "quality_max_feature_drop",
            "init_min_disparity": "init_min_disparity",
            "poseoptim_thresh": "pose_reprojection_threshold",
            "structure_optimization_max_pts": "structure_optimization_max_points",
            "relocalization_max_trials": "relocalization_max_trials",
            "img_align_max_level": "alignment_max_level",
            "img_align_min_level": "alignment_min_level",
            "img_align_est_illumination_gain": "alignment_estimate_gain",
            "img_align_est_illumination_offset": "alignment_estimate_offset",
            "kfselect_criterion": "keyframe_criterion",
            "kfselect_numkfs_upper_thresh": "keyframe_num_features_upper",
            "kfselect_numkfs_lower_thresh": "keyframe_num_features_lower",
            "kfselect_min_dist": "keyframe_min_distance",
            "kfselect_min_dist_metric": "keyframe_min_distance_metric",
            "kfselect_min_angle": "keyframe_min_angle_degrees",
            "kfselect_min_disparity": "keyframe_min_disparity",
            "kfselect_min_num_frames_between_kfs": "keyframe_min_frames",
            "seed_convergence_sigma2_thresh": "seed_convergence_sigma2_thresh",
        }
        known = {item.name for item in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in values.items():
            target = aliases.get(key, key)
            if target in known:
                kwargs[target] = value
        config = cls(**kwargs)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> SVOConfig:
        return cls.from_yaml_files([path])

    @classmethod
    def from_yaml_files(cls, paths: list[str | Path] | tuple[str | Path, ...]) -> SVOConfig:
        """Layer ROS-style parameter files from left to right.

        SVO Pro launch files commonly load a base file such as ``pinhole.yaml``
        and then a camera-specific override such as ``fisheye.yaml``.  Repeated
        CLI ``--config`` options use this method to preserve that behavior.
        """

        if not paths:
            raise ValueError("at least one configuration path is required")
        combined: dict[str, Any] = {}
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as stream:
                values = yaml.safe_load(stream) or {}
            if not isinstance(values, Mapping):
                raise ValueError(f"configuration root must be a mapping: {path}")
            combined.update(values)
        return cls.from_mapping(combined)

    def validate(self) -> None:
        if self.max_features < 8:
            raise ValueError("max_features must be at least 8")
        if self.grid_size < 4:
            raise ValueError("grid_size must be at least 4")
        if self.n_pyr_levels < 1:
            raise ValueError("n_pyr_levels must be positive")
        if not 0 < self.detector_threshold <= 1:
            raise ValueError("detector_threshold must be in (0, 1]")
        if not 0 <= self.detector_edgelet_ratio <= 1:
            raise ValueError("detector_edgelet_ratio must be in [0, 1]")
        if self.feature_border < 0:
            raise ValueError("feature_border must be non-negative")
        if self.patch_size < 2 or self.patch_size % 2:
            raise ValueError("patch_size must be a positive even number")
        if not 0 <= self.alignment_min_level <= self.alignment_max_level:
            raise ValueError("image-alignment levels must form a non-negative range")
        if self.alignment_iterations < 1 or self.alignment_min_update <= 0:
            raise ValueError("alignment iterations and update threshold must be positive")
        if self.quality_min_features < 6:
            raise ValueError("quality_min_features must be at least 6")
        if self.quality_max_feature_drop < 0:
            raise ValueError("quality_max_feature_drop must be non-negative")
        if min(self.init_min_features, self.init_min_tracked, self.init_min_inliers) < 8:
            raise ValueError("monocular initialization thresholds must be at least 8")
        if self.init_min_tracked > self.init_min_features:
            raise ValueError("init_min_tracked cannot exceed init_min_features")
        if self.init_min_inliers > self.init_min_tracked:
            raise ValueError("init_min_inliers cannot exceed init_min_tracked")
        if self.init_min_disparity < 0 or self.init_min_parallax < 0:
            raise ValueError("initial disparity and parallax must be non-negative")
        if self.init_map_scale <= 0:
            raise ValueError("init_map_scale must be positive")
        if self.init_ransac_iterations < 1 or not 0 < self.init_ransac_probability < 1:
            raise ValueError("invalid RANSAC iterations or probability")
        if self.pose_reprojection_threshold <= 0:
            raise ValueError("pose_reprojection_threshold must be positive")
        if self.pose_iterations < 1 or self.pose_huber_delta <= 0:
            raise ValueError("pose iterations and Huber delta must be positive")
        if self.structure_optimization_max_points < -1:
            raise ValueError("structure_optimization_max_points must be -1 or non-negative")
        if self.relocalization_max_trials < 1:
            raise ValueError("relocalization_max_trials must be positive")
        if self.max_keyframes < 0 or self.keyframe_min_frames < 0:
            raise ValueError("keyframe counts must be non-negative")
        if self.reprojector_max_keyframes < 1:
            raise ValueError("reprojector_max_keyframes must be positive")
        self.keyframe_criterion = self.keyframe_criterion.upper()
        if self.keyframe_criterion not in {"FORWARD", "DOWNLOOKING"}:
            raise ValueError("keyframe_criterion must be FORWARD or DOWNLOOKING")
        if not 0 <= self.keyframe_num_features_lower <= self.keyframe_num_features_upper:
            raise ValueError("keyframe feature thresholds must satisfy 0 <= lower <= upper")
        if (
            self.keyframe_min_distance < 0
            or self.keyframe_min_distance_metric < 0
            or self.keyframe_min_angle_degrees < 0
            or self.keyframe_min_disparity < 0
        ):
            raise ValueError("keyframe motion and disparity thresholds must be non-negative")
        if not 0 < self.keyframe_min_tracked_ratio <= 1:
            raise ValueError("keyframe_min_tracked_ratio must be in (0, 1]")
        if self.epipolar_max_steps < 2 or self.epipolar_step <= 0:
            raise ValueError("invalid epipolar search settings")
        if self.epipolar_max_score <= 0 or self.seed_convergence_sigma2_thresh <= 0:
            raise ValueError("depth-filter thresholds must be positive")
        if self.seed_max_updates < 1:
            raise ValueError("seed_max_updates must be positive")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available() and self.dtype != "float64":
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if device.type == "mps" and self.dtype == "float64":
            raise RuntimeError("MPS does not support float64; select float32 or another device")
        return device

    @property
    def image_pyramid_levels(self) -> int:
        """Levels needed by both detection and coarse image alignment."""

        return max(self.n_pyr_levels, self.alignment_max_level + 1)

    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32
