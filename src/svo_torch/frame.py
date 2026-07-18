"""Frames, features, landmarks, and the bounded local sparse map."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from .geometry import invert_transform, transform_points

if TYPE_CHECKING:
    from .camera import Camera


CORNER = 0
EDGELET = 1
INVALID_ID = -1


@dataclass(slots=True)
class FeatureSet:
    """Parallel feature arrays using level-zero ``(x, y)`` coordinates."""

    pixels: Tensor
    scores: Tensor
    levels: Tensor
    gradients: Tensor
    kinds: Tensor
    track_ids: Tensor
    landmark_ids: Tensor

    def __post_init__(self) -> None:
        n = self.pixels.shape[0]
        if self.pixels.ndim != 2 or self.pixels.shape[1] != 2:
            raise ValueError("pixels must have shape [N, 2]")
        expected = {
            "scores": (n,),
            "levels": (n,),
            "gradients": (n, 2),
            "kinds": (n,),
            "track_ids": (n,),
            "landmark_ids": (n,),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")

    @classmethod
    def empty(
        cls,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> FeatureSet:
        return cls(
            pixels=torch.empty((0, 2), device=device, dtype=dtype),
            scores=torch.empty(0, device=device, dtype=dtype),
            levels=torch.empty(0, device=device, dtype=torch.long),
            gradients=torch.empty((0, 2), device=device, dtype=dtype),
            kinds=torch.empty(0, device=device, dtype=torch.long),
            track_ids=torch.empty(0, device=device, dtype=torch.long),
            landmark_ids=torch.empty(0, device=device, dtype=torch.long),
        )

    @classmethod
    def from_detection(
        cls,
        pixels: Tensor,
        scores: Tensor,
        levels: Tensor,
        gradients: Tensor,
        kinds: Tensor | None = None,
        *,
        first_track_id: int = 0,
    ) -> FeatureSet:
        n = pixels.shape[0]
        device = pixels.device
        if kinds is None:
            kinds = torch.zeros(n, dtype=torch.long, device=device)
        return cls(
            pixels=pixels,
            scores=scores,
            levels=levels.to(device=device, dtype=torch.long),
            gradients=gradients,
            kinds=kinds.to(device=device, dtype=torch.long),
            track_ids=torch.arange(first_track_id, first_track_id + n, device=device),
            landmark_ids=torch.full((n,), INVALID_ID, dtype=torch.long, device=device),
        )

    def __len__(self) -> int:
        return self.pixels.shape[0]

    def __getitem__(self, index: Tensor | slice | list[int]) -> FeatureSet:
        return FeatureSet(
            pixels=self.pixels[index],
            scores=self.scores[index],
            levels=self.levels[index],
            gradients=self.gradients[index],
            kinds=self.kinds[index],
            track_ids=self.track_ids[index],
            landmark_ids=self.landmark_ids[index],
        )

    def to(self, *args: object, **kwargs: object) -> FeatureSet:
        pixels = self.pixels.to(*args, **kwargs)
        device = pixels.device
        return FeatureSet(
            pixels=pixels,
            scores=self.scores.to(device=device, dtype=pixels.dtype),
            levels=self.levels.to(device=device),
            gradients=self.gradients.to(device=device, dtype=pixels.dtype),
            kinds=self.kinds.to(device=device),
            track_ids=self.track_ids.to(device=device),
            landmark_ids=self.landmark_ids.to(device=device),
        )

    @staticmethod
    def concatenate(parts: Iterable[FeatureSet]) -> FeatureSet:
        values = list(parts)
        if not values:
            return FeatureSet.empty()
        return FeatureSet(
            pixels=torch.cat([item.pixels for item in values]),
            scores=torch.cat([item.scores for item in values]),
            levels=torch.cat([item.levels for item in values]),
            gradients=torch.cat([item.gradients for item in values]),
            kinds=torch.cat([item.kinds for item in values]),
            track_ids=torch.cat([item.track_ids for item in values]),
            landmark_ids=torch.cat([item.landmark_ids for item in values]),
        )


@dataclass(slots=True)
class Frame:
    """A grayscale tensor frame and its current world pose."""

    _counter: ClassVar[int] = 0

    camera: Camera
    image: Tensor
    timestamp_ns: int
    T_world_cam: Tensor
    pyramid: list[Tensor] = field(default_factory=list)
    features: FeatureSet | None = None
    id: int = -1
    is_keyframe: bool = False

    def __post_init__(self) -> None:
        if self.id < 0:
            self.id = Frame._counter
            Frame._counter += 1
        if self.T_world_cam.shape != (4, 4):
            raise ValueError("T_world_cam must have shape [4, 4]")
        if self.features is None:
            self.features = FeatureSet.empty(device=self.image.device, dtype=self.image.dtype)

    @property
    def T_cam_world(self) -> Tensor:
        return invert_transform(self.T_world_cam)

    @property
    def position(self) -> Tensor:
        return self.T_world_cam[:3, 3]

    def world_from_camera(self, points_camera: Tensor) -> Tensor:
        return transform_points(self.T_world_cam, points_camera)

    def camera_from_world(self, points_world: Tensor) -> Tensor:
        return transform_points(self.T_cam_world, points_world)


@dataclass(slots=True)
class Landmark:
    id: int
    position_world: Tensor
    observations: dict[int, int] = field(default_factory=dict)
    quality: float = 1.0

    def add_observation(self, frame_id: int, feature_index: int) -> None:
        self.observations[frame_id] = feature_index


class SparseMap:
    """Insertion-ordered local keyframe and landmark map."""

    def __init__(self, max_keyframes: int = 10) -> None:
        self.max_keyframes = max_keyframes
        self.keyframes: OrderedDict[int, Frame] = OrderedDict()
        self.landmarks: dict[int, Landmark] = {}
        self._next_landmark_id = 0

    def __len__(self) -> int:
        return len(self.keyframes)

    def clear(self) -> None:
        self.keyframes.clear()
        self.landmarks.clear()
        self._next_landmark_id = 0

    def add_keyframe(self, frame: Frame) -> list[int]:
        frame.is_keyframe = True
        self.keyframes[frame.id] = frame
        removed: list[int] = []
        while self.max_keyframes > 0 and len(self.keyframes) > self.max_keyframes:
            old_id, _ = self.keyframes.popitem(last=False)
            removed.append(old_id)
        if removed:
            for landmark_id, landmark in list(self.landmarks.items()):
                for frame_id in removed:
                    landmark.observations.pop(frame_id, None)
                if not landmark.observations:
                    del self.landmarks[landmark_id]
        return removed

    def create_landmark(
        self,
        position_world: Tensor,
        observations: dict[int, int] | None = None,
    ) -> Landmark:
        landmark = Landmark(
            id=self._next_landmark_id,
            position_world=position_world.detach().clone(),
            observations={} if observations is None else dict(observations),
        )
        self.landmarks[landmark.id] = landmark
        self._next_landmark_id += 1
        return landmark

    def landmark_tensors(
        self,
        ids: Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        valid_cpu = [int(value) in self.landmarks for value in ids.detach().cpu().tolist()]
        valid = torch.tensor(valid_cpu, dtype=torch.bool, device=device)
        chosen = [
            self.landmarks[int(value)].position_world
            for value, keep in zip(ids.detach().cpu().tolist(), valid_cpu, strict=True)
            if keep
        ]
        if not chosen:
            return torch.empty((0, 3), dtype=dtype, device=device), valid
        points = torch.stack([point.to(device=device, dtype=dtype) for point in chosen])
        return points, valid

    def points_tensor(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        if not self.landmarks:
            return torch.empty((0, 3), dtype=dtype, device=device)
        return torch.stack(
            [
                landmark.position_world.to(device=device, dtype=dtype)
                for landmark in self.landmarks.values()
            ]
        )
