import torch

from svo_torch.camera import PinholeCamera
from svo_torch.config import SVOConfig
from svo_torch.depth_filter import SynchronousDepthFilter
from svo_torch.frame import INVALID_ID, FeatureSet, Frame, SparseMap
from svo_torch.odometry import MonoSVO


def _features(pixels: torch.Tensor, track_ids: torch.Tensor) -> FeatureSet:
    count = pixels.shape[0]
    return FeatureSet(
        pixels=pixels,
        scores=torch.ones(count, dtype=pixels.dtype, device=pixels.device),
        levels=torch.zeros(count, dtype=torch.long, device=pixels.device),
        gradients=torch.tensor([1.0, 0.0], dtype=pixels.dtype, device=pixels.device).expand(
            count, -1
        ),
        kinds=torch.zeros(count, dtype=torch.long, device=pixels.device),
        track_ids=track_ids,
        landmark_ids=torch.full((count,), INVALID_ID, dtype=torch.long, device=pixels.device),
    )


def _frame(
    camera: PinholeCamera,
    pixels: torch.Tensor,
    track_ids: torch.Tensor,
    pose: torch.Tensor,
    timestamp_ns: int,
) -> Frame:
    return Frame(
        camera=camera,
        image=torch.zeros((camera.height, camera.width), dtype=camera.dtype),
        timestamp_ns=timestamp_ns,
        T_world_cam=pose,
        pyramid=[],
        features=_features(pixels, track_ids),
    )


def test_synchronous_depth_filter_updates_and_promotes_observed_seed() -> None:
    dtype = torch.float64
    camera = PinholeCamera(96, 72, 80.0, 80.0, 47.5, 35.5, dtype=dtype)
    source_pixel = torch.tensor([[44.0, 34.0]], dtype=dtype)
    track_ids = torch.tensor([17], dtype=torch.long)
    source = _frame(camera, source_pixel, track_ids, torch.eye(4, dtype=dtype), 0)
    depth_filter = SynchronousDepthFilter(camera, max_updates=10)

    assert depth_filter.add_keyframe(source, torch.tensor([4.0], dtype=dtype)) == 1
    assert depth_filter.active_count == 1
    initial_point = depth_filter.point_for_track(17)
    assert initial_point is not None
    torch.testing.assert_close(
        torch.linalg.vector_norm(initial_point), torch.tensor(4.0, dtype=dtype)
    )

    current_pose = torch.eye(4, dtype=dtype)
    current_pose[0, 3] = 0.2
    point_cur = initial_point - current_pose[:3, 3]
    current_pixel, visible = camera.project(point_cur[None])
    assert visible.item()
    current = _frame(camera, current_pixel, track_ids, current_pose, 1)

    assert depth_filter.update_observed(current) == 1
    record = depth_filter.keyframes[source.id]
    assert int(record.updates[0]) == 1
    assert float(record.state.sigma2[0]) < (record.mu_range**2) / 36.0

    sparse_map = SparseMap(max_keyframes=5)
    sparse_map.add_keyframe(source)
    assert depth_filter.promote_observed(current, sparse_map) == 1
    landmark_id = int(current.features.landmark_ids[0])
    assert landmark_id >= 0
    assert int(source.features.landmark_ids[0]) == landmark_id
    assert sparse_map.landmarks[landmark_id].observations == {source.id: 0, current.id: 0}
    assert depth_filter.active_count == 0


def test_synchronous_depth_filter_bounds_updates_and_discards_old_sources() -> None:
    dtype = torch.float64
    camera = PinholeCamera(64, 48, 55.0, 55.0, 31.5, 23.5, dtype=dtype)
    pixels = torch.tensor([[25.0, 20.0], [38.0, 28.0]], dtype=dtype)
    tracks = torch.tensor([3, 4], dtype=torch.long)
    source = _frame(camera, pixels, tracks, torch.eye(4, dtype=dtype), 0)
    depth_filter = SynchronousDepthFilter(camera, max_updates=1)
    assert depth_filter.add_keyframe(source, torch.tensor([], dtype=dtype), fallback_depth=3.0) == 2

    current_pose = torch.eye(4, dtype=dtype)
    current_pose[0, 3] = 0.15
    points = camera.unproject(pixels) * 3.0
    current_pixels, visible = camera.project(points - current_pose[:3, 3])
    assert visible.all()
    current = _frame(camera, current_pixels, tracks, current_pose, 1)

    assert depth_filter.update_observed(current) == 2
    assert depth_filter.update_observed(current) == 0
    depth_filter.discard_missing_keyframes(set())
    assert depth_filter.active_count == 0
    assert depth_filter.point_for_track(3) is None


def test_keyframe_landmark_count_excludes_pose_constraining_seeds() -> None:
    dtype = torch.float32
    camera = PinholeCamera(64, 48, 55.0, 55.0, 31.5, 23.5, dtype=dtype)
    frontend = MonoSVO(
        camera,
        SVOConfig(
            max_features=20,
            grid_size=12,
            quality_min_features=6,
            init_min_features=8,
            init_min_tracked=8,
            init_min_inliers=8,
            device="cpu",
        ),
    )
    pixels = torch.tensor([[20.0, 20.0], [30.0, 22.0], [40.0, 25.0]], dtype=dtype)
    tracks = torch.tensor([0, 1, 2], dtype=torch.long)
    frame = _frame(camera, pixels, tracks, torch.eye(4, dtype=dtype), 0)
    first = frontend.map.create_landmark(torch.tensor([-0.5, 0.0, 3.0]))
    second = frontend.map.create_landmark(torch.tensor([0.0, 0.0, 3.0]))
    frame.features.landmark_ids[:2] = torch.tensor([first.id, second.id])
    assert frontend.depth_filter.add_keyframe(frame, torch.tensor([3.0])) == 1

    indices, _ = frontend._landmark_points(frame.features)
    assert indices.numel() == 3
    # SVO Pro uses numTrackedLandmarks() for keyframe thresholds even though
    # both landmarks and seed references are measurements in pose optimization.
    assert frontend._num_tracked_landmarks(frame.features) == 2
