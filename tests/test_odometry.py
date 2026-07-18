import runpy
from pathlib import Path

import pytest
import torch

from svo_torch.alignment import SparseAlignmentResult, TrackingResult
from svo_torch.camera import PinholeCamera
from svo_torch.frame import EDGELET, FeatureSet, Frame
from svo_torch.image import build_image_pyramid, prepare_image
from svo_torch.odometry import MonoSVO, Stage, TrackingQuality, UpdateResult

_EXAMPLE = runpy.run_path(Path(__file__).parents[1] / "examples" / "synthetic_sequence.py")
demo_config = _EXAMPLE["demo_config"]
make_synthetic_sequence = _EXAMPLE["make_synthetic_sequence"]


@pytest.fixture(scope="module")
def sequence():
    return make_synthetic_sequence(count=5)


def test_rendered_scene_is_deterministic_and_nonplanar(sequence) -> None:
    repeated = make_synthetic_sequence(count=5)
    torch.testing.assert_close(repeated.images, sequence.images, rtol=0.0, atol=0.0)
    torch.testing.assert_close(repeated.T_world_camera, sequence.T_world_camera, rtol=0.0, atol=0.0)
    assert float(sequence.points_world[:, 2].std()) > 0.5
    assert not torch.equal(sequence.images[0], sequence.images[-1])


def test_current_feature_gradients_are_sampled_at_their_pyramid_levels() -> None:
    dtype = torch.float64
    horizontal_ramp = torch.arange(32, dtype=dtype).repeat(32, 1) / 31.0
    pyramid = build_image_pyramid(horizontal_ramp, 3)
    pixels = torch.tensor([[10.0, 12.0], [18.0, 20.0]], dtype=dtype)
    levels = torch.tensor([0, 1], dtype=torch.long)
    stale_vertical = torch.tensor([[0.0, 2.0], [0.0, 3.0]], dtype=dtype)

    refreshed = MonoSVO._current_gradients(pyramid, pixels, levels, stale_vertical)

    torch.testing.assert_close(
        refreshed,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=dtype),
        rtol=0.0,
        atol=1e-12,
    )


def test_frontend_passes_camera_mask_to_all_detector_calls(monkeypatch) -> None:
    mask = torch.ones((24, 32), dtype=torch.uint8)
    mask[:, :16] = 0
    camera = PinholeCamera(32, 24, 20.0, 20.0, 16.0, 12.0, mask=mask)
    frontend = MonoSVO(camera, demo_config())
    image = torch.zeros((24, 32), device=frontend.device, dtype=frontend.dtype)
    frame = Frame(
        camera=frontend.camera,
        image=image,
        timestamp_ns=0,
        T_world_cam=torch.eye(4, device=frontend.device, dtype=frontend.dtype),
        pyramid=build_image_pyramid(image, frontend.config.image_pyramid_levels),
    )
    observed: list[torch.Tensor | None] = []

    def detect(pyramid, detector_mask):
        assert pyramid is frame.pyramid
        observed.append(detector_mask)
        return FeatureSet.empty(device=frontend.device, dtype=frontend.dtype)

    monkeypatch.setattr(frontend.detector, "detect", detect)
    frontend._detect(frame)
    frontend._augment_keyframe_features(frame)

    assert len(observed) == 2
    assert all(mask is frontend.camera.mask for mask in observed)


def test_initializer_uses_full_grid_and_carries_forward_klt_estimates(
    sequence, monkeypatch
) -> None:
    config = demo_config()
    config.init_min_features = 8
    config.init_min_tracked = 8
    config.init_min_inliers = 8
    config.init_min_disparity = 1_000.0
    frontend = MonoSVO(sequence.camera, config)

    expected_capacity = ((frontend.camera.width + config.grid_size - 1) // config.grid_size) * (
        (frontend.camera.height + config.grid_size - 1) // config.grid_size
    )
    assert frontend.detector.max_features == config.max_features
    assert frontend.initializer_detector.max_features == expected_capacity

    frontend.start()
    frontend.process(sequence.images[0], sequence.timestamp(0))
    assert frontend._initial_reference is not None
    assert frontend._initial_reference.features is not None
    detected = frontend._initial_reference.features.pixels.clone()
    assert len(detected) > config.init_min_features

    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    delta = detected.new_tensor([0.75, -0.25])

    def track(ref_pyramid, cur_pyramid, pixels_ref, pixels_initial):
        del ref_pyramid, cur_pyramid
        status = torch.ones(pixels_ref.shape[0], dtype=torch.bool, device=pixels_ref.device)
        if not calls:
            status[-3:] = False
        calls.append((pixels_ref.clone(), pixels_initial.clone(), status.clone()))
        return TrackingResult(
            pixels=pixels_initial + delta,
            status=status,
            error=torch.zeros(
                pixels_ref.shape[0], device=pixels_ref.device, dtype=pixels_ref.dtype
            ),
            iterations=torch.ones(pixels_ref.shape[0], device=pixels_ref.device, dtype=torch.long),
        )

    monkeypatch.setattr(frontend.tracker, "track", track)
    frontend.process(sequence.images[1], sequence.timestamp(1))
    frontend.process(sequence.images[2], sequence.timestamp(2))

    first_ref, first_initial, first_status = calls[0]
    second_ref, second_initial, _ = calls[1]
    torch.testing.assert_close(first_ref, detected)
    torch.testing.assert_close(first_initial, detected)
    torch.testing.assert_close(second_ref, detected[first_status])
    torch.testing.assert_close(second_initial, (detected + delta)[first_status])


def test_visual_only_pose_prediction_matches_original_zero_motion_prior(sequence) -> None:
    frontend = MonoSVO(sequence.camera, demo_config())
    frontend.start()
    frontend.process(sequence.images[0], sequence.timestamp(0))
    initialized = frontend.process(sequence.images[1], sequence.timestamp(1))

    assert initialized.stage == Stage.TRACKING
    assert frontend.last_frame is not None
    assert frontend._last_motion is not None
    assert float(torch.linalg.vector_norm(frontend._last_motion[:3, 3])) > 0.0
    torch.testing.assert_close(frontend._predicted_pose(), frontend.last_frame.T_world_cam)


def test_unusable_sparse_alignment_does_not_replace_motion_prior(sequence, monkeypatch) -> None:
    frontend = MonoSVO(sequence.camera, demo_config())
    frontend.start()
    frontend.process(sequence.images[0], sequence.timestamp(0))
    frontend.process(sequence.images[1], sequence.timestamp(1))
    reference = frontend.last_frame
    assert reference is not None and reference.features is not None

    image = prepare_image(sequence.images[2], device=frontend.device, dtype=frontend.dtype)
    prior = frontend._predicted_pose()
    frame = Frame(
        camera=frontend.camera,
        image=image,
        timestamp_ns=sequence.timestamp(2),
        T_world_cam=prior.clone(),
        pyramid=build_image_pyramid(image, frontend.config.image_pyramid_levels),
    )
    bad_transform = torch.eye(4, device=frontend.device, dtype=frontend.dtype)
    bad_transform[0, 3] = 5.0
    count = len(reference.features)

    for converged, error in ((False, 0.001), (True, 1.0)):
        result = SparseAlignmentResult(
            T_cur_ref=bad_transform,
            valid=torch.ones(count, dtype=torch.bool, device=frontend.device),
            error=error,
            errors=torch.full((count,), error, device=frontend.device, dtype=frontend.dtype),
            iterations=1,
            converged=converged,
        )

        def align(*args, _result=result, **kwargs):
            torch.testing.assert_close(
                args[4],
                torch.eye(4, device=frontend.device, dtype=frontend.dtype),
                rtol=0.0,
                atol=0.0,
            )
            return _result

        monkeypatch.setattr(frontend.aligner, "align", align)
        frame.T_world_cam = prior.clone()
        frontend._direct_pose_prior(reference, frame)
        torch.testing.assert_close(frame.T_world_cam, prior)


def test_local_map_recovers_older_keyframe_landmarks_with_few_last_tracks(
    sequence, monkeypatch
) -> None:
    frontend = MonoSVO(sequence.camera, demo_config())
    frontend.start()
    frontend.process(sequence.images[0], sequence.timestamp(0))
    frontend.process(sequence.images[1], sequence.timestamp(1))

    keyframes = list(frontend.map.keyframes.values())
    assert len(keyframes) == 2
    older_keyframe, reference = keyframes
    assert frontend.last_frame is reference
    assert older_keyframe.features is not None and reference.features is not None

    # Simulate a severe last-frame dropout while retaining the older keyframe
    # and its landmark observations. Five propagated tracks alone are below the
    # old early-failure threshold and cannot satisfy pose optimization.
    landmark_indices = torch.nonzero(reference.features.landmark_ids >= 0, as_tuple=False).squeeze(
        -1
    )
    assert landmark_indices.numel() > 6
    center = reference.features.pixels.new_tensor(
        [reference.camera.width / 2.0, reference.camera.height / 2.0]
    )
    central_order = torch.argsort(
        torch.linalg.vector_norm(reference.features.pixels[landmark_indices] - center, dim=-1)
    )
    target_index = int(landmark_indices[central_order[0]])
    kept_indices = landmark_indices[central_order[1:6]]
    target_landmark = int(reference.features.landmark_ids[target_index])

    for landmark in frontend.map.landmarks.values():
        landmark.observations.pop(reference.id, None)
    reference.features = reference.features[kept_indices]
    for feature_index, landmark_id in enumerate(
        reference.features.landmark_ids.detach().cpu().tolist()
    ):
        frontend.map.landmarks[int(landmark_id)].add_observation(reference.id, feature_index)
    assert len(reference.features) == 5

    original_track = frontend.tracker.track
    older_keyframe_calls = 0

    def record_tracking_source(ref_pyramid, *args, **kwargs):
        nonlocal older_keyframe_calls
        if ref_pyramid is older_keyframe.pyramid:
            older_keyframe_calls += 1
        return original_track(ref_pyramid, *args, **kwargs)

    monkeypatch.setattr(frontend.tracker, "track", record_tracking_source)
    result = frontend.process(sequence.images[2], sequence.timestamp(2))

    assert older_keyframe_calls > 0
    assert result.stage == Stage.TRACKING
    assert result.quality == TrackingQuality.GOOD
    assert result.num_observations >= frontend.config.quality_min_features
    assert "recovered" in result.message
    assert frontend.last_frame is not None and frontend.last_frame.features is not None
    final_features = frontend.last_frame.features
    final_landmarks = [
        int(value)
        for value in final_features.landmark_ids.detach().cpu().tolist()
        if int(value) >= 0
    ]
    final_tracks = [int(value) for value in final_features.track_ids.detach().cpu().tolist()]
    assert target_landmark in final_landmarks
    assert len(final_features) <= frontend.config.max_features
    assert len(final_landmarks) == len(set(final_landmarks))
    assert len(final_tracks) == len(set(final_tracks))


def test_structure_optimization_is_bounded_and_least_recently_updated(
    sequence, monkeypatch
) -> None:
    config = demo_config()
    config.structure_optimization_max_points = 2
    frontend = MonoSVO(sequence.camera, config)
    frontend.start()
    frontend.process(sequence.images[0], sequence.timestamp(0))
    frontend.process(sequence.images[1], sequence.timestamp(1))
    frame = frontend.last_frame
    assert frame is not None and frame.features is not None

    eligible: list[tuple[int, int]] = []
    for feature_index, landmark_id in enumerate(frame.features.landmark_ids.tolist()):
        landmark = frontend.map.landmarks.get(int(landmark_id))
        if landmark is not None and len(landmark.observations) >= 2:
            eligible.append((feature_index, landmark.id))
    assert len(eligible) >= 4

    for landmark in frontend.map.landmarks.values():
        landmark.last_structure_optimization = 100
    edge_index, edge_id = eligible[0]
    first_index, first_id = eligible[1]
    second_index, second_id = eligible[2]
    _, later_id = eligible[3]
    frame.features.kinds[edge_index] = EDGELET
    frontend.map.landmarks[edge_id].last_structure_optimization = 0
    frontend.map.landmarks[first_id].last_structure_optimization = 0
    frontend.map.landmarks[second_id].last_structure_optimization = 0
    frontend.map.landmarks[later_id].last_structure_optimization = 5
    assert first_index < second_index

    landmark_by_tensor = {
        id(landmark.position_world): landmark.id for landmark in frontend.map.landmarks.values()
    }
    calls: list[tuple[int, int]] = []

    def record_optimization(point, poses, pixels, cameras, **kwargs):
        assert poses.shape[0] == pixels.shape[0] == len(cameras) == 2
        assert kwargs == {"max_iterations": 5, "huber_delta": config.pose_huber_delta}
        calls.append((landmark_by_tensor[id(point)], poses.shape[0]))
        return point + point.new_tensor([1e-4, 0.0, 0.0])

    monkeypatch.setattr("svo_torch.odometry.optimize_point", record_optimization)
    optimized = frontend._optimize_structure(frame)

    assert optimized == 2
    assert [landmark_id for landmark_id, _ in calls] == [first_id, second_id]
    assert frontend.map.landmarks[edge_id].last_structure_optimization == 0
    assert frontend.map.landmarks[later_id].last_structure_optimization == 5
    assert frontend.map.landmarks[first_id].last_structure_optimization == frame.id
    assert frontend.map.landmarks[second_id].last_structure_optimization == frame.id


def test_relocalization_uses_configured_trial_boundary(sequence) -> None:
    config = demo_config()
    config.relocalization_max_trials = 3
    frontend = MonoSVO(sequence.camera, config)
    frontend.start()
    image = prepare_image(sequence.images[0], device=frontend.device, dtype=frontend.dtype)
    frame = Frame(
        camera=frontend.camera,
        image=image,
        timestamp_ns=sequence.timestamp(0),
        T_world_cam=torch.eye(4, device=frontend.device, dtype=frontend.dtype),
        pyramid=build_image_pyramid(image, frontend.config.image_pyramid_levels),
    )

    first = frontend._tracking_failure(frame, "test failure")
    second = frontend._tracking_failure(frame, "test failure")
    assert first.stage == second.stage == Stage.RELOCALIZING
    assert frontend._relocalization_trials == 2

    boundary = frontend._tracking_failure(frame, "test failure")
    assert boundary.stage == Stage.INITIALIZING
    assert frontend.stage == Stage.INITIALIZING
    assert frontend._relocalization_trials == 0
    assert "restarting initialization" in boundary.message


def test_mono_svo_state_machine_end_to_end(sequence) -> None:
    frontend = MonoSVO(sequence.camera, demo_config())

    # A paused call is non-consuming: the same image and timestamp are valid
    # after start(), which is useful for event-loop integrations.
    paused = frontend.process(sequence.images[0], sequence.timestamp(0))
    assert paused.stage == Stage.PAUSED
    assert paused.quality == TrackingQuality.INSUFFICIENT
    assert paused.update == UpdateResult.DEFAULT
    assert paused.T_world_cam is None
    assert "start" in paused.message

    frontend.start()
    assert frontend.stage == Stage.INITIALIZING
    reference = frontend.process(sequence.images[0], sequence.timestamp(0))
    assert reference.stage == Stage.INITIALIZING
    assert reference.update == UpdateResult.DEFAULT
    assert reference.num_observations == 0
    assert "reference" in reference.message

    initialized = frontend.process(sequence.images[1], sequence.timestamp(1))
    assert initialized.stage == Stage.TRACKING
    assert initialized.quality == TrackingQuality.GOOD
    assert initialized.update == UpdateResult.KEYFRAME
    assert initialized.is_keyframe
    assert initialized.num_observations >= demo_config().init_min_inliers
    assert initialized.T_world_cam is not None
    assert initialized.sparse_points is not None
    assert initialized.sparse_points.shape[0] == initialized.num_observations
    # The initializer must recover a real depth-varying map rather than a
    # planar/homographic stand-in.
    assert float(initialized.sparse_points[:, 2].std()) > 0.2
    assert len(frontend.map) == 2

    # start() is idempotent once running; it must not silently discard a live
    # tracking state or begin a second initialization over the existing map.
    frontend.start()
    assert frontend.stage == Stage.TRACKING
    assert len(frontend.map) == 2

    tracked = frontend.process(sequence.images[2], sequence.timestamp(2))
    assert tracked.stage == Stage.TRACKING
    assert tracked.quality == TrackingQuality.GOOD
    assert tracked.update != UpdateResult.FAILURE
    assert tracked.is_keyframe == (tracked.update == UpdateResult.KEYFRAME)
    assert tracked.num_observations >= demo_config().quality_min_features
    assert tracked.T_world_cam is not None
    assert initialized.T_world_cam is not None
    assert float(tracked.T_world_cam[0, 3]) > float(initialized.T_world_cam[0, 3])

    # Rejected input does not mutate the frontend clock, so a corrected image
    # can be retried at exactly the same timestamp.
    with pytest.raises(ValueError, match="strictly increasing"):
        frontend.process(sequence.images[3], sequence.timestamp(2))
    with pytest.raises(ValueError, match="calibration expects"):
        frontend.process(sequence.images[3, :-1], sequence.timestamp(3))
    retried = frontend.process(sequence.images[3], sequence.timestamp(3))
    assert retried.stage == Stage.TRACKING
    assert retried.quality == TrackingQuality.GOOD
    assert retried.update != UpdateResult.FAILURE
    assert retried.is_keyframe == (retried.update == UpdateResult.KEYFRAME)

    frontend.reset()
    assert frontend.stage == Stage.PAUSED
    assert frontend.quality == TrackingQuality.INSUFFICIENT
    assert frontend.last_frame is None
    assert len(frontend.map) == 0
    assert len(frontend.map.landmarks) == 0

    # Reset clears timestamp history, and negative timestamps fail without
    # consuming the valid timestamp that follows.
    paused_again = frontend.process(sequence.images[0], sequence.timestamp(0))
    assert paused_again.stage == Stage.PAUSED
    frontend.start()
    with pytest.raises(ValueError, match="non-negative"):
        frontend.process(sequence.images[0], -1)
    restarted = frontend.process(sequence.images[0], sequence.timestamp(0))
    assert restarted.stage == Stage.INITIALIZING
    assert "reference" in restarted.message
