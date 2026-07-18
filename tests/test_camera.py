from pathlib import Path

import pytest
import torch

from svo_torch.camera import CameraRig, OmniCamera, PinholeCamera, load_camera_rig

DTYPE = torch.float64
OMNI_PARAMETERS = [
    -69.6915,
    0.0,
    5.4772e-4,
    2.1371e-5,
    -8.7523e-9,
    320.0,
    240.0,
    1.0,
    0.0,
    0.0,
    142.7468,
    104.8486,
    7.3973,
    17.4581,
    12.6308,
    -4.3751,
    6.9093,
    10.9703,
    -0.6053,
    -3.9119,
    -1.0675,
    0.0,
    0.0,
    1.0,
]


def test_pinhole_project_uses_xy_pixels_and_validity() -> None:
    camera = PinholeCamera(640, 480, 400.0, 410.0, 320.0, 240.0, dtype=DTYPE)
    points = torch.tensor(
        [[0.0, 0.0, 2.0], [1.0, 0.5, 2.0], [0.0, 0.0, -1.0], [10.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    pixels, valid = camera.project(points)
    torch.testing.assert_close(
        pixels[:2], torch.tensor([[320.0, 240.0], [520.0, 342.5]], dtype=DTYPE)
    )
    assert valid.tolist() == [True, True, False, False]
    assert camera.is_in_frame(torch.tensor([[639.9, 479.9], [640.0, 10.0]])).tolist() == [
        True,
        False,
    ]


@pytest.mark.parametrize(
    ("distortion", "parameters", "tolerance"),
    [
        ("none", [], 1e-12),
        ("radial-tangential", [-0.12, 0.03, 0.001, -0.002], 2e-8),
        ("equidistant", [-0.01, 0.02, -0.005, 0.001], 2e-9),
        ("fisheye(atan)", [0.8], 2e-10),
    ],
)
def test_pinhole_distortion_round_trip(
    distortion: str, parameters: list[float], tolerance: float
) -> None:
    camera = PinholeCamera(
        752,
        480,
        460.0,
        462.0,
        370.0,
        235.0,
        distortion=distortion,
        distortion_params=parameters,
        dtype=DTYPE,
    )
    points = torch.tensor([[0.15, -0.1, 1.0], [-0.35, 0.18, 1.2], [0.28, 0.22, 0.8]], dtype=DTYPE)
    expected = torch.nn.functional.normalize(points, dim=-1)
    pixels, valid = camera.project(points)
    assert valid.all()
    recovered = camera.unproject(pixels)
    torch.testing.assert_close(recovered, expected, atol=tolerance, rtol=tolerance)


def test_pinhole_project_is_differentiable_and_to_scale_preserve_calibration() -> None:
    camera = PinholeCamera(
        640,
        480,
        400.0,
        410.0,
        320.0,
        240.0,
        distortion="radtan",
        distortion_params=[-0.1, 0.02, 0.001, 0.0],
        dtype=DTYPE,
    )
    point = torch.tensor([0.2, -0.1, 1.5], dtype=DTYPE, requires_grad=True)
    pixels, valid = camera.project(point)
    assert bool(valid)
    pixels.sum().backward()
    assert point.grad is not None and torch.isfinite(point.grad).all()

    half = camera.scale(0.5)
    assert half.image_size == (320, 240)
    torch.testing.assert_close(half.intrinsics, camera.intrinsics * 0.5)
    single = half.to(dtype=torch.float32)
    assert single.dtype == torch.float32
    assert single.distortion == "radial-tangential"


def test_omni_24_parameter_projection_round_trip_mask_and_scale() -> None:
    camera = OmniCamera(752, 480, OMNI_PARAMETERS, dtype=DTYPE)
    pixels = torch.tensor(
        [[320.0, 240.0], [400.0, 240.0], [320.0, 300.0], [200.0, 200.0]], dtype=DTYPE
    )
    bearings = camera.unproject(pixels)
    torch.testing.assert_close(
        torch.linalg.vector_norm(bearings, dim=-1), torch.ones(4, dtype=DTYPE)
    )
    recovered, valid = camera.project(bearings)
    assert valid.all()
    torch.testing.assert_close(recovered, pixels, atol=4e-3, rtol=0.0)

    half = camera.scale(0.5)
    half_pixels, half_valid = half.project(bearings)
    assert half_valid.all()
    torch.testing.assert_close(half_pixels, pixels * 0.5, atol=2e-3, rtol=0.0)

    masked_parameters = OMNI_PARAMETERS.copy()
    masked_parameters[-2:] = [0.2, 0.4]
    masked = OmniCamera(752, 480, masked_parameters, dtype=DTYPE)
    _, center_valid = masked.project(masked.unproject(torch.tensor([320.0, 240.0], dtype=DTYPE)))
    assert not bool(center_valid)


def test_camera_rig_exposes_camera_to_body_transforms() -> None:
    camera = PinholeCamera(32, 24, 20.0, 20.0, 16.0, 12.0, dtype=DTYPE)
    transforms = torch.eye(4, dtype=DTYPE).repeat(2, 1, 1)
    transforms[1, 0, 3] = 0.2
    rig = CameraRig([camera, camera], transforms, label="stereo")
    assert len(rig) == 2
    assert rig.cameras[0] is camera
    torch.testing.assert_close(rig.T_camera_body[1, 0, 3], torch.tensor(-0.2, dtype=DTYPE))
    assert rig.to(dtype=torch.float32).T_body_camera.dtype == torch.float32


def test_load_canonical_svo_rig_yaml_with_t_body_camera(tmp_path: Path) -> None:
    calibration = tmp_path / "rig.yaml"
    calibration.write_text(
        """
label: test-rig
cameras:
  - camera:
      type: pinhole
      label: cam0
      image_width: 640
      image_height: 480
      intrinsics: {rows: 4, cols: 1, data: [400, 401, 320, 240]}
      distortion:
        type: radial-tangential
        parameters: {rows: 4, cols: 1, data: [-0.1, 0.02, 0.001, -0.001]}
    T_B_C:
      rows: 4
      cols: 4
      data: [1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  - camera:
      type: pinhole
      label: cam1
      image_width: 640
      image_height: 480
      intrinsics: {rows: 4, cols: 1, data: [400, 401, 320, 240]}
      distortion: {type: none, parameters: {rows: 0, cols: 1, data: []}}
    T_B_C:
      rows: 4
      cols: 4
      data: [1, 0, 0, -0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
""",
        encoding="utf-8",
    )
    rig = load_camera_rig(calibration, dtype=DTYPE)
    assert rig.label == "test-rig"
    assert len(rig.cameras) == 2
    assert rig.cameras[0].label == "cam0"
    assert isinstance(rig.cameras[0], PinholeCamera)
    assert rig.cameras[0].distortion == "radial-tangential"
    torch.testing.assert_close(rig.T_body_camera[:, 0, 3], torch.tensor([0.1, -0.1], dtype=DTYPE))
    torch.testing.assert_close(
        rig.T_body_camera[:, 3, :], torch.tensor([[0, 0, 0, 1]] * 2, dtype=DTYPE)
    )


def test_invalid_camera_calibration_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires 4 parameters"):
        PinholeCamera(
            640,
            480,
            400.0,
            400.0,
            320.0,
            240.0,
            distortion="equidistant",
            distortion_params=[0.1],
        )
    with pytest.raises(ValueError, match="24 parameters"):
        OmniCamera(640, 480, [0.0] * 23)


def test_pixel_error_angle_and_odd_dimension_scaling() -> None:
    camera = PinholeCamera(7, 5, 260.0, 250.0, 3.0, 2.0, dtype=DTYPE)
    expected = torch.atan(torch.tensor(1.0 / 520.0, dtype=DTYPE)) + torch.atan(
        torch.tensor(1.0 / 500.0, dtype=DTYPE)
    )
    torch.testing.assert_close(camera.pixel_error_angle(1.0), expected)
    scaled = camera.scale(0.5)
    assert scaled.image_size == (3, 2)


def test_camera_rig_rejects_non_rigid_transforms() -> None:
    camera = PinholeCamera(32, 24, 20.0, 20.0, 16.0, 12.0, dtype=DTYPE)
    transform = torch.eye(4, dtype=DTYPE)[None]
    transform[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        CameraRig([camera], transform)
    transform = torch.eye(4, dtype=DTYPE)[None]
    transform[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        CameraRig([camera], transform)
