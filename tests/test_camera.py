from pathlib import Path

import pytest
import torch
from PIL import Image

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


def test_pinhole_mask_uses_cpp_indexing_and_does_not_change_projection_validity() -> None:
    mask = torch.zeros((3, 4), dtype=torch.uint8)
    mask[1, 2] = 17
    camera = PinholeCamera(4, 3, 1.0, 1.0, 0.0, 0.0, mask=mask, dtype=DTYPE)

    pixels = torch.tensor(
        [[2.9, 1.9], [1.9, 1.9], [-0.1, 1.0], [4.0, 1.0]],
        dtype=DTYPE,
    )
    assert camera.is_masked(pixels).tolist() == [False, True, True, True]
    assert camera.mask is not None
    assert camera.mask.dtype == torch.uint8
    assert camera.mask[1, 2].item() == 1

    # Camera masks constrain feature detection in SVO, not geometric projection.
    masked_bearing = camera.unproject(torch.tensor([1.0, 1.0], dtype=DTYPE))
    _, valid = camera.project(masked_bearing)
    assert bool(valid)


def test_camera_mask_is_preserved_by_to_and_scale() -> None:
    mask = torch.zeros((4, 4), dtype=torch.uint8)
    mask[:, 2:] = 255
    camera = PinholeCamera(4, 4, 2.0, 2.0, 2.0, 2.0, mask=mask, dtype=DTYPE)

    converted = camera.to(dtype=torch.float32)
    assert converted.mask is not None
    assert converted.mask.dtype == torch.uint8
    torch.testing.assert_close(converted.mask, camera.mask)

    scaled = camera.scale(0.5)
    assert scaled.mask is not None
    assert scaled.mask.shape == (2, 2)
    torch.testing.assert_close(scaled.mask, torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8))


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
    assert bool(center_valid)
    assert masked.mask is not None
    assert bool(masked.is_masked(torch.tensor([320.0, 240.0], dtype=DTYPE)))


def test_load_camera_mask_relative_to_calibration_and_override_omni_annulus(
    tmp_path: Path,
) -> None:
    calibration_directory = tmp_path / "calibration"
    mask_directory = calibration_directory / "masks"
    mask_directory.mkdir(parents=True)
    mask_path = mask_directory / "override.png"
    bitmap = Image.new("L", (8, 6), 0)
    bitmap.putpixel((3, 2), 255)
    bitmap.save(mask_path)

    parameters = OMNI_PARAMETERS.copy()
    parameters[5:7] = [3.8, 2.9]
    parameters[-2:] = [0.2, 0.8]
    calibration = calibration_directory / "camera.yaml"
    calibration.write_text(
        f"""
type: omni
label: masked-omni
image_width: 8
image_height: 6
intrinsics: {{rows: 24, cols: 1, data: {parameters}}}
mask: masks/override.png
""",
        encoding="utf-8",
    )

    camera = load_camera_rig(calibration, dtype=DTYPE).cameras[0]
    assert isinstance(camera, OmniCamera)
    assert camera.mask is not None
    assert camera.mask.sum().item() == 1
    assert not bool(camera.is_masked(torch.tensor([3.8, 2.9], dtype=DTYPE)))
    # The center is outside the embedded annulus, proving the bitmap replaced it.
    generated = OmniCamera(8, 6, parameters, dtype=DTYPE)
    assert bool(generated.is_masked(torch.tensor([3.8, 2.9], dtype=DTYPE)))


def test_camera_mask_load_failures_are_clear(tmp_path: Path) -> None:
    calibration = tmp_path / "camera.yaml"

    def write_calibration(mask_name: str) -> None:
        calibration.write_text(
            f"""
type: pinhole
image_width: 4
image_height: 3
intrinsics: {{rows: 4, cols: 1, data: [2, 2, 2, 1]}}
distortion: {{type: none, parameters: {{rows: 0, cols: 1, data: []}}}}
mask: {mask_name}
""",
            encoding="utf-8",
        )

    write_calibration("missing.png")
    with pytest.raises(FileNotFoundError, match="camera mask file does not exist"):
        load_camera_rig(calibration)

    unreadable = tmp_path / "unreadable.png"
    unreadable.write_bytes(b"not an image")
    write_calibration(unreadable.name)
    with pytest.raises(ValueError, match="unable to read camera mask file"):
        load_camera_rig(calibration)

    wrong_size = tmp_path / "wrong-size.png"
    Image.new("L", (3, 2), 255).save(wrong_size)
    write_calibration(wrong_size.name)
    with pytest.raises(ValueError, match="calibration expects 4x3"):
        load_camera_rig(calibration)


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
