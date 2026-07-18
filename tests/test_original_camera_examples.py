"""Parity checks derived from the original SVO camera examples."""

from pathlib import Path

import torch

from svo_torch.camera import OmniCamera, PinholeCamera, load_camera_rig

DTYPE = torch.float64
FIXTURES = Path(__file__).parent / "data" / "original"

# The fixture files are copies from the sibling original checkout:
# - vikit/vikit_cameras/test/data/calib_omni.yaml
# - vikit/vikit_cameras/test/data/calib_cam.yaml
# - svo_ros/param/calib/davis_flyingroom.yaml
# The omni fixture omits its unresolved upstream ``test_mask.png`` reference;
# masks do not participate in projection or backprojection. Numerical
# expectations below come from
# vikit/vikit_cameras/test/test_cameras.cpp in that same checkout.


def test_original_omni_projection_and_backprojection_examples() -> None:
    camera = load_camera_rig(FIXTURES / "calib_omni.yaml", dtype=DTYPE).cameras[0]
    assert isinstance(camera, OmniCamera)

    bearing = camera.unproject(torch.tensor([400.0, 300.0], dtype=DTYPE))
    expected_bearing = torch.tensor(
        [0.733011271294813, 0.549758453471110, 0.400574735838165],
        dtype=DTYPE,
    )
    # The original test uses 1e-5.  A tighter 3e-6 still accommodates the
    # rounded polynomial coefficients serialized in the shared YAML fixture.
    torch.testing.assert_close(bearing, expected_bearing, rtol=0.0, atol=3e-6)

    pixel, valid = camera.project(torch.tensor([1.0, 1.0, -1.0], dtype=DTYPE))
    expected_pixel = torch.tensor(
        [4.729118411664447e2, 3.929118411664447e2],
        dtype=DTYPE,
    )
    assert bool(valid)
    # The original test uses 1e-4; 2e-5 covers only coefficient-rounding
    # differences while remaining substantially stricter than the C++ check.
    torch.testing.assert_close(pixel, expected_pixel, rtol=0.0, atol=2e-5)


def test_original_equidistant_projection_and_backprojection_examples() -> None:
    rig = load_camera_rig(FIXTURES / "davis_flyingroom.yaml", dtype=DTYPE)
    calibrated = rig.cameras[0]
    assert isinstance(calibrated, PinholeCamera)
    assert calibrated.distortion == "equidistant"
    torch.testing.assert_close(
        calibrated.intrinsics,
        torch.tensor(
            [
                198.71975957912073,
                198.68261014084223,
                163.13401862233954,
                143.59070352169638,
            ],
            dtype=DTYPE,
        ),
    )

    # test_cameras.cpp constructs this ideal equidistant camera directly, so
    # zero Kannala-Brandt coefficients reproduce that original model.
    camera = PinholeCamera(
        640,
        480,
        350.0,
        350.0,
        320.0,
        240.0,
        distortion="equidistant",
        distortion_params=[0.0, 0.0, 0.0, 0.0],
        dtype=DTYPE,
    )
    pixel, _ = camera.project(torch.tensor([10.0, 20.0, 15.0], dtype=DTYPE))
    torch.testing.assert_close(
        pixel,
        torch.tensor([473.38230111, 546.76460222], dtype=DTYPE),
        rtol=0.0,
        atol=1e-7,
    )

    optical_axis, valid = camera.project(torch.tensor([0.0, 0.0, 15.0], dtype=DTYPE))
    assert bool(valid)
    torch.testing.assert_close(
        optical_axis,
        torch.tensor([320.0, 240.0], dtype=DTYPE),
        rtol=0.0,
        atol=1e-7,
    )

    bearing = camera.unproject(torch.tensor([20.0, 190.0], dtype=DTYPE))
    torch.testing.assert_close(
        bearing,
        torch.tensor(
            [-0.7532713858288931, -0.12554523097148218, 0.6456164606573597],
            dtype=DTYPE,
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_original_pinhole_fixture_projection_round_trip() -> None:
    camera = load_camera_rig(FIXTURES / "calib_cam.yaml", dtype=DTYPE).cameras[0]
    assert isinstance(camera, PinholeCamera)

    point = torch.tensor([0.1, 0.2, 2.0], dtype=DTYPE)
    pixel, valid = camera.project(point)
    assert bool(valid)
    recovered = camera.unproject(pixel)

    # This is the same normalized-ray assertion and tolerance used by the
    # CameraProjection case in the original C++ test.
    torch.testing.assert_close(
        recovered / recovered[2],
        point / point[2],
        rtol=0.0,
        atol=1e-8,
    )
