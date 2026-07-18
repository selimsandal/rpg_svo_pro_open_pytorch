# SVO Torch

SVO Torch is a standalone, tensor-native PyTorch port of the visual front end
from [RPG SVO Pro](https://github.com/uzh-rpg/rpg_svo_pro_open). It keeps the
original semi-direct design—grid-distributed features, pyramidal patch
tracking, sparse photometric alignment, robust pose refinement, inverse-depth
seeds, and a bounded keyframe map—without requiring ROS, OpenCV, Ceres, GTSAM,
or OpenGV.

The implementation uses the convention `T_a_b`: a 4×4 transform maps points
from frame `b` into frame `a`. Pixels are `(x, y)`, camera coordinates are
x-right/y-down/z-forward, and public poses are `T_world_cam`.

## Install with uv

```bash
cd rpg_svo_pro_open_pytorch
uv sync --dev
uv run pytest
```

Inspect an original SVO calibration:

```bash
uv run svo-torch inspect-calib configs/euroc_mono.yaml
```

Run on EuRoC and write a TUM-format trajectory:

```bash
uv run svo-torch run /datasets/MH_01_easy \
  --calibration configs/euroc_mono.yaml \
  --config configs/default.yaml \
  --format euroc \
  --trajectory trajectory.txt
```

The trajectory contains successfully initialized/tracked poses only;
initialization attempts and failed/relocalizing updates are not serialized.

`--device cuda` keeps images, geometry, direct alignment, and optimization on
the GPU. CPU is selected automatically when no accelerator is available.

The CLI also reads:

- the original SVO benchmark layout with `data/images.txt`;
- a directory of lexically sorted PNG/JPEG/PGM images (`--period-ns` sets its clock);
- EuRoC `mav0/cam0/data.csv` sequences.

Calibration files use the canonical SVO
`cameras: [{camera: ..., T_B_C: ...}]` YAML schema. Pinhole cameras support no
distortion, radial-tangential, equidistant, and one-parameter ATAN/fisheye
distortion. The original 24-parameter omnidirectional model is supported too.

## Python API

```python
from svo_torch import MonoSVO, SVOConfig, Stage, UpdateResult, load_camera_rig
from svo_torch.datasets import EurocDataset

config = SVOConfig.from_yaml("configs/default.yaml")
rig = load_camera_rig("configs/euroc_mono.yaml", device=config.torch_device())
dataset = EurocDataset("/datasets/MH_01_easy", camera=rig.cameras[0])

vo = MonoSVO(rig.cameras[0], config)
vo.start()
for sample in dataset:
    result = vo.process(sample.image, sample.timestamp_ns)
    if (
        result.T_world_cam is not None
        and result.stage == Stage.TRACKING
        and result.update != UpdateResult.FAILURE
    ):
        print(result.stage, result.T_world_cam[:3, 3])
```

Input images can be `uint8 [H,W]` or floating-point tensors. Internally they
are grayscale `[1,1,H,W]` tensors in `[0,1]`. All timestamps are integer
nanoseconds and must be strictly increasing.

## What was ported

| SVO Pro subsystem | PyTorch implementation |
| --- | --- |
| `vikit_cameras` | Differentiable pinhole/omni projection and SVO YAML rig loading |
| `svo_common` | Frames, parallel feature tensors, landmarks, bounded keyframe map |
| `svo_direct` | Shi–Tomasi/edgelet detection, patch sampling, epipolar matching, inverse-depth Bayesian updates |
| `svo_tracker` | Batched inverse-compositional pyramidal patch tracking |
| `svo_img_align` | Coarse-to-fine sparse photometric SE(3) alignment |
| `svo` | Two-view initialization, robust pose refinement, keyframe/map state machine |

The port is deliberately tensor-native: numerical kernels are torch operations,
camera projection and sampling remain differentiable, and no OpenCV/Kornia
fallback hides CPU work.

## Scope

This port covers visual-only monocular odometry and reusable multi-camera
calibration geometry. Its self-contained initializer uses normalized
eight-point RANSAC instead of the C++ project's OpenGV five-point/homography
implementations. ROS adapters, IMU preintegration and the OKVIS/Ceres backend,
DBoW2 loop closure, pose-graph optimization, and the GTSAM global map are
external integrations in the C++ project and are not reimplemented here.

See [PORTING_NOTES.md](PORTING_NOTES.md) for conventions and parity details.

SVO Torch is a derived implementation distributed under GPL-3.0-only.
See [LICENSE](LICENSE) and the original project's attribution and citations.

If used academically, cite the original SVO work:

- C. Forster, M. Pizzoli, D. Scaramuzza, “SVO: Fast Semi-Direct Monocular
  Visual Odometry,” ICRA 2014.
- C. Forster et al., “SVO: Semi-Direct Visual Odometry for Monocular and
  Multi-Camera Systems,” IEEE TRO 2017.
