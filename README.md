# SVO Torch

SVO Torch is a PyTorch port of the visual front end from RPG SVO Pro. It keeps
the semi-direct monocular pipeline while dropping ROS, OpenCV, Ceres, GTSAM,
and OpenGV. The main camera, tracking, alignment, depth-filter, and optimization
work stays in torch, so it can run on either CPU or CUDA.

## Get started

```bash
uv sync --dev
uv run pytest
```

Inspect an SVO calibration:

```bash
uv run svo-torch inspect-calib configs/euroc_mono.yaml
```

Run a EuRoC sequence and save a TUM-format trajectory:

```bash
uv run svo-torch run /datasets/MH_01_easy \
  --calibration configs/euroc_mono.yaml \
  --config configs/default.yaml \
  --format euroc \
  --trajectory trajectory.txt
```

Pass `--device cuda` to use a GPU. Without it, the CLI picks CUDA when it is
available and otherwise uses the CPU.

The input can be a EuRoC sequence, an original SVO benchmark directory with
`data/images.txt`, or a directory of PNG, JPEG, or PGM images. Calibration uses
the usual SVO `cameras: [{camera: ..., T_B_C: ...}]` YAML layout and supports
pinhole and omnidirectional cameras, distortion, and camera masks.

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
        print(result.T_world_cam[:3, 3])
```

Images may be `uint8 [H, W]` or floating-point tensors. Timestamps are integer
nanoseconds and must increase on every frame.

Poses use `T_a_b`, meaning the transform maps points from frame `b` to frame
`a`. Public poses are `T_world_cam`; pixels are `(x, y)` and the optical frame
is x-right, y-down, z-forward.

## Real-data example

The included example downloads the official Airground pinhole bag, extracts a
small slice, runs the tracker, and evaluates the result:

```bash
uv sync --dev --extra examples
uv run --extra examples python examples/rpg_svo_pro_open/prepare_bag.py pinhole --download
uv run svo-torch run real-data/pinhole \
  --calibration real-data/pinhole/calib.yaml \
  --config examples/rpg_svo_pro_open/config/pinhole.yaml \
  --format benchmark \
  --trajectory real-data/pinhole/trajectory-pytorch.txt
uv run python examples/rpg_svo_pro_open/evaluate_trajectory.py \
  real-data/pinhole/data/stamped_groundtruth.txt \
  pytorch=real-data/pinhole/trajectory-pytorch.txt \
  --images real-data/pinhole/data/images.txt
```

The bag is about 1.77 GB. See
[`examples/rpg_svo_pro_open/README.md`](examples/rpg_svo_pro_open/README.md) for
alternate datasets, checksums, and C++ comparison steps.

## Scope

This package covers visual-only monocular odometry and reusable multi-camera
geometry. It includes pyramidal patch tracking, sparse photometric alignment,
robust pose refinement, inverse-depth seeds, and a bounded keyframe map.

It does not include the original project's ROS interface, IMU backend, loop
closure, pose graph, or global map. See [PORTING_NOTES.md](PORTING_NOTES.md) for
the implementation details and intentional differences.
