# Official RPG SVO Pro real-data examples

This directory turns the two ROS bags used by RPG SVO Pro's own examples into
the original `data/images.txt` benchmark layout. The same decoded PNGs can then
be consumed by SVO Pro's C++ benchmark and by `svo-torch`, avoiding image-codec
or timestamp differences during comparison.

Run every command below from the `rpg_svo_pro_open_pytorch` repository root.
The bag extractor needs `rosbags`; ROS is not needed by the PyTorch frontend.

## Pinhole/ATAN Airground example

This is the primary example because the bag contains pose ground truth. Its
canonical camera is a pinhole model with SVO's one-parameter ATAN/fisheye
distortion.

### 1. Install and select a device

```bash
uv sync --dev --extra examples
DEVICE="$(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
printf 'using device: %s\n' "$DEVICE"
```

`DEVICE` becomes `cuda` only when the installed PyTorch build can use a CUDA
device; it otherwise falls back to `cpu`. Passing `--device "$DEVICE"` makes
that choice visible and repeatable.

### 2. Download, verify, and extract 500 frames

```bash
uv run --extra examples python examples/rpg_svo_pro_open/prepare_bag.py pinhole --download
```

The helper validates the official bag's exact byte count and SHA-256 before
decoding. The default is deliberately bounded to the first 500 images; it does
not decode the complete recording. `--download` still retrieves and caches the
full 1.77 GB verified bag. To use an already downloaded official bag:

```bash
uv run --extra examples python examples/rpg_svo_pro_open/prepare_bag.py pinhole \
  --bag /absolute/path/to/airground_rig_s3_2013-03-18_21-38-48.bag
```

The output is:

```text
real-data/pinhole/
├── calib.yaml
├── example.json
└── data/
    ├── images.txt
    ├── stamped_groundtruth.txt
    └── img/frame_000000.png ...
```

`example.json` records the source bag, expected SHA-256, selected frame range,
timestamps, extracted frame count, and ground-truth pose count. Use
`--start-frame N` to choose another bounded slice. Use `--max-frames 0` only
when the complete bag is wanted; if the destination already exists, add
`--force` to replace that one prepared example.

### 3. Inspect the original calibration and run PyTorch

```bash
uv run svo-torch inspect-calib real-data/pinhole/calib.yaml \
  --device "$DEVICE" \
  --dtype float32

uv run svo-torch run real-data/pinhole \
  --calibration real-data/pinhole/calib.yaml \
  --config examples/rpg_svo_pro_open/config/pinhole.yaml \
  --format benchmark \
  --device "$DEVICE" \
  --trajectory real-data/pinhole/trajectory-pytorch.txt
```

The standalone settings not present in the ROS file retain their SVO-compatible
defaults, while `pinhole.yaml` is the original RPG runtime file and overrides
them through its original key names. Original ROS-only keys that do not apply
to the standalone frontend are ignored.

The CLI prints one status line per frame and a final processed/written count.
Only successfully initialized and tracked poses are serialized, so
`trajectory-pytorch.txt` normally contains fewer records than `images.txt`.
Add `--quiet` when only the final count is wanted.

### 4. Evaluate against ground truth

```bash
uv run python examples/rpg_svo_pro_open/evaluate_trajectory.py \
  real-data/pinhole/data/stamped_groundtruth.txt \
  pytorch=real-data/pinhole/trajectory-pytorch.txt \
  --images real-data/pinhole/data/images.txt \
  --json real-data/pinhole/evaluation-pytorch.json
```

The evaluator performs one-to-one nearest-timestamp association (20 ms by
default), estimates an Umeyama Sim(3) transform for monocular scale and frame
alignment, and reports coverage, ATE, absolute rotation RMSE, and 1-second RPE.
For this Airground bag, use coverage and Sim(3)-aligned position ATE as the
implementation parity metrics. The `Rig` ground-truth quaternion
basis/convention is not directly camera-comparable: in our native 400-frame
verification, position ATE was 4.50 mm while absolute rotation RMSE was 178.49°.
Rotation and rotational-RPE values remain in the report as diagnostics, not as
a meaningful C++/PyTorch parity test.

The human-readable report goes to the terminal and the complete report goes to
`real-data/pinhole/evaluation-pytorch.json`. Supplying `--images` makes coverage
use all 500 requested input frames as its denominator rather than only the
ground-truth records.

To score a trajectory produced by the original C++ benchmark on exactly the
same extracted images, add another `NAME=PATH` argument:

```bash
uv run python examples/rpg_svo_pro_open/evaluate_trajectory.py \
  real-data/pinhole/data/stamped_groundtruth.txt \
  cpp=/absolute/path/to/stamped_traj_estimate.txt \
  pytorch=real-data/pinhole/trajectory-pytorch.txt \
  --images real-data/pinhole/data/images.txt \
  --json real-data/pinhole/evaluation-cpp-pytorch.json
```

Every estimate is associated and Sim(3)-aligned independently. This makes the
report suitable for comparing monocular implementations, but it is not a
frame-by-frame equality assertion: initialization and failure/recovery policy
can produce different coverage.

## Optional Omni/fisheye example

The second official bag exercises the 24-parameter Omni model and its external
bitmap mask. It has no pose ground-truth topic, so it is a real-image tracking
smoke test rather than an accuracy benchmark.

```bash
DEVICE="$(uv run python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
uv run --extra examples python examples/rpg_svo_pro_open/prepare_bag.py fisheye --download

uv run svo-torch inspect-calib real-data/fisheye/calib.yaml \
  --device "$DEVICE" \
  --dtype float32

uv run svo-torch run real-data/fisheye \
  --calibration real-data/fisheye/calib.yaml \
  --config examples/rpg_svo_pro_open/config/pinhole.yaml \
  --config examples/rpg_svo_pro_open/config/fisheye.yaml \
  --format benchmark \
  --device "$DEVICE" \
  --trajectory real-data/fisheye/trajectory-pytorch.txt
```

The two explicit configuration layers mirror the original launch convention:
original pinhole base parameters followed by the original fisheye-specific
overrides; standalone-only fields retain their SVO-compatible defaults.
Preparation copies
`25000826_fisheye_mask.png` beside `calib.yaml`; the relative YAML `mask` path
therefore resolves without path rewriting. Zero-valued pixels are excluded
from feature detection, while projection and unprojection remain geometric.
The default extraction is again 500 frames. The result is
`real-data/fisheye/trajectory-pytorch.txt`; no evaluation JSON is produced
because this bag has no reference trajectory.

## Dataset provenance and integrity

No bag data is redistributed in this repository. `prepare_bag.py` downloads
from the RPG/UZH dataset host and rejects a file unless both its size and
SHA-256 match the values below.

| Example | Official bag | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Pinhole | [airground_rig_s3_2013-03-18_21-38-48.bag](https://download.ifi.uzh.ch/rpg/web/datasets/airground_rig_s3_2013-03-18_21-38-48.bag) | 1,772,552,296 | `f84eb9c5931f6da9e3fe54197c761dcab7992fde61d6151b9513ca39e1b26f5f` |
| Omni/fisheye | [test_fisheye.bag](https://download.ifi.uzh.ch/rpg/web/datasets/test_fisheye.bag) | 742,137,657 | `0b1d8b489309932a8bd12ac44ce41a44c1694ac5ff08c2a1a654327613968f45` |

Downloads are cached by default under `~/.cache/svo-torch/`. Change that with
`--cache PATH`; change the prepared dataset parent with `--output PATH`.

The bundled calibration and runtime files come from
[RPG SVO Pro commit `ca371f304637e7fb355cf4624d0a02da4e3da220`](https://github.com/uzh-rpg/rpg_svo_pro_open/tree/ca371f304637e7fb355cf4624d0a02da4e3da220).
The two pinhole text files only normalize upstream trailing whitespace/final
newline; the three fisheye assets are byte-for-byte copies.

| Bundled asset | Upstream path | Bundled SHA-256 |
| --- | --- | --- |
| [`calib/svo_test_pinhole.yaml`](calib/svo_test_pinhole.yaml) | `svo_ros/param/calib/svo_test_pinhole.yaml` | `2c7ac955b12dce918e685d78fb55483661f386950dd3506999fb20a2d5086c39` |
| [`config/pinhole.yaml`](config/pinhole.yaml) | `svo_ros/param/pinhole.yaml` | `c1221bad49877ccfbe90e3b94b6dfe0322543cb520281e880ede7f9644c4a343` |
| [`calib/bluefox_25000826_fisheye.yaml`](calib/bluefox_25000826_fisheye.yaml) | `svo_ros/param/calib/bluefox_25000826_fisheye.yaml` | `fba831dd550f6b48aa2802e4d6bca2149e8ebcac2cdbc34fb8c5fde178510802` |
| [`calib/25000826_fisheye_mask.png`](calib/25000826_fisheye_mask.png) | `svo_ros/param/calib/25000826_fisheye_mask.png` | `4d4c6ee7ba17712e0c63d9806aa11f0e9aae017d7aa1df3b19f9f6de8adc28d5` |
| [`config/fisheye.yaml`](config/fisheye.yaml) | `svo_ros/param/fisheye.yaml` | `bc082142f201db03625230de8faa2ab61bd5da3b61c30c90144e06108b5634bd` |

## Command reference

```bash
uv run python examples/rpg_svo_pro_open/prepare_bag.py --help
uv run svo-torch run --help
uv run python examples/rpg_svo_pro_open/evaluate_trajectory.py --help
```

`prepare_bag.py --force` replaces only the selected prepared output directory;
it does not delete the verified download cache. A corrupt or unofficial bag,
an unexpected image topic/encoding, non-monotonic timestamps, and mismatched
image dimensions all stop with an explicit error rather than producing a
partially trusted benchmark.
