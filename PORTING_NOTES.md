# Porting notes

## Conventions

- `T_a_b` maps a point from frame `b` to frame `a`.
- The public pose is `T_world_cam`; the optical frame is x-right, y-down,
  z-forward.
- Lie tangents are `[tx, ty, tz, rx, ry, rz]`; optimizer updates multiply on
  the left.
- Pixels and gradients are stored row-wise as `[N,2]` in `(x,y)` order.
- Calibration `T_B_C` maps camera to body and is exposed as
  `CameraRig.T_body_camera`.
- Text trajectories use `timestamp tx ty tz qx qy qz qw`.

## Algorithm correspondence

Frames use floor-sized 2×2 box-average pyramids. Detector `n_pyr_levels` and
the image-alignment maximum level remain independent, as in the original; the
shared image pyramid is large enough for both. Detection computes a Shi–Tomasi
response and gradient edgelets at multiple levels, then retains at most one
candidate per level-zero occupancy cell. Pyramidal tracking uses the
inverse-compositional 8×8 patch formulation. Sparse alignment jointly optimizes
a relative SE(3) pose over projected landmark patches with robust weights.

Monocular initialization tracks a fixed reference until sufficient disparity,
uses normalized eight-point RANSAC, selects the essential-matrix decomposition
by cheirality, triangulates rays, and scales median scene distance to the
configured prior. Normal tracking follows the SVO ordering: direct pose prior,
patch tracking/reprojection, robust geometric pose refinement, quality check,
keyframe selection, triangulation, feature replenishment, and bounded-map
pruning. After propagating last-frame tracks, landmarks that are missing from
the current observations are projected with the motion prior and patch-matched
from their newest valid retained-keyframe observation. These local-map matches
are deduplicated and take priority over seedless tracks within the configured
feature budget. `reprojector_max_keyframes` (including the original
`reprojector_max_n_kfs` alias) bounds how many recent keyframes are searched.

Inverse-depth seeds retain SVO's `[mu,sigma2,a,b]` Gaussian/uniform plus Beta
mixture update and its pixel-angle uncertainty model. The implementation is
synchronous; callers can place independent sequences on separate streams or
workers without hidden background threads.

## Intentional differences

- Image intensities are normalized to `[0,1]`; photometric thresholds are
  scaled accordingly.
- Autograd supplies camera-model and pose Jacobians. This is easier to audit
  across distortion models; analytic fused kernels can replace it later.
- Eight-point essential estimation replaces OpenGV/homography initialization,
  keeping the package free of non-PyTorch vision dependencies.
- The visual front end triangulates fresh tracks at keyframes. The standalone
  inverse-depth filter remains available for applications that update seeds at
  every frame; the epipolar/seed fields in `SVOConfig` retain the corresponding
  original parameter names but are not an asynchronous `MonoSVO` worker.
- Sparse photometric alignment currently assumes fixed normalized brightness.
  Original illumination gain/offset keys are accepted for configuration-file
  compatibility, but affine brightness estimation is not implemented.
- Omni annular validity radii embedded in the 24 calibration parameters are
  enforced. External bitmap paths in a YAML `mask` field are not loaded; apply
  such a mask to input images before processing when it contains additional
  irregular invalid regions.
- Relocalization retries the last good local frame; there is no bag-of-words
  place recognition database.

## Out of scope

The original ROS1 transport/UI, OKVIS-derived sliding-window visual-inertial
Ceres backend, DBoW2 loop closure, Ceres pose graph, and GTSAM/iSAM2 global map
are integration layers rather than tensor kernels. They remain in the source
repository and are not claimed as part of this port.
