"""Tensor-native semi-direct visual odometry.

The public convention is ``T_a_b``: a homogeneous transform mapping points
from frame ``b`` into frame ``a``. Images use ``[..., H, W]`` tensors and
keypoints use ``[..., 2]`` tensors in ``(x, y)`` order.
"""

from .alignment import PyramidalPatchTracker, SparseImageAligner
from .camera import Camera, CameraRig, OmniCamera, PinholeCamera, load_camera_rig
from .config import SVOConfig
from .depth import DepthSeeds, EpipolarMatcher
from .features import GridFeatureDetector
from .frame import FeatureSet, Frame, Landmark, SparseMap
from .odometry import MonoSVO, OdometryResult, Stage, TrackingQuality, UpdateResult

__all__ = [
    "Camera",
    "CameraRig",
    "DepthSeeds",
    "EpipolarMatcher",
    "FeatureSet",
    "Frame",
    "GridFeatureDetector",
    "Landmark",
    "MonoSVO",
    "OdometryResult",
    "OmniCamera",
    "PinholeCamera",
    "PyramidalPatchTracker",
    "SVOConfig",
    "SparseImageAligner",
    "SparseMap",
    "Stage",
    "TrackingQuality",
    "UpdateResult",
    "load_camera_rig",
]

__version__ = "0.1.0"
