"""Fixed camera extrinsics for 4-view human scan (yaw around subject)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# OpenCV camera-from-world convention: [R|t] 3x4
VIEW_YAWS_DEG = {"front": 0.0, "right": 90.0, "back": 180.0, "left": 270.0}


def yaw_to_extrinsic(yaw_deg: float, radius: float = 2.5, height: float = 1.0) -> np.ndarray:
    """Camera looks at origin from a point on a circle in the XZ plane."""
    yaw = math.radians(yaw_deg)
    cam_x = radius * math.sin(yaw)
    cam_z = radius * math.cos(yaw)
    cam_y = height
    cam_pos = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8
    up = np.cross(right, forward)

    R = np.stack([right, up, -forward], axis=0)
    t = -R @ cam_pos
    ext = np.eye(4, dtype=np.float64)
    ext[:3, :3] = R
    ext[:3, 3] = t
    return ext[:3, :4]


def fixed_extrinsics_for_views(view_ids: Sequence[str]) -> list[np.ndarray]:
    return [yaw_to_extrinsic(VIEW_YAWS_DEG[v]) for v in view_ids]


def default_intrinsic(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
    K = np.array(
        [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    return K
