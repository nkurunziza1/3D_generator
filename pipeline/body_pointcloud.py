"""Shared procedural body point cloud from parameter dict."""

from __future__ import annotations

import numpy as np


def build_point_cloud_from_params(params: dict, n: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    height = float(params.get("height_m", 1.7))
    scale = height / 1.7
    shoulder = float(params.get("shoulder_width_m", 0.42)) * scale
    hip = float(params.get("hip_width_m", 0.34)) * scale
    torso_d = float(params.get("torso_depth_m", 0.22)) * scale
    head_r = float(params.get("head_radius_m", 0.11)) * scale
    build = str(params.get("build", "average")).lower()

    build_scale = {"slim": 0.88, "average": 1.0, "athletic": 1.12}.get(build, 1.0)
    shoulder *= build_scale
    hip *= build_scale

    skin = np.array(params.get("skin_rgb", [200, 160, 130]), dtype=np.float32) / 255.0
    shirt = np.array(params.get("shirt_rgb", [50, 80, 160]), dtype=np.float32) / 255.0
    pants = np.array(params.get("pants_rgb", [40, 45, 55]), dtype=np.float32) / 255.0

    center_y = height * 0.5
    head_y = height * 0.88
    rng = np.random.default_rng(7)

    points = []
    colors = []
    for _ in range(n):
        part = rng.choice(["torso", "head", "leg", "arm"], p=[0.4, 0.12, 0.28, 0.2])
        if part == "torso":
            p = rng.normal([0, center_y, 0], [shoulder * 0.35, height * 0.22, torso_d * 0.4])
            c = shirt
        elif part == "head":
            p = rng.normal([0, head_y, 0], [head_r, head_r, head_r * 0.9])
            c = skin
        elif part == "leg":
            side = rng.choice([-1, 1])
            p = rng.normal([hip * 0.2 * side, height * 0.22, 0], [hip * 0.15, height * 0.28, torso_d * 0.3])
            c = pants
        else:
            side = rng.choice([-1, 1])
            p = rng.normal([shoulder * 0.45 * side, center_y + height * 0.12, 0], [0.06, height * 0.14, 0.06])
            c = shirt if rng.random() > 0.3 else skin
        points.append(p)
        colors.append(np.clip(c + rng.normal(0, 0.04, 3), 0, 1))

    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.float32)
