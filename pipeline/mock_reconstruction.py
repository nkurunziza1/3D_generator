"""Demo reconstruction when VGGT / GPU is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.export_glb import points_to_glb, write_pointcloud_bin
from pipeline.fixed_poses import fixed_extrinsics_for_views


def _sample_colors(images: list[Image.Image], n: int) -> np.ndarray:
    """Fast tint from per-view average color (avoids slow per-point projection)."""
    tints = []
    for img in images:
        thumb = img.resize((32, 32))
        arr = np.asarray(thumb, dtype=np.float32) / 255.0
        tints.append(arr.mean(axis=(0, 1)))
    weights = np.linspace(0.2, 1.0, len(tints))
    base = np.average(tints, axis=0, weights=weights)
    jitter = np.random.default_rng(42).normal(0, 0.06, (n, 3))
    return np.clip(base + jitter, 0, 1).astype(np.float32)


def reconstruct_mock(
    images: dict[str, Image.Image],
    output_dir: Path,
    job_id: str,
) -> dict:
    """Procedural humanoid point cloud fused from 4 views (demo)."""
    view_ids = ["front", "right", "back", "left"]
    pil_list = [images[v] for v in view_ids]
    extrinsics = fixed_extrinsics_for_views(view_ids)

    rng = np.random.default_rng(42)
    n = 8000
    # Ellipsoid torso + head + limbs
    parts = []
    for _ in range(n):
        part = rng.choice(["torso", "head", "leg", "arm"])
        if part == "torso":
            p = rng.normal([0, 1.0, 0], [0.22, 0.35, 0.15])
        elif part == "head":
            p = rng.normal([0, 1.65, 0], [0.12, 0.12, 0.12])
        elif part == "leg":
            side = rng.choice([-1, 1])
            p = rng.normal([0.12 * side, 0.45, 0], [0.08, 0.4, 0.08])
        else:
            side = rng.choice([-1, 1])
            p = rng.normal([0.35 * side, 1.2, 0], [0.06, 0.25, 0.06])
        parts.append(p)
    points = np.array(parts, dtype=np.float64)
    _ = extrinsics
    colors = _sample_colors(pil_list, n)

    glb_path = output_dir / f"{job_id}.glb"
    pc_path = output_dir / f"{job_id}.bin"
    points_to_glb(points, colors, glb_path)
    write_pointcloud_bin(points, colors, pc_path)

    return {
        "glb_path": glb_path,
        "point_cloud_path": pc_path,
        "model": "mock-demo",
        "message": "VGGT not installed — showing demo point cloud. Install torch+vggt for real reconstruction.",
    }
