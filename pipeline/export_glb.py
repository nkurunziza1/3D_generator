"""Export point clouds to GLB and custom binary — no trimesh/scipy required."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _pad4(data: bytes) -> bytes:
    return data + b" " * ((4 - len(data) % 4) % 4)


def points_to_glb(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_path: Path,
) -> Path:
    """Write a minimal GLB with POINTS primitive (viewable in Three.js)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if colors is None:
        colors = np.full((n, 3), 0.7, dtype=np.float32)
    else:
        colors = np.asarray(colors, dtype=np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0

    positions_bin = points.tobytes()
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    colors_bin = colors_u8.tobytes()

    buffer_data = positions_bin + colors_bin
    # GLB: JSON chunk + BIN chunk
    pos_offset = 0
    col_offset = len(positions_bin)
    pos_accessor = {
        "bufferView": 0,
        "componentType": 5126,
        "count": n,
        "type": "VEC3",
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
    }
    col_accessor = {
        "bufferView": 1,
        "componentType": 5121,
        "count": n,
        "type": "VEC3",
        "normalized": True,
    }
    gltf = {
        "asset": {"version": "2.0", "generator": "human-3d-scan"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "COLOR_0": 1},
                        "mode": 0,
                    }
                ]
            }
        ],
        "accessors": [pos_accessor, col_accessor],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": len(positions_bin)},
            {"buffer": 0, "byteOffset": col_offset, "byteLength": len(colors_bin)},
        ],
        "buffers": [{"byteLength": len(buffer_data)}],
    }

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    bin_chunk = _pad4(buffer_data)

    glb = (
        struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(json_chunk) + 8 + len(bin_chunk))
        + struct.pack("<I4s", len(json_chunk), b"JSON")
        + json_chunk
        + struct.pack("<I4s", len(bin_chunk), b"BIN\x00")
        + bin_chunk
    )
    output_path.write_bytes(glb)
    return output_path


def write_pointcloud_bin(
    points: np.ndarray,
    colors: np.ndarray | None,
    output_path: Path,
) -> Path:
    """Custom binary for Three.js viewer: uint32 count + float32 xyz + uint8 rgb."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    if colors is None:
        colors = np.ones((n, 3), dtype=np.float32) * 0.7

    buf = bytearray()
    buf.extend(struct.pack("<I", n))
    for i in range(n):
        buf.extend(struct.pack("<fff", *points[i]))
        c = colors[i]
        if c.max() <= 1:
            r, g, b = (c * 255).astype(np.uint8)
        else:
            r, g, b = c.astype(np.uint8)
        buf.extend(bytes([int(r), int(g), int(b)]))

    output_path.write_bytes(buf)
    return output_path
