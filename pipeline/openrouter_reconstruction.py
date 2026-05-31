"""
3D preview via OpenRouter vision model (course pattern: OpenAI client + base_url).

Sends 4 guided views to a vision LLM, gets body proportions JSON, builds a GLB point cloud.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.body_pointcloud import build_point_cloud_from_params
from pipeline.export_glb import points_to_glb, write_pointcloud_bin

VIEW_ORDER = ["front", "right", "back", "left"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Strong vision + spatial reasoning; swap via OPENROUTER_MODEL in .env
DEFAULT_MODEL = "google/gemini-2.5-flash"

SYSTEM_PROMPT = """You are a 3D human scanning assistant.
Analyze four photos of the same standing person: front (0°), right (90°), back (180°), left (270°).

Return ONLY valid JSON (no markdown) with estimated body parameters in meters and RGB 0-255:
{
  "height_m": 1.7,
  "shoulder_width_m": 0.42,
  "hip_width_m": 0.34,
  "torso_depth_m": 0.22,
  "head_radius_m": 0.11,
  "skin_rgb": [200, 160, 130],
  "shirt_rgb": [50, 80, 160],
  "pants_rgb": [40, 45, 55],
  "build": "average",
  "summary": "one sentence description"
}
build must be one of: slim, average, athletic
Use the photos to estimate proportions and dominant clothing/skin colors."""


def _client():
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in backend/.env")

    return OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=api_key,
    )


def _image_to_data_url(img: Image.Image, max_side: int = 768) -> str:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


def reconstruct_openrouter(
    images: dict[str, Image.Image],
    output_dir: Path,
    job_id: str,
    use_fixed_poses: bool = True,
) -> dict:
    _ = use_fixed_poses
    client = _client()
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    content: list[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT + "\n\nPhotos follow in order: front, right, back, left."}
    ]
    labels = {"front": "0° front", "right": "90° right", "back": "180° back", "left": "270° left"}
    for vid in VIEW_ORDER:
        content.append({"type": "text", "text": f"--- {labels[vid]} ---"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_to_data_url(images[vid])},
            }
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        extra_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost:3000"),
            "X-Title": "Human 3D Scan",
        },
    )

    raw = response.choices[0].message.content or "{}"
    params = _parse_json_response(raw)
    points, colors = build_point_cloud_from_params(params)

    glb_path = output_dir / f"{job_id}.glb"
    pc_path = output_dir / f"{job_id}.bin"
    points_to_glb(points, colors, glb_path)
    write_pointcloud_bin(points, colors, pc_path)

    summary = params.get("summary", "AI-estimated body from 4 views")
    return {
        "glb_path": glb_path,
        "point_cloud_path": pc_path,
        "model": f"openrouter/{model}",
        "message": f"OpenRouter reconstruction: {summary}",
        "body_params": params,
    }
