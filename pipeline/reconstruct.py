"""Unified reconstruction entry — selects VGGT, VGG-T³, or mock."""

from __future__ import annotations

import io
import os
from pathlib import Path
from uuid import uuid4

from PIL import Image

from pipeline.mock_reconstruction import reconstruct_mock
from pipeline.replicate_reconstruction import replicate_ready

VIEW_ORDER = ["front", "right", "back", "left"]


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _openrouter_ready() -> tuple[bool, str]:
    if not os.getenv("OPENROUTER_API_KEY"):
        return False, "Set OPENROUTER_API_KEY in backend/.env (from https://openrouter.ai/keys)"
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        return False, "Install openai: pip install openai (included in requirements.txt)"
    return True, ""


def _real_inference_ready(model_mode: str) -> tuple[bool, str]:
    """Check whether VGGT / VGG-T³ can run in this environment."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, (
            "PyTorch is not installed in backend/.venv. "
            "Set RECONSTRUCTION_MODEL=mock in backend/.env for local dev on Mac."
        )

    if model_mode in ("vgg-t3", "vggt3", "vgg_t3"):
        try:
            import vggttt  # noqa: F401
        except ImportError:
            return False, (
                "vgg-ttt is not installed (pip install from github.com/nv-dvl/vgg-ttt). "
                "Requires Linux + NVIDIA GPU. Using demo reconstruction."
            )
    else:
        try:
            import vggt  # noqa: F401
        except ImportError:
            return False, (
                "vggt is not installed (pip install git+https://github.com/facebookresearch/vggt.git). "
                "Using demo reconstruction."
            )

    return True, ""


def run_reconstruction(
    image_bytes: dict[str, bytes],
    output_dir: Path,
    use_fixed_poses: bool = True,
) -> dict:
    job_id = uuid4().hex[:12]
    images = {
        k: Image.open(io.BytesIO(v)).convert("RGB")
        for k, v in image_bytes.items()
        if k in VIEW_ORDER
    }

    for v in VIEW_ORDER:
        if v not in images:
            raise ValueError(f"Missing required view: {v}")

    model_mode = _env("RECONSTRUCTION_MODEL", "mock").lower()

    if model_mode == "replicate":
        ready, skip_reason = replicate_ready()
        if not ready:
            result = reconstruct_mock(images, output_dir, job_id)
            result["message"] = skip_reason
        else:
            try:
                from pipeline.replicate_reconstruction import reconstruct_replicate

                result = reconstruct_replicate(
                    images, output_dir, job_id, use_fixed_poses=use_fixed_poses
                )
            except Exception as exc:
                result = reconstruct_mock(images, output_dir, job_id)
                result["message"] = f"Replicate failed ({exc}). Using demo point cloud."
    elif model_mode == "openrouter":
        ready, skip_reason = _openrouter_ready()
        if not ready:
            result = reconstruct_mock(images, output_dir, job_id)
            result["message"] = skip_reason
        else:
            try:
                from pipeline.openrouter_reconstruction import reconstruct_openrouter

                result = reconstruct_openrouter(
                    images, output_dir, job_id, use_fixed_poses=use_fixed_poses
                )
            except Exception as exc:
                result = reconstruct_mock(images, output_dir, job_id)
                result["message"] = f"OpenRouter failed ({exc}). Using demo point cloud."
    elif model_mode in ("vggt", "vgg-t3", "vggt3"):
        ready, skip_reason = _real_inference_ready(model_mode)
        if not ready:
            result = reconstruct_mock(images, output_dir, job_id)
            result["message"] = skip_reason
        else:
            try:
                from pipeline.vggt_inference import reconstruct_vggt

                result = reconstruct_vggt(
                    images,
                    output_dir,
                    job_id,
                    use_fixed_poses=use_fixed_poses,
                    model_name=model_mode,
                )
            except Exception as exc:
                result = reconstruct_mock(images, output_dir, job_id)
                result["message"] = f"Reconstruction failed ({exc}). Using demo point cloud."
    else:
        result = reconstruct_mock(images, output_dir, job_id)
        if model_mode == "mock":
            result["message"] = (
                "Preview uses a demo body shape from your photos. "
                "For a real 3D human mesh, run the backend on a Linux GPU with RECONSTRUCTION_MODEL=vgg-t3."
            )

    glb = result["glb_path"]
    pc = result.get("point_cloud_path")
    return {
        "jobId": job_id,
        "glbUrl": f"/outputs/{glb.name}",
        "pointCloudUrl": f"/outputs/{pc.name}" if pc else None,
        "model": result.get("model", model_mode),
        "message": result.get("message"),
    }
