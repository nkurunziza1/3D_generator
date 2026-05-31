"""
Human 3D Scan — FastAPI reconstruction service.

Endpoints:
  POST /reconstruct       — 4 images → GLB + point cloud
  POST /remove-background — placeholder for matting API
  GET  /outputs/{file}    — serve generated assets
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.reconstruct import run_reconstruction

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Human 3D Scan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/health")
def health():
    from pipeline.reconstruct import _env, _openrouter_ready, _real_inference_ready
    from pipeline.replicate_reconstruction import replicate_ready

    mode = _env("RECONSTRUCTION_MODEL", "mock").lower()
    if mode == "replicate":
        ready, reason = replicate_ready()
    elif mode == "openrouter":
        ready, reason = _openrouter_ready()
    elif mode in ("vggt", "vgg-t3", "vggt3"):
        ready, reason = _real_inference_ready(mode)
    else:
        ready, reason = True, ""

    return {
        "status": "ok",
        "model": mode,
        "replicate_model": _env("REPLICATE_MODEL") or None,
        "openrouter_model": _env("OPENROUTER_MODEL") or None,
        "inference_ready": ready,
        "inference_note": reason or None,
        "hf_token_set": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")),
        "openrouter_key_set": bool(os.getenv("OPENROUTER_API_KEY")),
        "replicate_token_set": bool(os.getenv("REPLICATE_API_TOKEN")),
    }


@app.post("/reconstruct")
async def reconstruct(
    front: UploadFile = File(...),
    right: UploadFile = File(...),
    back: UploadFile = File(...),
    left: UploadFile = File(...),
    use_fixed_poses: str = Form("true"),
):
    try:
        image_bytes = {
            "front": await front.read(),
            "right": await right.read(),
            "back": await back.read(),
            "left": await left.read(),
        }
        fixed = use_fixed_poses.lower().strip() in ("true", "1", "yes")
        result = run_reconstruction(image_bytes, OUTPUT_DIR, use_fixed_poses=fixed)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse(
            {"detail": str(exc), "message": str(exc)},
            status_code=500,
        )


@app.post("/remove-background")
async def remove_background(image: UploadFile = File(...)):
    """
    Placeholder for background removal (e.g. rembg, Segment Anything, HF Inference).
    Returns the original image until you wire a matting model.
    """
    data = await image.read()
    out_path = OUTPUT_DIR / f"nobg_{image.filename or 'capture.jpg'}"
    out_path.write_bytes(data)
    return FileResponse(out_path, media_type=image.content_type or "image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
