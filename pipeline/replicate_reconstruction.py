"""
Image-to-3D mesh via Replicate (TRELLIS.2 on cloud GPU — no local CUDA).

Uses your front capture photo. Get token: https://replicate.com/account/api-tokens
Model: https://replicate.com/fishwowater/trellis2
"""

from __future__ import annotations

import os
import ssl
import tempfile
import time
from pathlib import Path

import httpx
from PIL import Image

DEFAULT_REPLICATE_MODEL = "fishwowater/trellis2"
FALLBACK_REPLICATE_MODEL = "firtoz/trellis"
DEFAULT_PIPELINE_TYPE = "1024"
MAX_RETRIES = 4
RETRYABLE_STATUS = ("500", "502", "503", "504", "Internal server error")
NETWORK_ERRORS = (
    "handshake operation timed out",
    "ssl",
    "timed out",
    "timeout",
    "connection reset",
    "connection refused",
    "network is unreachable",
    "connecterror",
    "readtimeout",
)


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def replicate_ready() -> tuple[bool, str]:
    if not _env("REPLICATE_API_TOKEN"):
        return False, (
            "Set REPLICATE_API_TOKEN in backend/.env — "
            "create at https://replicate.com/account/api-tokens"
        )
    try:
        import replicate  # noqa: F401
    except ImportError:
        return False, "Install replicate: pip install replicate"
    return True, ""


def _httpx_timeout() -> httpx.Timeout:
    """Replicate SDK default connect=5s causes SSL handshake timeouts on slow links."""
    return httpx.Timeout(
        connect=_env_float("REPLICATE_CONNECT_TIMEOUT", 120.0),
        read=_env_float("REPLICATE_READ_TIMEOUT", 900.0),
        write=_env_float("REPLICATE_WRITE_TIMEOUT", 120.0),
        pool=_env_float("REPLICATE_POOL_TIMEOUT", 60.0),
    )


def _get_client():
    import replicate

    token = _env("REPLICATE_API_TOKEN")
    os.environ["REPLICATE_API_TOKEN"] = token
    os.environ.setdefault("REPLICATE_POLL_INTERVAL", "2.0")
    return replicate.Client(api_token=token, timeout=_httpx_timeout())


def _resolve_model_ref(model_slug: str) -> str:
    client = _get_client()
    slug = model_slug.strip()
    if ":" in slug:
        return slug
    owner, name = slug.split("/", 1)
    model = client.models.get(owner, name)
    return f"{owner}/{name}:{model.latest_version.id}"


def _format_replicate_error(exc: Exception) -> str:
    msg = str(exc)
    lower = msg.lower()
    if "402" in msg or "Insufficient credit" in msg:
        return (
            "Replicate account has insufficient credit. "
            "Add billing at https://replicate.com/account/billing (~$0.82 per TRELLIS run)."
        )
    if "404" in msg or "not found" in lower:
        return (
            f"Replicate model not found ({_env('REPLICATE_MODEL', DEFAULT_REPLICATE_MODEL)}). "
            "Check REPLICATE_MODEL in backend/.env"
        )
    if "429" in msg or "throttled" in lower:
        return "Replicate rate limit — wait 1–2 minutes and try again."
    if any(s in msg for s in RETRYABLE_STATUS):
        return (
            "Replicate server error (500) — their GPU failed on this run. "
            "Try again, or set REPLICATE_PIPELINE_TYPE=512 in .env."
        )
    if any(e in lower for e in NETWORK_ERRORS):
        return (
            "Network timeout reaching Replicate (SSL/connect). "
            "Check your internet, disable VPN if possible, wait 30s and retry. "
            "TRELLIS runs take 2–10 minutes — keep the page open."
        )
    return msg


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(s.lower() in msg for s in RETRYABLE_STATUS):
        return True
    if any(e in msg for e in NETWORK_ERRORS):
        return True
    if isinstance(exc, (TimeoutError, ssl.SSLError, OSError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    return False


def _prepare_front_image(front: Image.Image) -> str:
    img = front.convert("RGB")
    w, h = img.size
    max_side = int(_env("REPLICATE_MAX_IMAGE_SIDE", "1024") or "1024")
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    if min(img.size) < 256:
        img = img.resize((512, max(512, int(512 * h / max(w, 1)))), Image.Resampling.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=90, optimize=True)
    return tmp.name


def _extract_model_url(output) -> str | None:
    if output is None:
        return None
    if isinstance(output, str) and output.startswith("http"):
        return output
    if isinstance(output, dict):
        for key in ("model_file", "glb", "model", "mesh"):
            val = output.get(key)
            url = _extract_model_url(val)
            if url:
                return url
        return None
    if hasattr(output, "url"):
        url = getattr(output, "url", None)
        if url:
            return str(url)
    return None


def _download_glb(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(
        connect=60.0,
        read=_env_float("REPLICATE_DOWNLOAD_TIMEOUT", 600.0),
        write=60.0,
        pool=30.0,
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "human-3d-scan/1.0"})
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                return
        except Exception as exc:
            last_err = exc
            if attempt < 2 and _is_retryable(exc):
                time.sleep(3 * (attempt + 1))
                continue
            raise
    if last_err:
        raise last_err


def _run_replicate_once(model_slug: str, image_path: str) -> object:
    client = _get_client()
    model_ref = _resolve_model_ref(model_slug)
    is_trellis2 = "trellis2" in model_slug.lower()

    with open(image_path, "rb") as image_file:
        if is_trellis2:
            input_params = {
                "image": image_file,
                "generate_model": True,
                "generate_video": _env("REPLICATE_GENERATE_VIDEO", "false").lower()
                in ("true", "1", "yes"),
                "preprocess_image": True,
                "pipeline_type": _env("REPLICATE_PIPELINE_TYPE", DEFAULT_PIPELINE_TYPE),
                "texture_size": int(_env("REPLICATE_TEXTURE_SIZE", "1024") or "1024"),
                "randomize_seed": True,
            }
        else:
            input_params = {
                "images": [image_file],
                "generate_model": True,
                "generate_color": False,
                "generate_normal": False,
                "texture_size": int(_env("REPLICATE_TEXTURE_SIZE", "1024") or "1024"),
                "mesh_simplify": 0.9,
                "randomize_seed": True,
            }
        return client.run(model_ref, input=input_params)


def _run_with_retries(model_slug: str, image_path: str) -> object:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return _run_replicate_once(model_slug, image_path)
        except Exception as exc:
            last_err = exc
            if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                time.sleep(8 * (attempt + 1))
                continue
            raise
    raise last_err  # type: ignore[misc]


def reconstruct_replicate(
    images: dict[str, Image.Image],
    output_dir: Path,
    job_id: str,
    use_fixed_poses: bool = True,
) -> dict:
    _ = use_fixed_poses
    model = _env("REPLICATE_MODEL", DEFAULT_REPLICATE_MODEL)
    tmp_path = _prepare_front_image(images["front"])

    try:
        models_to_try = [model]
        if model != FALLBACK_REPLICATE_MODEL:
            models_to_try.append(FALLBACK_REPLICATE_MODEL)

        output = None
        used_model = model

        for candidate in models_to_try:
            try:
                output = _run_with_retries(candidate, tmp_path)
                used_model = candidate
                break
            except Exception as exc:
                if candidate == models_to_try[-1]:
                    raise RuntimeError(_format_replicate_error(exc)) from exc

        glb_url = _extract_model_url(output)
        if not glb_url:
            raise RuntimeError(f"Replicate returned no GLB URL. Output: {output!r}")

        glb_path = output_dir / f"{job_id}.glb"
        _download_glb(glb_url, glb_path)

        return {
            "glb_path": glb_path,
            "point_cloud_path": None,
            "model": f"replicate/{used_model}",
            "message": (
                "Real textured GLB mesh from TRELLIS (Replicate). "
                "Built from your front photo."
            ),
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
