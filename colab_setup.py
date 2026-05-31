"""
Run this FastAPI backend on Colab GPU + expose via ngrok.

From Colab notebook:
  %cd /content/backend
  !python colab_setup.py

Environment:
  HF_TOKEN, NGROK_AUTHTOKEN, RECONSTRUCTION_MODEL, BACKEND_DIR (optional)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or default).strip()


def check_gpu() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Set Runtime → GPU (T4) in Colab.")
        else:
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {name} ({mem:.1f} GB)")
    except ImportError:
        print("PyTorch not installed yet — will install in next step.")


def install_deps(backend_dir: Path, model_mode: str) -> None:
    print("Installing backend requirements…")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(backend_dir / "requirements.txt")]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "pyngrok"]
    )

    if model_mode in ("vgg-t3", "vggt3", "vgg_t3"):
        vgg_dir = Path("/content/vgg-ttt")
        if not vgg_dir.exists():
            print("Cloning NVIDIA VGG-T³…")
            subprocess.check_call(
                ["git", "clone", "--depth", "1", "https://github.com/nv-dvl/vgg-ttt.git", str(vgg_dir)]
            )
        print("Installing vgg-ttt…")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(vgg_dir / "requirements.txt")]
        )
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(vgg_dir)])
    elif model_mode == "vggt":
        print("Installing Meta VGGT…")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "git+https://github.com/facebookresearch/vggt.git",
            ]
        )


def write_env(backend_dir: Path, model_mode: str, hf_token: str) -> None:
    env_path = backend_dir / ".env"
    lines = [
        f"RECONSTRUCTION_MODEL={model_mode}",
        f"HF_TOKEN={hf_token}",
        "HF_MODEL_ID=nvidia/vgg-ttt",
        "HOST=0.0.0.0",
        "PORT=8000",
        "CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {env_path}")


def verify_inference(model_mode: str) -> tuple[bool, str]:
    if model_mode in ("mock",):
        return True, ""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return False, str(exc)

    if model_mode in ("vgg-t3", "vggt3", "vgg_t3"):
        try:
            from vggttt.nets.vggt.models.vggt import VGGT  # noqa: F401
        except ImportError as exc:
            return False, f"vgg-ttt import failed: {exc}"
    elif model_mode == "vggt":
        try:
            from vggt.models.vggt import VGGT  # noqa: F401
        except ImportError as exc:
            return False, f"vggt import failed: {exc}"
    return True, ""


def start_uvicorn(backend_dir: Path, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(backend_dir)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        cwd=str(backend_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def wait_for_health(port: int, timeout: int = 120) -> bool:
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(resp.read().decode())
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    return False


def start_cloudflared(port: int) -> str:
    """Free public URL — no signup required (Cloudflare quick tunnel)."""
    import re

    cf = Path("/usr/local/bin/cloudflared")
    if not cf.exists():
        print("Downloading cloudflared…")
        subprocess.check_call(
            [
                "wget",
                "-q",
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "-O",
                str(cf),
            ]
        )
        cf.chmod(0o755)

    proc = subprocess.Popen(
        [str(cf), "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 90
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line and proc.poll() is not None:
            break
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0).rstrip("/")
            print(f"\nPublic backend URL (cloudflared):\n  {public_url}\n")
            return public_url

    proc.kill()
    raise RuntimeError("cloudflared failed — try setting NGROK_AUTHTOKEN instead")


def start_ngrok(port: int) -> str:
    from pyngrok import ngrok

    token = _env("NGROK_AUTHTOKEN")
    if not token:
        raise ValueError("NGROK_AUTHTOKEN not set")

    ngrok.set_auth_token(token)
    tunnel = ngrok.connect(port, bind_tls=True)
    public_url = tunnel.public_url.rstrip("/")
    print(f"\nPublic backend URL (ngrok):\n  {public_url}\n")
    return public_url


def start_public_tunnel(port: int) -> str:
    if _env("NGROK_AUTHTOKEN"):
        try:
            return start_ngrok(port)
        except Exception as exc:
            print(f"ngrok failed ({exc}) — falling back to cloudflared…")

    print("Using cloudflared (free, no signup). Or set NGROK_AUTHTOKEN to use ngrok.")
    return start_cloudflared(port)


def main() -> None:
    backend_dir = Path(_env("BACKEND_DIR") or str(Path(__file__).resolve().parent)).resolve()
    port = int(_env("PORT", "8000"))
    model_mode = _env("RECONSTRUCTION_MODEL", "vgg-t3").lower()
    hf_token = _env("HF_TOKEN")

    if model_mode not in ("mock", "vggt", "vgg-t3", "vggt3"):
        print(f"Unknown RECONSTRUCTION_MODEL={model_mode!r}, using vgg-t3")
        model_mode = "vgg-t3"

    if model_mode != "mock" and not hf_token:
        print("ERROR: Set HF_TOKEN (Hugging Face → Settings → Access Tokens)")
        sys.exit(1)

    if not (backend_dir / "main.py").is_file():
        print(f"ERROR: main.py not found in {backend_dir}")
        sys.exit(1)

    check_gpu()
    install_deps(backend_dir, model_mode)
    write_env(backend_dir, model_mode, hf_token)

    ready, note = verify_inference(model_mode)
    if not ready:
        print(f"WARNING: inference not ready: {note}")
    else:
        print(f"Inference ready ({model_mode})")

    print(f"Starting uvicorn on port {port}…")
    proc = start_uvicorn(backend_dir, port)

    if not wait_for_health(port):
        print("Server failed to start — check Colab output above for errors.")
        proc.kill()
        sys.exit(1)

    public_url = start_public_tunnel(port)

    print("=" * 60)
    print("Connect your Mac frontend")
    print("=" * 60)
    print(f"1. Edit frontend/.env.local on your Mac:")
    print(f"   NEXT_PUBLIC_BACKEND_URL={public_url}")
    print("2. Restart: cd frontend && npm run dev")
    print("3. Open http://localhost:3000 → capture → Build 3D model")
    print("4. First run downloads ~2GB weights — stay on the preview page.")
    print("5. When done, set NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000 again.")
    print("=" * 60)
    print("Press Stop in Colab to shut down the GPU session.\n")

    try:
        while proc.poll() is None:
            time.sleep(30)
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
