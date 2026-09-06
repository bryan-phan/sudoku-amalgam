import re
import sys
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocr.pipeline import load_grid_inputs, process_grid_inputs

IMAGE_DIR = ROOT / "data" / "images"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/heic": ".heic",
    "image/heic-sequence": ".heic",
    "image/heif": ".heif",
    "image/heif-sequence": ".heif",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def sanitize_filename(filename):
    candidate = Path(filename or "").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")


def default_suffix(content_type):
    normalized = content_type.split(";", 1)[0].strip().lower()
    return ALLOWED_IMAGE_TYPES.get(normalized, ".img")


def build_destination_path(filename, content_type):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = sanitize_filename(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if cleaned:
        stem = Path(cleaned).stem or "upload"
        suffix = Path(cleaned).suffix or default_suffix(content_type)
        candidate = f"{timestamp}_{stem}{suffix}"
    else:
        candidate = f"{timestamp}{default_suffix(content_type)}"

    destination = IMAGE_DIR / candidate
    counter = 1
    while destination.exists():
        destination = IMAGE_DIR / f"{destination.stem}_{counter}{destination.suffix}"
        counter += 1

    return destination


def save_upload(filename, content_type, data):
    if not data:
        raise HTTPException(status_code=400, detail="Request body is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Upload exceeds {max_mb} MB limit.")

    destination = build_destination_path(filename, content_type)
    destination.write_bytes(data)
    return destination


def process_image(image_path, export_cells=False):
    grid_inputs, cell_export_dir = load_grid_inputs(image_path=image_path, export_cells=export_cells)
    return process_grid_inputs(grid_inputs, cell_export_dir=cell_export_dir)


async def read_upload_payload(request: Request, photo: UploadFile | None):
    if photo is not None:
        content_type = photo.content_type or "application/octet-stream"
        filename = photo.filename or ""
        data = await photo.read()
        await photo.close()
        return filename, content_type, data

    body = await request.body()
    filename = request.query_params.get("filename") or request.headers.get("X-Filename", "")
    content_type = request.headers.get("Content-Type", "application/octet-stream")
    return filename, content_type, body
