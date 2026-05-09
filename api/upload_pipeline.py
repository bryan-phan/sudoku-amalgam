import contextlib
import io
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import HTTPException, Request, UploadFile

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cv.scan import Scan
from logic.board import Board
from logic.digit_recognizer import DigitRecognizer
from logic.techniques import Techniques

IMAGE_DIR = ROOT / "cv" / "test_imgs"
CELL_EXPORT_DIR = ROOT / "ml" / "cell_export"
PREDICTIONS_PATH = ROOT / "sudoku_predictions.txt"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/heic": ".heic",
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
    scanner = Scan(image_path=image_path)
    cells = scanner.extract_cells()

    if export_cells:
        scanner.export_cells(CELL_EXPORT_DIR)

    recognizer = DigitRecognizer()
    predictions, scores = recognizer.recognize_grid(cells)
    np.savetxt(PREDICTIONS_PATH, predictions, fmt="%d")

    given_count = int(np.count_nonzero(predictions))
    result = {
        "predictions": predictions.tolist(),
        "scores": np.round(scores, 4).tolist(),
        "given_count": given_count,
        "predictions_file": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "solved": False,
        "solution": None,
    }

    if export_cells:
        result["cell_export_dir"] = str(CELL_EXPORT_DIR.relative_to(ROOT))

    if given_count < 12:
        result["warning"] = (
            f"Only {given_count} digits recognized, so solving was skipped."
        )
        return result

    board = Board()
    board.load(predictions.tolist())
    techniques = Techniques(board)
    with contextlib.redirect_stdout(io.StringIO()):
        solved = bool(techniques.solve(use_backtracking=True))

    result["solved"] = solved
    result["solution"] = board.grid
    return result


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
