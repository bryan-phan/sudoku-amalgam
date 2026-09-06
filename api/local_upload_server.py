import argparse
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.upload_pipeline import (
    IMAGE_DIR,
    MAX_UPLOAD_BYTES,
    process_image,
    read_upload_payload,
    save_upload,
)

app = FastAPI(title="Sudoku Upload API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local FastAPI server for uploading Sudoku photos into data/images."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to. Use 0.0.0.0 for phone access.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    return parser.parse_args()


@app.get("/")
async def root():
    return {
        "service": "sudoku-upload",
        "framework": "fastapi",
        "upload_only": "POST /upload",
        "upload_and_process": "POST /process?export_cells=1",
        "test_form": "GET /upload-form",
        "save_dir": str(IMAGE_DIR.relative_to(ROOT)),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.get("/upload-form", response_class=HTMLResponse)
async def upload_form():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sudoku Upload</title>
  <style>
    body { font-family: Segoe UI, sans-serif; margin: 2rem; }
    form { display: grid; gap: 0.75rem; max-width: 28rem; }
    button { width: fit-content; padding: 0.5rem 1rem; }
  </style>
</head>
<body>
  <h1>Sudoku Upload</h1>
  <form action="/process?export_cells=1" method="post" enctype="multipart/form-data">
    <input type="file" name="photo" accept="image/*" required>
    <button type="submit">Upload and process</button>
  </form>
  <p>Use <code>/upload</code> to save only or <code>/process</code> to save and solve.</p>
</body>
</html>"""


@app.post("/upload")
async def upload(
    request: Request,
    photo: UploadFile | None = File(default=None),
):
    filename, content_type, data = await read_upload_payload(request, photo)
    saved_path = save_upload(filename, content_type, data)
    return {
        "saved": True,
        "file": str(saved_path.relative_to(ROOT)),
    }


@app.post("/process")
async def process(
    request: Request,
    export_cells: bool = Query(default=False),
    photo: UploadFile | None = File(default=None),
):
    filename, content_type, data = await read_upload_payload(request, photo)
    saved_path = save_upload(filename, content_type, data)
    response = {
        "saved": True,
        "file": str(saved_path.relative_to(ROOT)),
    }

    try:
        response.update(process_image(saved_path, export_cells=export_cells))
    except Exception as exc:
        response["error"] = str(exc)
        return JSONResponse(status_code=422, content=response)

    return response


def main():
    args = parse_args()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
