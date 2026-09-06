import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv.scan import Scan
from ocr.pipeline import DATA_DIR, ROOT, get_recognizer, process_grid_inputs

SUPPORTED_IMAGE_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_INPUT_DIR = DATA_DIR / "images" / "batch_drop" / "phone_photos"
DEFAULT_OUTPUT_PATH = DATA_DIR / "metrics" / "metrics_labels.json"
DEFAULT_CELL_EXPORT_ROOT = DATA_DIR / "exports" / "batch_cell_export"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan a folder of Sudoku photos and save quantitative metrics/predictions."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Folder to scan. Defaults to {DEFAULT_INPUT_DIR.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"JSON metrics output path. Defaults to {DEFAULT_OUTPUT_PATH.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Only measure decode/grid/cell extraction, without digit recognition.",
    )
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Run OCR but skip Sudoku solving.",
    )
    parser.add_argument(
        "--max-backtracking-nodes",
        type=int,
        default=5000,
        help="Per-image solve cap for OCR grids that need backtracking.",
    )
    parser.add_argument(
        "--export-cells",
        action="store_true",
        help=f"Export each image's 81 cell crops under {DEFAULT_CELL_EXPORT_ROOT.relative_to(ROOT)}.",
    )
    return parser.parse_args()


def relative_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def image_paths(folder):
    folder = Path(folder)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def rounded(value, digits=4):
    return round(float(value), digits)


def summarize_array(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}

    return {
        "count": int(values.size),
        "min": rounded(np.min(values)),
        "max": rounded(np.max(values)),
        "mean": rounded(np.mean(values)),
        "median": rounded(np.median(values)),
        "std": rounded(np.std(values)),
    }


def shape_payload(array):
    height, width = array.shape[:2]
    payload = {"height": int(height), "width": int(width)}
    if array.ndim == 3:
        payload["channels"] = int(array.shape[2])
    return payload


def export_cells(scanner, image_path):
    output_dir = DEFAULT_CELL_EXPORT_ROOT / image_path.stem
    scanner.export_cells(output_dir)
    return output_dir


def evaluate_image(path, recognizer, args):
    record = {
        "file": relative_path(path),
        "name": path.name,
        "extension": path.suffix.lower(),
        "bytes": int(path.stat().st_size),
        "decode_ok": False,
        "scan_ok": False,
        "ocr_ok": False,
        "solved": False,
    }
    timings = {}
    total_start = perf_counter()

    scanner = Scan(path)
    try:
        start = perf_counter()
        image = scanner.load_image()
        timings["decode"] = rounded(perf_counter() - start)
        record["decode_ok"] = True
        record["original_shape"] = shape_payload(image)

        start = perf_counter()
        contour = scanner.find_grid_contour(image)
        warped = scanner.warp(image=image, contour=contour)
        cells = scanner.splice(warped)
        timings["scan"] = rounded(perf_counter() - start)

        record["scan_ok"] = True
        record["warped_shape"] = shape_payload(warped)
        record["contour_area_px"] = rounded(cv2.contourArea(contour.astype(np.float32)), 2)
        record["cell_count"] = int(np.size(cells))
    except Exception as exc:
        timings["total"] = rounded(perf_counter() - total_start)
        record["timing_sec"] = timings
        record["error"] = {"stage": "scan", "type": type(exc).__name__, "message": str(exc)}
        return record

    cell_export_dir = None
    if args.export_cells:
        cell_export_dir = export_cells(scanner, path)
        record["cell_export_dir"] = relative_path(cell_export_dir)

    if not args.skip_ocr:
        try:
            start = perf_counter()
            result = process_grid_inputs(
                cells,
                recognizer=recognizer,
                cell_export_dir=cell_export_dir,
                predictions_path=None,
                solve=not args.skip_solve,
                use_backtracking=True,
                max_backtracking_nodes=args.max_backtracking_nodes,
            )
            timings["ocr_and_solve"] = rounded(perf_counter() - start)

            predictions = np.asarray(result["predictions"], dtype=int)
            scores = np.asarray(result["scores"], dtype=float)
            digit_scores = scores[predictions > 0]

            record.update(
                {
                    "ocr_ok": True,
                    "given_count": int(result["given_count"]),
                    "raw_given_count": int(result.get("raw_given_count", result["given_count"])),
                    "recognized_cell_ratio": rounded(result["given_count"] / 81.0),
                    "solved": bool(result["solved"]),
                    "predictions": result["predictions"],
                    "scores": result["scores"],
                    "score_summary": summarize_array(scores),
                    "recognized_score_summary": summarize_array(digit_scores),
                }
            )

            if result.get("solution") is not None:
                record["solution"] = result["solution"]
            if result.get("raw_predictions") is not None:
                record["raw_predictions"] = result["raw_predictions"]
            if result.get("repair_actions"):
                record["repair_actions"] = result["repair_actions"]
                record["repair_removed_count"] = int(result["repair_removed_count"])
            if result.get("conflicts"):
                record["conflict_count"] = len(result["conflicts"])
                record["conflicts"] = result["conflicts"]
            if result.get("warning"):
                record["warning"] = result["warning"]
        except Exception as exc:
            record["error"] = {"stage": "ocr", "type": type(exc).__name__, "message": str(exc)}

    timings["total"] = rounded(perf_counter() - total_start)
    record["timing_sec"] = timings
    return record


def aggregate(records):
    total = len(records)
    scan_times = [
        record["timing_sec"]["scan"]
        for record in records
        if record.get("scan_ok") and "scan" in record.get("timing_sec", {})
    ]
    ocr_times = [
        record["timing_sec"]["ocr_and_solve"]
        for record in records
        if record.get("ocr_ok") and "ocr_and_solve" in record.get("timing_sec", {})
    ]
    given_counts = [record["given_count"] for record in records if record.get("ocr_ok")]
    raw_given_counts = [record["raw_given_count"] for record in records if record.get("ocr_ok")]
    repair_removed_counts = [
        record.get("repair_removed_count", 0)
        for record in records
        if record.get("ocr_ok")
    ]

    return {
        "total_images": total,
        "decode_success": sum(1 for record in records if record.get("decode_ok")),
        "scan_success": sum(1 for record in records if record.get("scan_ok")),
        "ocr_success": sum(1 for record in records if record.get("ocr_ok")),
        "solve_success": sum(1 for record in records if record.get("solved")),
        "scan_success_rate": rounded(sum(1 for record in records if record.get("scan_ok")) / total) if total else 0.0,
        "ocr_success_rate": rounded(sum(1 for record in records if record.get("ocr_ok")) / total) if total else 0.0,
        "solve_success_rate": rounded(sum(1 for record in records if record.get("solved")) / total) if total else 0.0,
        "scan_time_sec": summarize_array(scan_times),
        "ocr_and_solve_time_sec": summarize_array(ocr_times),
        "given_count": summarize_array(given_counts),
        "raw_given_count": summarize_array(raw_given_counts),
        "repair_removed_count": summarize_array(repair_removed_counts),
    }


def main():
    args = parse_args()
    paths = image_paths(args.folder)
    recognizer = None if args.skip_ocr else get_recognizer()
    records = []

    for index, path in enumerate(paths, start=1):
        record = evaluate_image(path, recognizer, args)
        records.append(record)
        status = "OK" if record.get("scan_ok") else "FAIL"
        detail = f"givens={record.get('given_count', '-')}" if record.get("ocr_ok") else ""
        print(f"{index:02d}/{len(paths):02d} {path.name} {status} {detail}".rstrip())

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": relative_path(args.folder),
        "output_file": relative_path(args.output),
        "skip_ocr": bool(args.skip_ocr),
        "skip_solve": bool(args.skip_solve),
        "max_backtracking_nodes": int(args.max_backtracking_nodes),
        "aggregate": aggregate(records),
        "images": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved {relative_path(output_path)}")


if __name__ == "__main__":
    main()
