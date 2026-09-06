import argparse
import contextlib
import io
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logic.board import Board
from logic.solver import Solver
from logic.techniques import Techniques
from ocr.pipeline import DATA_DIR, ROOT

DEFAULT_METRICS_PATH = DATA_DIR / "metrics" / "metrics_labels.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "metrics" / "evaluation_stats.json"
DEFAULT_LABEL_TEMPLATE_PATH = DATA_DIR / "metrics" / "ground_truth_template.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR/solve metrics from metrics_labels.json."
    )
    parser.add_argument(
        "--metrics",
        default=str(DEFAULT_METRICS_PATH),
        help=f"Pipeline metrics JSON. Defaults to {DEFAULT_METRICS_PATH.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--labels",
        help="Optional ground-truth givens JSON for real digit recognition accuracy.",
    )
    parser.add_argument(
        "--label-template",
        default=str(DEFAULT_LABEL_TEMPLATE_PATH),
        help=f"Template to create when labels are missing. Defaults to {DEFAULT_LABEL_TEMPLATE_PATH.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output stats JSON. Defaults to {DEFAULT_OUTPUT_PATH.relative_to(ROOT)}.",
    )
    parser.add_argument(
        "--max-backtracking-calls",
        type=int,
        default=1_000_000,
        help="Cap each baseline/optimized backtracking comparison.",
    )
    return parser.parse_args()


def relative_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rounded(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def percent(numerator, denominator):
    return rounded(numerator / denominator) if denominator else None


def summarize(values):
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


def as_grid(value):
    grid = np.asarray(value, dtype=int)
    if grid.shape != (9, 9):
        raise ValueError(f"Expected a 9x9 grid, got {grid.shape}.")
    return grid


def load_label_map(labels_path):
    if labels_path is None:
        return {}

    payload = read_json(labels_path)
    images = payload.get("images", payload)
    if isinstance(images, list):
        return {
            entry.get("name") or Path(entry.get("file", "")).name: entry["ground_truth_grid"]
            for entry in images
            if "ground_truth_grid" in entry
        }

    return {
        name: entry.get("ground_truth_grid", entry)
        for name, entry in images.items()
    }


def create_label_template(metrics, output_path):
    images = {}
    for record in metrics.get("images", []):
        starter = record.get("predictions") or [[0 for _ in range(9)] for _ in range(9)]
        images[record["name"]] = {
            "file": record["file"],
            "label_status": "needs_review",
            "ground_truth_grid": starter,
        }

    payload = {
        "schema": "sudoku_ground_truth_givens_v1",
        "notes": (
            "Replace each ground_truth_grid with the actual printed givens from the photo. "
            "Use 0 for empty cells. The starter grids are current pipeline predictions, "
            "so review them before using digit accuracy."
        ),
        "images": images,
    }
    write_json(output_path, payload)


def compare_prediction_to_truth(prediction, truth):
    prediction = as_grid(prediction)
    truth = as_grid(truth)
    true_given = truth > 0
    predicted_given = prediction > 0
    correct_digits = (prediction == truth) & true_given & predicted_given

    substitutions = true_given & predicted_given & (prediction != truth)
    false_positives = predicted_given & ~true_given
    missed_givens = true_given & ~predicted_given

    return {
        "cell_accuracy": percent(int(np.sum(prediction == truth)), 81),
        "digit_precision": percent(int(np.sum(correct_digits)), int(np.sum(predicted_given))),
        "digit_recall": percent(int(np.sum(correct_digits)), int(np.sum(true_given))),
        "given_f1": f1_score(int(np.sum(correct_digits)), int(np.sum(predicted_given)), int(np.sum(true_given))),
        "exact_grid_match": bool(np.array_equal(prediction, truth)),
        "true_given_count": int(np.sum(true_given)),
        "predicted_given_count": int(np.sum(predicted_given)),
        "correct_digit_count": int(np.sum(correct_digits)),
        "substitution_count": int(np.sum(substitutions)),
        "false_positive_count": int(np.sum(false_positives)),
        "missed_given_count": int(np.sum(missed_givens)),
    }


def f1_score(correct, predicted, truth):
    precision = correct / predicted if predicted else 0.0
    recall = correct / truth if truth else 0.0
    if precision + recall == 0:
        return 0.0
    return rounded(2 * precision * recall / (precision + recall))


def digit_accuracy(metrics, labels):
    per_image = []
    raw_precision_values = []
    repaired_precision_values = []

    for record in metrics.get("images", []):
        truth = labels.get(record["name"]) or labels.get(record.get("file"))
        if truth is None:
            continue

        repaired = compare_prediction_to_truth(record["predictions"], truth)
        raw = compare_prediction_to_truth(record.get("raw_predictions", record["predictions"]), truth)
        per_image.append(
            {
                "name": record["name"],
                "raw": raw,
                "repaired": repaired,
            }
        )
        raw_precision_values.append(raw["digit_precision"] or 0.0)
        repaired_precision_values.append(repaired["digit_precision"] or 0.0)

    return {
        "available": bool(per_image),
        "labeled_images": len(per_image),
        "raw_digit_precision_mean": rounded(np.mean(raw_precision_values)) if raw_precision_values else None,
        "repaired_digit_precision_mean": rounded(np.mean(repaired_precision_values)) if repaired_precision_values else None,
        "per_image": per_image,
    }


def proxy_digit_consistency(metrics):
    per_image = []
    raw_values = []
    repaired_values = []

    for record in metrics.get("images", []):
        solution = record.get("solution")
        if solution is None:
            continue

        solution = as_grid(solution)
        repaired = as_grid(record["predictions"])
        raw = as_grid(record.get("raw_predictions", record["predictions"]))

        raw_mask = raw > 0
        repaired_mask = repaired > 0
        raw_consistency = percent(int(np.sum(raw[raw_mask] == solution[raw_mask])), int(np.sum(raw_mask)))
        repaired_consistency = percent(
            int(np.sum(repaired[repaired_mask] == solution[repaired_mask])),
            int(np.sum(repaired_mask)),
        )

        raw_values.append(raw_consistency or 0.0)
        repaired_values.append(repaired_consistency or 0.0)
        per_image.append(
            {
                "name": record["name"],
                "raw_solution_consistency": raw_consistency,
                "repaired_solution_consistency": repaired_consistency,
                "raw_given_count": int(np.sum(raw_mask)),
                "repaired_given_count": int(np.sum(repaired_mask)),
            }
        )

    return {
        "definition": (
            "Proxy only: predicted nonzero cells compared to the solved board. "
            "This is not true digit recognition accuracy because it cannot identify false positives "
            "that happen to match the solution in originally empty cells."
        ),
        "raw_mean": rounded(np.mean(raw_values)) if raw_values else None,
        "repaired_mean": rounded(np.mean(repaired_values)) if repaired_values else None,
        "per_image": per_image,
    }


def solve_rate(metrics):
    records = metrics.get("images", [])
    total = len(records)
    solved = sum(1 for record in records if record.get("solved"))
    scan_ok = sum(1 for record in records if record.get("scan_ok"))
    ocr_ok = sum(1 for record in records if record.get("ocr_ok"))

    return {
        "total_images": total,
        "scan_success": scan_ok,
        "ocr_success": ocr_ok,
        "solve_success": solved,
        "end_to_end_solve_rate": percent(solved, total),
    }


def valid_on_grid(grid, row, col, value):
    for c in range(9):
        if grid[row][c] == value:
            return False
    for r in range(9):
        if grid[r][col] == value:
            return False

    row_start = (row // 3) * 3
    col_start = (col // 3) * 3
    for r in range(row_start, row_start + 3):
        for c in range(col_start, col_start + 3):
            if grid[r][c] == value:
                return False
    return True


def first_empty(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                return row, col
    return None


def naive_backtracking(grid, state, max_calls):
    state["calls"] += 1
    if state["calls"] > max_calls:
        state["capped"] = True
        return False

    empty = first_empty(grid)
    if empty is None:
        return True

    row, col = empty
    for value in range(1, 10):
        if valid_on_grid(grid, row, col, value):
            grid[row][col] = value
            state["nodes"] += 1
            if naive_backtracking(grid, state, max_calls):
                return True
            grid[row][col] = 0

    return False


def logic_reduced_grid(grid):
    board = Board()
    board.load(deepcopy(grid))
    techniques = Techniques(board)
    with contextlib.redirect_stdout(io.StringIO()):
        solved_by_logic = bool(techniques.solve_logic())
    return deepcopy(board.grid), solved_by_logic


def optimized_backtracking(grid, max_calls):
    board = Board()
    board.load(deepcopy(grid))
    solver = Solver(board)
    state = {"nodes": 0, "calls": 0}
    solved = solver.backtracking(max_nodes=max_calls, state=state)
    return {
        "solved": bool(solved),
        "calls": int(state.get("calls", 0)),
        "nodes": int(state.get("nodes", 0)),
        "capped": bool(state.get("calls", 0) > max_calls or state.get("nodes", 0) > max_calls),
    }


def backtracking_reduction(metrics, max_calls):
    per_image = []
    baseline_calls = []
    optimized_calls = []
    capped = 0

    for record in metrics.get("images", []):
        if not record.get("solved"):
            continue

        reduced_grid, solved_by_logic = logic_reduced_grid(record["predictions"])
        if solved_by_logic:
            baseline = {"solved": True, "calls": 0, "nodes": 0, "capped": False}
            optimized = {"solved": True, "calls": 0, "nodes": 0, "capped": False}
        else:
            baseline_state = {"nodes": 0, "calls": 0, "capped": False}
            baseline_grid = deepcopy(reduced_grid)
            baseline_solved = naive_backtracking(baseline_grid, baseline_state, max_calls)
            baseline = {
                "solved": bool(baseline_solved),
                "calls": int(baseline_state["calls"]),
                "nodes": int(baseline_state["nodes"]),
                "capped": bool(baseline_state["capped"]),
            }
            optimized = optimized_backtracking(reduced_grid, max_calls)

        if baseline["calls"] > 0:
            baseline_calls.append(baseline["calls"])
            optimized_calls.append(optimized["calls"])
        if baseline["capped"] or optimized["capped"]:
            capped += 1

        per_image.append(
            {
                "name": record["name"],
                "logic_solved": bool(solved_by_logic),
                "baseline_naive": baseline,
                "optimized_mrv": optimized,
                "call_reduction": (
                    rounded(1.0 - optimized["calls"] / baseline["calls"])
                    if baseline["calls"] > 0
                    else None
                ),
            }
        )

    total_baseline = int(sum(baseline_calls))
    total_optimized = int(sum(optimized_calls))
    return {
        "definition": "Naive row-first backtracking calls vs optimized MRV backtracking calls after the same logic techniques run.",
        "max_calls_per_image": int(max_calls),
        "capped_comparisons": capped,
        "comparison_count": len(per_image),
        "aggregate_baseline_calls": total_baseline,
        "aggregate_optimized_calls": total_optimized,
        "aggregate_call_reduction": (
            rounded(1.0 - total_optimized / total_baseline)
            if total_baseline
            else None
        ),
        "baseline_calls": summarize(baseline_calls),
        "optimized_calls": summarize(optimized_calls),
        "per_image": per_image,
    }


def main():
    args = parse_args()
    metrics = read_json(args.metrics)
    labels = load_label_map(args.labels)
    label_template_path = Path(args.label_template)

    if not labels and not label_template_path.exists():
        create_label_template(metrics, label_template_path)

    accuracy = digit_accuracy(metrics, labels)
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_file": relative_path(args.metrics),
        "labels_file": relative_path(args.labels) if args.labels else None,
        "label_template_file": relative_path(label_template_path) if not labels else None,
        "digit_recognition_accuracy": accuracy,
        "proxy_digit_consistency": proxy_digit_consistency(metrics),
        "end_to_end_solve_rate": solve_rate(metrics),
        "backtracking_call_reduction": backtracking_reduction(metrics, args.max_backtracking_calls),
    }

    write_json(args.output, output)
    print(f"Saved {relative_path(args.output)}")
    if not accuracy["available"]:
        print(f"Ground-truth labels missing; created/reused {relative_path(label_template_path)}")


if __name__ == "__main__":
    main()
