import contextlib
import io
from functools import lru_cache
from pathlib import Path

import numpy as np

from cv.scan import Scan
from logic.board import Board
from logic.digit_recognizer import DigitRecognizer
from logic.techniques import Techniques

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CELL_EXPORT_DIR = DATA_DIR / "exports" / "cell_export"
PREDICTIONS_PATH = DATA_DIR / "metrics" / "sudoku_predictions.txt"
MIN_GIVENS_TO_SOLVE = 12
MAX_BACKTRACKING_NODES = 50000
MAX_REPAIR_REMOVALS = 12


@lru_cache(maxsize=1)
def get_recognizer():
    return DigitRecognizer()


def build_grid_paths(cell_export_path):
    cell_export_path = Path(cell_export_path)
    return np.array(
        [
            [str(cell_export_path / f"cell_{row}_{col}.png") for col in range(9)]
            for row in range(9)
        ],
        dtype=object,
    )


def load_grid_inputs(image_path=None, cells_dir=None, export_cells=False):
    if cells_dir:
        return build_grid_paths(cells_dir), None

    scanner = Scan(image_path=image_path)
    cells = scanner.extract_cells()
    exported_dir = scanner.export_cells(CELL_EXPORT_DIR) if export_cells else None
    return cells, exported_dir


def relative_to_root(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _duplicate_values(cells):
    positions_by_value = {}
    for value, row, col in cells:
        if value == 0:
            continue

        positions_by_value.setdefault(int(value), []).append([int(row), int(col)])

    return [
        {"digit": digit, "cells": positions}
        for digit, positions in sorted(positions_by_value.items())
        if len(positions) > 1
    ]


def find_prediction_conflicts(predictions):
    predictions = np.asarray(predictions, dtype=int)
    conflicts = []

    for row in range(9):
        duplicates = _duplicate_values((predictions[row, col], row, col) for col in range(9))
        for duplicate in duplicates:
            conflicts.append({"house": "row", "index": row, **duplicate})

    for col in range(9):
        duplicates = _duplicate_values((predictions[row, col], row, col) for row in range(9))
        for duplicate in duplicates:
            conflicts.append({"house": "column", "index": col, **duplicate})

    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            cells = (
                (predictions[row, col], row, col)
                for row in range(box_row, box_row + 3)
                for col in range(box_col, box_col + 3)
            )
            duplicates = _duplicate_values(cells)
            for duplicate in duplicates:
                conflicts.append({"house": "box", "index": [box_row, box_col], **duplicate})

    return conflicts


def solve_predictions(predictions, use_backtracking=True, max_backtracking_nodes=MAX_BACKTRACKING_NODES):
    predictions = np.asarray(predictions, dtype=int)
    given_count = int(np.count_nonzero(predictions))
    if given_count < MIN_GIVENS_TO_SOLVE:
        warning = f"Only {given_count} digits recognized, so solving was skipped."
        return given_count, False, None, warning, []

    conflicts = find_prediction_conflicts(predictions)
    if conflicts:
        warning = (
            f"Recognized grid has {len(conflicts)} duplicate-house conflict(s), "
            "so solving was skipped."
        )
        return given_count, False, None, warning, conflicts

    board = Board()
    board.load(predictions.tolist())
    techniques = Techniques(board)
    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        solved = bool(
            techniques.solve(
                use_backtracking=use_backtracking,
                max_backtracking_nodes=max_backtracking_nodes,
            )
        )

    warning = None
    if not solved and use_backtracking:
        warning = f"Solver stopped before completion after {max_backtracking_nodes} backtracking nodes."

    return given_count, solved, board.grid if solved else None, warning, []


def _remove_cell(predictions, cell, reason, scores, conflicts_seen=None):
    row, col = cell
    action = {
        "row": int(row),
        "col": int(col),
        "digit": int(predictions[row, col]),
        "score": round(float(scores[row, col]), 4),
        "reason": reason,
    }
    if conflicts_seen is not None:
        action["conflicts_seen"] = int(conflicts_seen)

    predictions[row, col] = 0
    return action


def _remove_worst_conflict_cell(predictions, scores, conflicts):
    conflict_counts = {}
    for conflict in conflicts:
        for row, col in conflict["cells"]:
            cell = (int(row), int(col))
            conflict_counts[cell] = conflict_counts.get(cell, 0) + 1

    candidates = [
        (count, float(scores[cell]), cell)
        for cell, count in conflict_counts.items()
        if predictions[cell] != 0
    ]
    if not candidates:
        return None

    count, _, cell = max(candidates, key=lambda item: (item[0], item[1]))
    return _remove_cell(predictions, cell, "duplicate_conflict", scores, conflicts_seen=count)


def _remove_worst_scored_given(predictions, scores):
    candidates = [
        (float(scores[row, col]), (row, col))
        for row in range(9)
        for col in range(9)
        if predictions[row, col] != 0
    ]
    if not candidates:
        return None

    _, cell = max(candidates, key=lambda item: item[0])
    return _remove_cell(predictions, cell, "unsolved_low_confidence", scores)


def repair_and_solve_predictions(
    predictions,
    scores,
    *,
    use_backtracking=True,
    max_backtracking_nodes=MAX_BACKTRACKING_NODES,
    max_repair_removals=MAX_REPAIR_REMOVALS,
):
    repaired = np.asarray(predictions, dtype=int).copy()
    scores = np.asarray(scores, dtype=float)
    repair_actions = []

    while True:
        given_count, solved, solution, warning, conflicts = solve_predictions(
            repaired,
            use_backtracking=use_backtracking,
            max_backtracking_nodes=max_backtracking_nodes,
        )
        if solved:
            return repaired, given_count, True, solution, None, [], repair_actions

        if len(repair_actions) >= max_repair_removals or given_count <= MIN_GIVENS_TO_SOLVE:
            return repaired, given_count, False, solution, warning, conflicts, repair_actions

        if conflicts:
            action = _remove_worst_conflict_cell(repaired, scores, conflicts)
        else:
            action = _remove_worst_scored_given(repaired, scores)

        if action is None:
            return repaired, given_count, False, solution, warning, conflicts, repair_actions

        repair_actions.append(action)


def process_grid_inputs(
    grid_inputs,
    recognizer=None,
    cell_export_dir=None,
    predictions_path=PREDICTIONS_PATH,
    solve=True,
    use_backtracking=True,
    max_backtracking_nodes=MAX_BACKTRACKING_NODES,
    repair=True,
    max_repair_removals=MAX_REPAIR_REMOVALS,
):
    recognizer = get_recognizer() if recognizer is None else recognizer
    raw_predictions, scores = recognizer.recognize_grid(grid_inputs)
    predictions = raw_predictions

    repair_actions = []
    if predictions_path is not None:
        np.savetxt(predictions_path, raw_predictions, fmt="%d")

    if solve:
        if repair:
            (
                predictions,
                given_count,
                solved,
                solution,
                warning,
                conflicts,
                repair_actions,
            ) = repair_and_solve_predictions(
                raw_predictions,
                scores,
                use_backtracking=use_backtracking,
                max_backtracking_nodes=max_backtracking_nodes,
                max_repair_removals=max_repair_removals,
            )
        else:
            given_count, solved, solution, warning, conflicts = solve_predictions(
                raw_predictions,
                use_backtracking=use_backtracking,
                max_backtracking_nodes=max_backtracking_nodes,
            )
    else:
        given_count = int(np.count_nonzero(raw_predictions))
        solved = False
        solution = None
        warning = None
        conflicts = find_prediction_conflicts(raw_predictions)

    result = {
        "predictions": predictions.tolist(),
        "scores": np.round(scores, 4).tolist(),
        "given_count": given_count,
        "raw_given_count": int(np.count_nonzero(raw_predictions)),
        "solved": solved,
        "solution": solution,
    }

    if repair_actions:
        result["raw_predictions"] = raw_predictions.tolist()
        result["repair_actions"] = repair_actions
        result["repair_removed_count"] = len(repair_actions)

    if predictions_path is not None:
        np.savetxt(predictions_path, predictions, fmt="%d")
        result["predictions_file"] = relative_to_root(predictions_path)

    if cell_export_dir is not None:
        result["cell_export_dir"] = relative_to_root(cell_export_dir)

    if conflicts:
        result["conflicts"] = conflicts

    if warning is not None:
        result["warning"] = warning

    return result
