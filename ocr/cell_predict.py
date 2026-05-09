import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv.scan import Scan
from logic.board import Board
from logic.digit_recognizer import DigitRecognizer
from logic.techniques import Techniques


def build_grid_paths(cell_export_path):
    grid_paths = np.empty((9, 9), dtype=object)

    for row in range(9):
        for col in range(9):
            grid_paths[row, col] = str(cell_export_path / f"cell_{row}_{col}.png")

    return grid_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize Sudoku givens from a board image or exported cells.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a Sudoku photo. Defaults to cv/test_imgs/angled.jpg",
    )
    parser.add_argument(
        "--cells-dir",
        help="Use an existing 9x9 folder of cell images instead of scanning a board photo.",
    )
    parser.add_argument(
        "--export-cells",
        action="store_true",
        help="Save raw cell crops to ml/cell_export after scanning the board image.",
    )
    return parser.parse_args()


def recognize_grid(recognizer, grid_inputs):
    predictions, scores = recognizer.recognize_grid(grid_inputs)

    for row in range(9):
        for col in range(9):
            print(f"({row},{col}): {predictions[row, col]}  score={scores[row, col]:.3f}")

    return predictions, scores


if __name__ == "__main__":
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    export_path = root / "ml" / "cell_export"

    if args.cells_dir:
        grid_inputs = build_grid_paths(Path(args.cells_dir))
    else:
        scanner = Scan(args.image)
        cells = scanner.extract_cells()
        grid_inputs = cells

        if args.export_cells:
            scanner.export_cells(export_path)
            print(f"[OK] Exported raw cells to {export_path}")

    recognizer = DigitRecognizer()

    print("Recognizing Sudoku grid...")
    sudoku_predictions, scores = recognize_grid(recognizer, grid_inputs)

    print("\n[OK] Sudoku Grid Predictions:")
    print(sudoku_predictions)

    np.savetxt(root / "sudoku_predictions.txt", sudoku_predictions, fmt="%d")
    print("\n[DONE] Saved to sudoku_predictions.txt")

    given_count = int(np.count_nonzero(sudoku_predictions))
    if given_count < 12:
        print(f"\n[WARN] Only {given_count} digits recognized, skipping solve because the grid is too sparse.")
    else:
        board = Board()
        board.load(sudoku_predictions.tolist())
        tech = Techniques(board)

        print("\nSolving...")
        solved = tech.solve(use_backtracking=True)
        print(f"Solved = {solved}")

        print("\nFinal board:")
        board.print_board()
