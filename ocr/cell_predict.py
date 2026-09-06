import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.pipeline import load_grid_inputs, process_grid_inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Recognize Sudoku givens from a board image or exported cells.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to a Sudoku photo. Defaults to data/images/angled.jpg",
    )
    parser.add_argument(
        "--cells-dir",
        help="Use an existing 9x9 folder of cell images instead of scanning a board photo.",
    )
    parser.add_argument(
        "--export-cells",
        action="store_true",
        help="Save raw cell crops to data/exports/cell_export after scanning the board image.",
    )
    return parser.parse_args()


def print_scores(predictions, scores):
    for row in range(9):
        for col in range(9):
            print(f"({row},{col}): {predictions[row, col]}  score={scores[row, col]:.3f}")


def main():
    args = parse_args()
    grid_inputs, cell_export_dir = load_grid_inputs(
        image_path=args.image,
        cells_dir=args.cells_dir,
        export_cells=args.export_cells,
    )
    if cell_export_dir is not None:
        print(f"[OK] Exported raw cells to {cell_export_dir}")

    print("Recognizing Sudoku grid...")
    result = process_grid_inputs(grid_inputs, cell_export_dir=cell_export_dir)
    sudoku_predictions = np.asarray(result["predictions"], dtype=int)
    scores = np.asarray(result["scores"], dtype=float)
    print_scores(sudoku_predictions, scores)

    print("\n[OK] Sudoku Grid Predictions:")
    print(sudoku_predictions)

    print(f"\n[DONE] Saved to {result['predictions_file']}")
    if "warning" in result:
        print(f"\n[WARN] {result['warning']}")
        return

    print("\nSolving...")
    print(f"Solved = {result['solved']}")
    print("\nFinal board:")
    print(np.asarray(result["solution"], dtype=int))


if __name__ == "__main__":
    main()
