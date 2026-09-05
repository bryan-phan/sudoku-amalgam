sudoku-amalgam — Computer Vision Sudoku Solver

Reads a photo of a Sudoku puzzle and returns the completed board. The vision front end finds the grid and recognizes the givens; the logic back end solves it with constraint propagation before falling back to search. Solves 142/142 puzzles in the test set.

Written in Python with OpenCV and NumPy.

Pipeline
puzzle image ──> board detection ──> perspective warp ──> 81 cell crops ──> digit recognition ──> grid ──> solver ──> solved board
                     (cv/)                                                       (cv/)                        (logic/)
Find the board. Detect the puzzle's outer contour and correct the perspective distortion so a photo taken at an angle becomes a flat, square grid.
Segment the cells. Split the warped grid into 81 individual cell crops.
Read the givens. Extract a digit mask from each cell, match it against generated digit templates, and reject low-confidence matches before committing anything to the grid, so a misread is left blank rather than guessed wrong.
Solve. Apply ten constraint-solving techniques to prune the search space, then finish with backtracking search only where logic alone cannot.
Results
142 / 142 test images solved.
Constraint propagation reduces the backtracking search by 99.83%, from 1,133,651 recursive calls down to 1,887, before any brute-force search begins.
Project structure
cv/         image processing: board detection, warping, cell segmentation, digit recognition
logic/      the constraint solver and backtracking search
test_imgs/  the puzzle images used to validate the pipeline

Separating the vision (cv/) from the solving (logic/) keeps each half independently testable: the solver can run on a known grid, and the vision output can be checked without solving.

Usage
bash
# point the solver at one of the sample puzzles
python main.py test_imgs/<puzzle>.jpg
Tech stack

Python, OpenCV, NumPy, template matching, constraint propagation, backtracking search
