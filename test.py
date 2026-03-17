from board import Board
from techniques import Techniques


GRID = [
    [0, 0, 0, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 0, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 0, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [0, 4, 0, 2, 8, 6, 1, 7, 9],
]


def print_candidates(tech, cells):
    for cell in cells:
        candidates = tech.solver.candidates.get(cell)
        print(cell, sorted(candidates) if candidates is not None else None)


board = Board()
board.load(GRID)
tech = Techniques(board)

watch = [(0, 0), (0, 1), (0, 2)]

print("Before board:")
board.print_board()

print("\nWatched candidates before naked_pair:")
print_candidates(tech, watch)

print("\nRun naked_pair:")
print("changed =", tech.naked_pair())

print("\nWatched candidates after naked_pair:")
print_candidates(tech, watch)

print("\nRun naked_single:")
print("changed =", tech.naked_single())

print("\nAfter board:")
board.print_board()

print("\nWatched candidates after naked_single:")
print_candidates(tech, watch)
