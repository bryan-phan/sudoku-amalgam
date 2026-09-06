try:
    from .board import Board
except ImportError:
    from board import Board

try:
    from .techniques import Techniques
except ImportError:
    from techniques import Techniques

GRID =  [
    [6, 0, 0, 7, 4, 0, 0, 8, 0],
    [2, 0, 0, 0, 9, 0, 0, 0, 3],
    [0, 1, 0, 8, 0, 0, 0, 7, 9],
    [0, 6, 1, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 1, 3, 0],
    [9, 8, 0, 0, 0, 2, 0, 5, 0],
    [3, 0, 0, 0, 8, 0, 0, 0, 1],
    [0, 2, 0, 0, 3, 7, 0, 0, 6]
  ]

board = Board()
board.load(GRID)

tech = Techniques(board)

print("Before:")
board.print_board()

print("\nSolving...")
solved = tech.solve_logic()

print("\nSolved =", solved)
print("\nAfter:")
board.print_board()