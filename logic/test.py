from board import Board
from techniques import Techniques

GRID = [
    [0,0,0, 0,0,0, 4,0,0],
    [2,0,0, 1,0,0, 5,9,0],
    [0,0,0, 7,0,3, 0,0,8],

    [6,9,0, 0,0,0, 0,0,7],
    [0,8,0, 0,0,0, 0,6,0],
    [0,1,0, 0,0,0, 9,3,0],

    [0,0,0, 5,4,0, 0,0,1],
    [0,0,0, 0,2,9, 3,0,0],
    [0,0,0, 0,6,0, 0,0,0]
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