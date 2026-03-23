from board import Board

class Solver:
    def __init__(self, board):
        self.board = board
        self.candidates = {}
        self.initialize_candidates()

    def initialize_candidates(self):
        self.candidates = {}

        for r in range(9):
            for c in range(9):
                if self.board.get(r, c) == 0:
                    self.candidates[(r, c)] = self.cell_candidates(r, c)

    def place_value(self, r, c, value):
        if not (self.board.set(r, c, value)):
            return False

        # delete candidates for filled cell
        if (r, c) in self.candidates:
            del self.candidates[(r, c)]

        # deletes the filled cell from all candidates within its house
        for col in range(9):
            if (r, col) in self.candidates:
                self.candidates[(r, col)].discard(value)

        for row in range(9):
            if (row, c) in self.candidates:
                self.candidates[(row, c)].discard(value)

        box_row_start = (r // 3) * 3
        box_col_start = (c // 3) * 3

        for row in range(box_row_start, box_row_start + 3):
            for col in range(box_col_start, box_col_start + 3):
                if (row, col) in self.candidates:
                    self.candidates[(row, col)].discard(value)

        return True
    
    def empty_cell(self):
        for r in range(9):
            for c in range(9):
                if self.board.get(r, c) == 0:
                    return (r, c)
        
        return None  
    # if all else fails
    def backtracking(self):
        empty = self.empty_cell()

        if empty is None:
            return True
        
        r, c = empty

        for num in range(1, 10):
            if self.board.set(r, c, num):
                if self.backtracking():
                    return True
                
                self.board.clear(r, c)

        return False
    #big boy solving 
    def cell_candidates(self, r, c):
        candidates = set()

        for num in range(1, 10):
            if self.board.is_valid(r, c, num):
                candidates.add(num)
        
        return candidates

