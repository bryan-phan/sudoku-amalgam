try:
    from .board import Board
except ImportError:
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

    def best_empty_cell(self):
        best = None
        best_candidates = None

        for r in range(9):
            for c in range(9):
                if self.board.get(r, c) != 0:
                    continue

                candidates = self.cell_candidates(r, c)
                if not candidates:
                    return r, c, candidates

                if best is None or len(candidates) < len(best_candidates):
                    best = (r, c)
                    best_candidates = candidates

        if best is None:
            return None

        r, c = best
        return r, c, best_candidates

    # if all else fails
    def backtracking(self, max_nodes=None, state=None):
        if state is None:
            state = {"nodes": 0}
        state["calls"] = state.get("calls", 0) + 1

        empty = self.best_empty_cell()

        if empty is None:
            return True
        
        r, c, candidates = empty
        if not candidates:
            return False

        for num in sorted(candidates):
            state["nodes"] += 1
            if max_nodes is not None and state["nodes"] > max_nodes:
                return False

            if self.board.set(r, c, num):
                if self.backtracking(max_nodes=max_nodes, state=state):
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

