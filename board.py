
class Board:
    # For board
    def __init__(self):
        self.grid = [[0 for i in range(9)] for j in range(9)]

    def set(self, row, col, value):
        if not (0 <= row < 9 and 0 <= col < 9 and 1 <= value <= 9):
            return False

        if self.grid[row][col] != 0:
            return False

        if not self.is_valid(row, col, value):
            return False

        self.grid[row][col] = value
        return True
    
    def clear(self, row, col):
        if not (0 <= row < 9 and 0 <= col < 9):
            return False

        self.grid[row][col] = 0
        return True

    def get(self, row, col):
        if 0 <= row < 9 and 0 <= col < 9:
            return self.grid[row][col]
        return None

    def print_board(self):
        for r in range(9):
            if r % 3 == 0 and r != 0:
                print("-" * 21)

            for c in range(9):
                if c % 3 == 0 and c != 0:
                    print("|", end=" ")

                val = self.grid[r][c]
                print(val if val != 0 else ".", end=" ")

            print()

    # Checking validity of board
    def is_valid(self, row, col, value):
        for c in range(9):
            if self.grid[row][c] == value:
                return False

        for r in range(9):
            if self.grid[r][col] == value:
                return False

        box_row_start = (row // 3) * 3
        box_col_start = (col // 3) * 3
        
        for r in range(box_row_start, box_row_start + 3):
            for c in range(box_col_start, box_col_start + 3):
                if self.grid[r][c] == value:
                    return False

        return True
        
    def load(self, grid):
        self.grid = [row[:] for row in grid]



    


    
 


    