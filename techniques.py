from board import Board
from solver import Solver

'''
I don't think I'll make a class for this. I guess one approach of this would be to try one technique and see if it can be applied anywhere on the board. 
If not, proceed with the next technique.

This sounds pretty hard considering things like hidden pairs and triples can happen on the vertical or horizontal

Okay so, after filling in a cell, it should remove any candidates within its house and then reasses 

Techniques should go in order of magnitude. so cell count for example. 
if i do techinque with 2 cells, i should run techinque with 1 cell right after

'''
'''After updating the board, it should its candidates'''


"""
        i need to make helper functions... 

        okay so heres the plan:

        I need to get all possible candidates for each house. So..., i'll make a dict for keys being the numbers and its value with the points
        If theres only one point inside the number, its a hidden single
        If theres only two points iside the number, and another cell has those same numbers, hidden pair
        same logic for hidden triples

        The idea is that for anything beyond singles, it NEEDS to remove the candidates in the house. 

        That means any technique involving singles will justww solve the cell. SO. hidden single and naked single will be our cell fill ins.
"""


class Techniques:

    def __init__(self, board):
        self.solver = Solver(board)

    def row_cells(self, r):
        return [(r, c) for c in range(9)]
    
    def col_cells(self, c):
        return [(r, c) for r in range(9)]
    
    def box_cells(self, r, c):
        cells = []

        box_row_start = (r // 3) * 3
        box_col_start = (c // 3) * 3

        for r in range(box_row_start, box_row_start + 3):
            for c in range(box_col_start, box_col_start + 3):
                cells.append((r, c))

        return cells

    def house_cells(self, cells):
        '''
        hashing sets with 1-9 as its keys and the sets as the value
        '''
        positions = {n: [] for n in range(1, 10)}

        for r, c in cells:
            if (r, c) not in self.solver.candidates:
                continue

            candidates = self.solver.candidates[(r, c)]

            for num in candidates:
                positions[num].append((r, c))

        return positions

    def naked_single(self):
        '''
        If a cell has only one candidate, it's a naked single so js fill it in
        '''
        #if i didn't use list, it would raise an error if place_value changed the length dict
        for (r, c), candidates in list(self.solver.candidates.items()):
            if len(candidates) == 1:
                if self.solver.place_value(r, c, next(iter(candidates))):
                    print(f"Naked Single used at {r, c}. ")
                    return True

        return False
        #if changed, run this function again 

    def hidden_single(self): 
        '''
        Assume that a cell has multiple candidates...
        If a number can only exist within a house (3x3, row, or col), then it is a hidden single
        '''

        for r in range(9):
            row_check = self.house_cells(self.row_cells(r))
            for num, cells in row_check.items():
                if len(cells) == 1:
                    row, col = cells[0]
                    if self.solver.place_value(row, col, num):
                        print(f"Hidden single used at {row, col}")
                        return True

        for c in range(9):
            col_check = self.house_cells(self.col_cells(c))
            for num, cells in col_check.items():
                if len(cells) == 1:
                    row, col = cells[0]
                    if self.solver.place_value(row, col, num):
                        print(f"Hidden single used at {row, col}")
                        return True
                
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box_check = self.house_cells(self.box_cells(r, c))
                for num, cells in box_check.items():
                    if len(cells) == 1:
                        row, col = cells[0]
                        if self.solver.place_value(row, col, num):
                            print(f"Hidden single used at {row, col}")
                            return True
                    
        return False

# from this point beyond, every function WILL call naked and hidden single to fill in cell
    def pair_candidates(self, cells):
        #bascially house_cells but for pairs instead
        pair_map = {}

        for r, c in cells:
            if (r, c) not in self.solver.candidates:
                continue
                
            candidates = self.solver.candidates[(r, c)]

            if len(candidates) == 2:
                pair = tuple(sorted(candidates))

                if pair not in pair_map:
                    pair_map[pair] = []
                
                pair_map[pair].append((r, c))
        
        return pair_map

    def naked_pair(self):
        change = False
        '''
        can i even hash by numbers anymore?
        I have to look at it by houses again
        '''
        #start with rows
        for r in range(9):
            row_check = self.pair_candidates(self.row_cells(r))
            for pair, positions in row_check.items():
                if len(positions) == 2:
                    
                    p1, p2 = pair
                    for c in range(9):
                        if (r, c) in self.solver.candidates and (r, c) not in positions:
                            before = set(self.solver.candidates[(r, c)])
                            self.solver.candidates[(r, c)].discard(p1)
                            self.solver.candidates[(r, c)].discard(p2)

                            if self.solver.candidates[(r, c)] != before:
                                print(f"Naked pair: {pair} Row: {r} Positions: {positions} ")
                                return True

        for c in range(9):        
            col_check = self.pair_candidates(self.col_cells(c))
            for pair, positions in col_check.items():
                if len(positions) == 2:
                    p1, p2 = pair
                    for r in range(9):
                        if (r, c) in self.solver.candidates and (r, c) not in positions:
                            before = set(self.solver.candidates[(r, c)])
                            self.solver.candidates[(r, c)].discard(p1)
                            self.solver.candidates[(r, c)].discard(p2)
                            if self.solver.candidates[(r, c)] != before:
                                print(f"Naked pair: {pair} Col: {c} Positions: {positions} ")
                                return True

        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                box_check = self.pair_candidates(self.box_cells(r, c))
                for pair, positions in box_check.items():
                    if len(positions) == 2:
                        p1, p2 = pair
                        for row, col in self.box_cells(r, c):
                            if (row, col) in self.solver.candidates and (row, col) not in positions:
                                before = set(self.solver.candidates[(row, col)])
                                self.solver.candidates[(row, col)].discard(p1)
                                self.solver.candidates[(row, col)].discard(p2)
                                if self.solver.candidates[(row, col)] != before:
                                    print(f"Naked pair: {pair} Box: ({r}, {c}) Positions: {positions}")
                                    return True

        return change
    
    #def hidden_pair(self):
    



