try:
    from .board import Board
except ImportError:
    from board import Board

try:
    from .solver import Solver
except ImportError:
    from solver import Solver
from itertools import combinations

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

    def all_houses(self):
        #for each row in the board
        for r in range(9):
            yield "Row", r, self.row_cells(r)

        #for each col
        for c in range(9):
            yield "Col", c, self.col_cells(c)

        #for each box
        for r in range(0, 9, 3):
            for c in range(0, 9, 3):
                yield "Box", (r, c), self.box_cells(r, c)

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

#hiddens
    def hidden_single(self): 

        #get a house
        for type, id, cells in self.all_houses():
            #run thru each cell in the house
            for nums, union_points in self.hidden_subsets(cells, 1):
                #num combo is eligible nums in a tuple
                #union points is a set
                num = nums[0]
                r, c = next(iter(union_points))

                if self.solver.place_value(r, c, num):
                    print(f"Hidden single found in {type}: {id}; Placed {num} at point {(r, c)}")
                    return True

        return False

    def hidden_pair(self):
        result = self.apply_hidden(2)
        if result:
            house_truth, house_type, house_ID, points = result
            print(f"Hidden pair at {house_type}: {house_ID} at point(s): {points}")
            return house_truth
        return False
    
    def hidden_triple(self):
        result = self.apply_hidden(3)
        if result:
            house_truth, house_type, house_ID, points = result
            print(f"Hidden triple at {house_type}: {house_ID} at point(s): {points}")
            return house_truth
        return False
    
    def hidden_quad(self):
        result = self.apply_hidden(4)
        if result:
            house_truth, house_type, house_ID, points = result
            print(f"Hidden quad at {house_type}: {house_ID} at point(s): {points}")
            return house_truth
        return False

#nakeds    
    def naked_single(self):
        '''
        If a cell has only one candidate, it's a naked single so js fill it in
        '''
        #if i didn't use list, it would raise an error if place_value changed the length dict
        for (r, c), candidates in list(self.solver.candidates.items()):
            if len(candidates) == 1:
                num = next(iter(candidates))
                if self.solver.place_value(r, c, num):
                    print(f"Naked Single placed {num} at point {r, c}. ")
                    return True

        return False
        #if changed, run this function again 
        
    def naked_pair(self):
        result = self.apply_naked(2)
        if result:
            house_truth, house_type, house_ID, points = result
            print(f"Naked pair at {house_type}: {house_ID} at point(s): {points}")
            return house_truth
        return False
    
    def naked_triple(self):
        result = self.apply_naked(3)
        if result:
            house_truth, house_type, house_ID, points = result
            print(f"Naked triple at {house_type}: {house_ID} at point(s): {points}")
            return house_truth
        return False

    def naked_quad(self):
        result = self.apply_naked(4)
        if result:
            house_truth, house_type, house_ID, points = result
            print(f"Naked quad at {house_type}: {house_ID} at point(s): {points}")
            return house_truth
        return False

#helper functions for naked and hiddens
    def naked_subsets(self, cells, size):
        #get all cells that are possible for checking for size 'size' nakeds
        eligible = []

        for r, c in cells:
            #since cells doesn't care if filled or not
            if (r, c) not in self.solver.candidates:
                continue
            
            #get the candidates from the cell
            candidates = self.solver.candidates[(r, c)]

            if 1 < len(candidates) <= size:
                eligible.append((r, c))

        results = []

        for combo_points in combinations(eligible, size):
            #make the union set
            union_nums = set()

            #for points in each combo, add it to the union set
            for point in combo_points:
                #note that this union candidates, NOT points
                union_nums |= self.solver.candidates[point]
                
            #if the union set == size, then TRUE; add the combo and set to results
            #combo to know which cells are the nakeds
            #union_set so you know which cells to eliminate/keep
            if len(union_nums) == size:
                results.append((combo_points, union_nums))
        
        return results
            
    def apply_naked(self, size):
        for type, id, cells in self.all_houses():
            for combo_points, union_nums in self.naked_subsets(cells, size):
                changed = False
                for (r, c) in cells:

                    if (r, c) in combo_points or (r, c) not in self.solver.candidates:
                        continue

                    before = set(self.solver.candidates[(r, c)])
                    self.solver.candidates[(r, c)] -= union_nums
                    if before != self.solver.candidates[(r, c)]:
                        removed = before - self.solver.candidates[(r, c)]
                        print(f"Naked {size} removed {sorted(removed)} from {(r, c)} in {type} {id}")
                        changed = True

                if changed:
                    return True, type, id, combo_points
                     
        return False
               
    def hidden_subsets(self, cells, size):

        #do the same thing as naked subsets but by points
        positions = self.house_cells(cells)

        results = []
        eligible = []

        for num, points in positions.items():
            if 1 <= len(points) <= size:
                eligible.append(num)

        for combo in combinations(eligible, size):
            union_points = set()

            #unions all the points for that number
            for num in combo:
                union_points |= set(positions[num])

            #if the set is equal to size, its a hidden subset
            if len(union_points) == size:
                results.append((combo, union_points))

        return results

    def apply_hidden(self, size):
        for type, id, cells in self.all_houses():
            for nums, union_points in self.hidden_subsets(cells, size):
                changed = False
                #this part can be simplified to just for (r, c) in union points but this makes more sense conceptually (to me at least)
                for (r, c) in cells:

                    if (r, c) not in union_points or (r, c) not in self.solver.candidates:
                        continue

                    #compare before and after to see if any change was made
                    before = set(self.solver.candidates[(r, c)])
                    self.solver.candidates[(r, c)] &= set(nums)

                    if before != self.solver.candidates[(r, c)]:
                        removed = before - self.solver.candidates[(r, c)]
                        print(f"Hidden {size} removed {sorted(removed)} from {(r, c)} in {type} {id}")
                        changed = True

                if changed:
                    return True, type, id, union_points

        return False

    #box-line reduction
    '''
    use hidden pair approach. 
    look thru a box, (use box function)
    sort them by how many time a number appears
    BUT it has to be in the same row or col


    basically, within a box, go thru each row or col again

    '''

    def boxline_reduction(self):
        for r in range(0, 9 , 3):
            for c in range(0, 9, 3):
                positions = self.house_cells(self.box_cells(r, c))
                #house_cells is a dictionary --> 0: (x, y)

                #iterate thru each number
                for num, points in positions.items():
                    #hidden single case
                    if len(points) < 2:
                        continue

                    rows = set(r for r, _ in points)
                    #if a number all share a row (i.e. they share the same candidates), the set should be 1 element
                    if len(rows) == 1:
                        row = next(iter(rows))
                        remove = [(row, col) for col in range(9) if not c <= col < c + 3]
                        #now remove is a list of tuples (tuples with points) that we need to remove
                        #go thru each point and remove that number
                        for x, y in remove:
                            if (x, y) not in self.solver.candidates:
                                continue

                            before = set(self.solver.candidates[(x, y)])
                            self.solver.candidates[(x, y)].discard(num)
                            if before != self.solver.candidates[(x, y)]:
                                print(f"Box-line reduction removed {num} from {(x, y)} // {num} is locked to row {row} in box {(r, c)}")
                                return True

                    #do the same as rows
                    cols = set(c for _, c in points)
                    if len(cols) == 1:
                        col = next(iter(cols))
                        remove = [(row, col) for row in range(9) if not r <= row < r + 3]
                        for x, y in remove:
                            if (x, y) not in self.solver.candidates:
                                continue

                            before = set(self.solver.candidates[(x, y)])
                            self.solver.candidates[(x, y)].discard(num)
                            if before != self.solver.candidates[(x, y)]:
                                print(f"Box-line reduction removed {num} from {(x, y)} // {num} is locked to col {col} in box {(r, c)}")
                                return True
        return False
    
    def linebox_reduction(self):

        for r in range(9):
            positions = self.house_cells(self.row_cells(r))

            for num, points in positions.items():
                if len(points) < 2:
                    continue
                
                #sees if the row is also inside a box
                box_cols = set(c // 3 for _, c in points)

                #checks if box share the same candidate
                if len(box_cols) == 1:
                    #gets the box index
                    box_r = (r // 3) * 3
                    box_c = next(iter(box_cols)) * 3
                    changed = False
                    
                    #now grab all candidates inside box
                    for row, col in self.box_cells(box_r, box_c):
                        #skips box that we're keeping
                        if row == r or (row, col) not in self.solver.candidates:
                            continue
                        
                        if num in self.solver.candidates[(row, col)]:
                            self.solver.candidates[(row, col)].discard(num)
                            print(f"Line-box reduction removed {num} from {(row, col)} // row {r} locks {num} into box {(box_r, box_c)}")
                            changed = True
                         
                    if changed:
                        print(f"Line-box reduction removed {num} from {(row, col)} // row {r} locks {num} into box {(box_r, box_c)}")
                        return changed


        for c in range(9):
            positions = self.house_cells(self.col_cells(c))

            for num, points in positions.items():
                if len(points) < 2:
                    continue
                
                #sees if the row is also inside a box
                box_rows = set(r // 3 for r, _ in points)

                #checks if box share the same candidate
                if len(box_rows) == 1:
                    #gets the box index
                    box_c = (c // 3) * 3
                    box_r = next(iter(box_rows)) * 3
                    changed = False
                    
                    #now grab all candidates inside box
                    for row, col in self.box_cells(box_r, box_c):
                        #skips box that we're keeping
                        if col == c or (row, col) not in self.solver.candidates:
                            continue
                        
                        if num in self.solver.candidates[(row, col)]:
                            self.solver.candidates[(row, col)].discard(num)
                            print(f"Line-box reduction removed {num} from {(row, col)} // col {c} locks {num} into box {(box_r, box_c)}")
                            changed = True
                         
                    if changed:
                        print(f"Line-box reduction removed {num} from {(row, col)} // col {c} locks {num} into box {(box_r, box_c)}")
                        return changed
                        
        return False
    
#solves the puzzle 
    def solve_logic(self):
        techniques = [
            self.naked_single,
            self.hidden_single,
            self.boxline_reduction,
            self.linebox_reduction,
            self.naked_pair,
            self.hidden_pair,
            self.naked_triple,
            self.hidden_triple,
            self.naked_quad,
            self.hidden_quad,
        ]

        while True:
            if self.solver.empty_cell() is None:
                return True

            for technique in techniques:
                if technique():
                    break   # restart from naked_single
            else:
                return False

    def solve(self, use_backtracking=False, max_backtracking_nodes=None):
        if self.solve_logic():
            return True

        if not use_backtracking:
            return False

        solved = self.solver.backtracking(max_nodes=max_backtracking_nodes)

        if solved:
            self.solver.initialize_candidates()

        return solved


