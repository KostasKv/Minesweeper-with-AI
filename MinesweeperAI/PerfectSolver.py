from Agent import Agent
from Game import Game
from random import randint
import time
from itertools import chain, combinations
from sympy import Matrix, pprint


class NoUnnecessaryGuessSolver(Agent):
    def __init__(self):
        self.sure_moves_not_played_yet = set()
        self.frontier_tiles = []
        self.segregated_frontier_tiles_and_fringes = []

    def nextMove(self):
        if self.game_state == Game.State.START:
            move = self.clickRandom()
        elif self.sure_moves_not_played_yet:
            move = self.sure_moves_not_played_yet.pop()
        elif sure_moves_found := self.lookForSureMovesUsingSimplerStrategies():
            move = sure_moves_found.pop()
            self.sure_moves_not_played_yet.update(sure_moves_found)
        else:
            sure_moves_found = self.lookForMovesUsingTankStrategy()

            if sure_moves_found:
                move = sure_moves_found.pop()
                self.sure_moves_not_played_yet.update(sure_moves_found)
            else:
                move = self.clickRandom()
        
        return move

    def lookForSureMovesUsingSimplerStrategies(self):
        if sure_moves_found := self.singlePointStrategy():
            return sure_moves_found
        
        return None
        
    def singlePointStrategy(self):
        sure_moves_found = set()

        for row in self.grid:
            for tile in row:
                if not tile.uncovered or tile.num_adjacent_mines == 0:
                    continue

                adjacent_tiles = self.getAdjacentTiles(tile)
                
                num_flagged = 0
                adjacent_covered_tiles = []

                for adjacent_tile in adjacent_tiles:
                    if adjacent_tile.uncovered:
                        continue

                    if adjacent_tile.is_flagged:
                        num_flagged += 1
                    else:
                        adjacent_covered_tiles.append(adjacent_tile)
                
                if not adjacent_covered_tiles:
                    continue

                adjacent_mines_not_flagged = tile.num_adjacent_mines - num_flagged

                if adjacent_mines_not_flagged == 0:
                    # Case: all adjacent covered are definitely safe
                    for adj_tile in adjacent_covered_tiles:
                        move = (adj_tile.x, adj_tile.y, False)
                        sure_moves_found.add(move)
                elif adjacent_mines_not_flagged == len(adjacent_covered_tiles):
                    # Case: all adjacent covered are definitely mines
                    for adj_tile in adjacent_covered_tiles:
                        move = (adj_tile.x, adj_tile.y, True)
                        sure_moves_found.add(move)
        
        return sure_moves_found
                
    def getAdjacentTiles(self, tile):
        max_x = len(self.grid[0]) - 1
        max_y = len(self.grid) - 1

        x, y = tile.x, tile.y
        adjacent_tiles = []

        for i in [-1, 0, 1]:
            new_x = x + i

            # Out of bounds, no tile exists there.
            if new_x < 0 or new_x > max_x:
                continue

            for j in [-1, 0, 1]:
                new_y = y + j

                # Out of bounds, no tile exists there.
                if new_y < 0 or new_y > max_y:
                    continue
                
                # We want adjacent tiles, not the tile itself
                if new_x == x and new_y == y:
                    continue

                adjacent_tile = self.grid[new_y][new_x]
                adjacent_tiles.append(adjacent_tile)

        return adjacent_tiles

    # Implementation from https://docs.python.org/3/library/itertools.html#recipes
    @staticmethod
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def comboToString(self, combo):
        return str([(tile.x, tile.y) for tile in combo])

    def lookForMovesUsingTankStrategy(self):
        sure_moves = set()
        self.segregated_frontier_tiles_and_fringes = self.getSegregatedFrontierTilesAndFringes()

        bruteforce_candidates = []

        for (frontier, fringe) in self.segregated_frontier_tiles_and_fringes:
            # Case of 1 fringe tile has already been tried by single point strategy
            if len(fringe) <= 1:
                continue

            self.cheekyHighlight([self.grid[y][x] for (x, y) in frontier], 2)
            self.cheekyHighlight([self.grid[y][x] for (x, y, _) in fringe], 4)

            # Tiles need an order as a way to reference which column of the matrix
            # represents which frontier tile. (i'th column in matrix represents i'th
            # tile in list)
            frontier = list(frontier)


            matrix = self.frontierAndFringeToMatrix(frontier, fringe)
            matrix = matrix.rref(pivots=False)            

            (sure_moves, adjusted_matrix) = self.matrixQuickerSearch(frontier, matrix)
            
            # Bruteforce later if necessary
            if not sure_moves:
                bruteforce_candidates.append((frontier, adjusted_matrix))
            
            self.removeHighlights([self.grid[y][x] for (x, y) in frontier], 2)
            self.removeHighlights([self.grid[y][x] for (x, y, _) in fringe], 4)

        while not sure_moves and bruteforce_candidates:
            (frontier, adjusted_matrix) = bruteforce_candidates.pop()
            sure_moves = self.matrixBruteForceSearch(frontier, adjusted_matrix)

        return sure_moves

    def matrixQuickerSearch(self, frontier, matrix):
        sure_moves = set()
        easy_solution_search_finished = False
        DEBUG_solutions_found = False
        while not easy_solution_search_finished:
            solutions = self.minMaxBoundarySolutionSearch(matrix)
            matrix = self.updateMatrixWithSolutions(matrix, solutions)

            if solutions:
                DEBUG_solutions_found = True
                for (frontier_index, is_mine) in solutions:
                    (x, y) = frontier[frontier_index]
                    sure_moves.add((x, y, is_mine))

                    # DEBUG
                    if is_mine:
                        self.cheekyHighlight(self.grid[y][x], 12)
                    else:
                        self.cheekyHighlight(self.grid[y][x], 11)
            else:
                easy_solution_search_finished = True

        if not DEBUG_solutions_found:
            pprint(matrix)
            print()

        #DEBUG
        self.removeHighlights([self.grid[y][x] for (x, y, is_mine) in sure_moves if is_mine], 12)
        self.removeHighlights([self.grid[y][x] for (x, y, is_mine) in sure_moves if not is_mine], 11)

        

        return sure_moves, matrix

    @staticmethod
    def minMaxBoundarySolutionSearch(matrix):
        solutions = set()

        for i in range(matrix.rows):
            min_bound = 0
            max_bound = 0
            row_start_index = i * matrix.cols

            for j in range(matrix.cols - 1):
                min_bound = min(min_bound + matrix[row_start_index + j], min_bound)
                max_bound = max(max_bound + matrix[row_start_index + j], max_bound)
            
            min_bound_solution_exists = matrix[row_start_index + matrix.cols - 1] == min_bound
            max_bound_solution_exists = matrix[row_start_index + matrix.cols - 1] == max_bound
            
            if min_bound_solution_exists or max_bound_solution_exists:
                # Extract solutions from row
                for j in range(i, matrix.cols - 1):
                    if matrix[row_start_index + j] == 1:
                        solutions.add((j, max_bound_solution_exists))
                    elif matrix[row_start_index + j] == -1:
                        solutions.add((j, min_bound_solution_exists))
        
        return solutions

    @staticmethod
    def updateMatrixWithSolutions(matrix, solutions):
        for (column, is_mine) in solutions:
            for i in range(matrix.rows):
                if is_mine and matrix[i, column] != 0:
                    matrix[i, matrix.cols - 1] -= matrix[i, column]
                
                matrix[i, column] = 0
        
        # Get rid of rows with just zero entries
        rows, cols = matrix.shape
        nonzero_rows = [i for i in range(rows) if any(matrix[i, j] != 0 for j in range(cols))]
        all_cols = list(range(cols))
        matrix = matrix[nonzero_rows, all_cols]

        return matrix

    def matrixBruteForceSearch(self, frontier, adjusted_matrix):
        # Brute force all columns from where the diagonal of 1's breaks, to the rightmost column.
        # If there is no diagonal break, then just need to bruteforce final column (2nd from right).
        # Note that limit is relative to number of rows, since rows <= cols.
        bruteforce_leftmost_col = adjusted_matrix.rows

        for i in range(adjusted_matrix.rows):
            if adjusted_matrix[i, i] == 0:
                bruteforce_leftmost_col = i
                break
        
        brute_indexes = list(range(bruteforce_leftmost_col, adjusted_matrix.cols - 1))
        combinations = self.powerset(brute_indexes)
        num_mines_over_all_configs = [0] * (adjusted_matrix.cols - 1)
        can_be_definite_move = [True] * (adjusted_matrix.cols - 1)

        for combo in combinations:
            mines_in_current_config = [0] * (adjusted_matrix.cols - 1)
            valid_config = True

            for i in range(0, bruteforce_leftmost_col):
                brute_sum = sum(adjusted_matrix[i, j] for j in combo)
                remaining_total = adjusted_matrix[i, adjusted_matrix.cols - 1] - brute_sum
                
                if remaining_total in [0, 1]:
                    mines_in_current_config[i] = remaining_total
                else:
                    valid_config = False
                    break
            
            if not valid_config:
                continue
            
            for i in range(bruteforce_leftmost_col + 1, adjusted_matrix.rows):
                brute_sum = sum(adjusted_matrix[i, j] for j in combo)
                remaining_total = adjusted_matrix[i, adjusted_matrix.cols - 1] - brute_sum
                
                if remaining_total == 0:
                    mines_in_current_config[i] = remaining_total
                else:
                    valid_config = False
                    break

            if not valid_config:
                continue

                
            
            for i in combo:
                mines_in_current_config[i] = 1
            
            for i, num_mines in enumerate(mines_in_current_config):
                num_mines_over_all_configs[i] += num_mines


        sure_moves = set()
        num_configs = 2 ** len(brute_indexes)

        for (i, num_mines) in enumerate(num_mines_over_all_configs):
            # If a frontier tile exclusively has a mine in all possible valid configuartions
            # or vice versa, then it is a definite safe move (no guessing required).
            if num_mines in [0, num_configs]:
                (x, y) = frontier[i]
                is_mine = (num_mines == num_configs)
                sure_moves.add((x, y, is_mine))
        
        return sure_moves


    def getSegregatedFrontierTilesAndFringes(self): 
        frontier_groups_and_associated_fringes = []

        # For each uncovered tile, get its adjacent frontier tiles and decide whether those
        # belong in a known segregated group, or whether to create a new group for them.
        for row in self.grid:
            for tile in row:
                if not tile.uncovered:
                    continue
                
                adjacent_tiles = self.getAdjacentTiles(tile)
            
                mines_left_around_tile = tile.num_adjacent_mines
                adjacent_frontier_tiles = []
                for adjacent_tile in adjacent_tiles:
                    if not adjacent_tile.uncovered:
                        if adjacent_tile.is_flagged:
                            mines_left_around_tile -= 1
                        else:
                            adjacent_frontier_tiles.append((adjacent_tile.x, adjacent_tile.y))

                if not adjacent_frontier_tiles:
                    continue

                # Looks for groups that shares any of the adjacent frontier tiles
                common_groups_and_fringes = []
                for adjacent_tile in adjacent_frontier_tiles:
                    for (group, fringe) in frontier_groups_and_associated_fringes:
                        if adjacent_tile in group and (group, fringe) not in common_groups_and_fringes:
                            common_groups_and_fringes.append((group, fringe))
                
                if common_groups_and_fringes:
                    # All groups that share any of the adjacent frontier tiles should be
                    # one group so merge them, and their associated fringe tiles too.
                    (some_group, some_fringe) = common_groups_and_fringes.pop()

                    # Add adjacents and fringe tile
                    some_group.update(adjacent_frontier_tiles) 
                    some_fringe.add((tile.x, tile.y, mines_left_around_tile))

                    # Merge groups and merge fringe tiles
                    for group_and_fringe in common_groups_and_fringes:
                        some_group.update(group_and_fringe[0])
                        some_fringe.update(group_and_fringe[1])
                        frontier_groups_and_associated_fringes.remove(group_and_fringe)
                else:
                    # New group discovered.
                    new_group = set(adjacent_frontier_tiles)
                    fringe = set()
                    fringe.add((tile.x, tile.y, mines_left_around_tile))
                    frontier_groups_and_associated_fringes.append((new_group, fringe))

                # self.removeHighlights(adjacent_frontier_tiles, 8)
                # self.removeHighlights([tile], 7)
                
        return frontier_groups_and_associated_fringes

    def getAdjacentCoveredNoFlagTiles(self, tile):
        adjacent_tiles = self.getAdjacentTiles(tile)
        return list(filter(lambda tile: (not tile.uncovered and not tile.is_flagged), adjacent_tiles))

    def frontierAndFringeToMatrix(self, frontier, fringe):
        matrix = []

        for (fringe_x, fringe_y, num_adjacent_unknown_mines_around_fringe_tile) in fringe:
            matrix_row = []

            for (frontier_x, frontier_y) in frontier:
                if abs(frontier_x - fringe_x) <= 1 and abs(frontier_y - fringe_y) <= 1:
                    matrix_row.append(1)
                else:
                    matrix_row.append(0)

            matrix_row.append(num_adjacent_unknown_mines_around_fringe_tile)
            matrix.append(matrix_row)

        return Matrix(matrix)        

    def clickRandom(self):
        x = randint(0, len(self.grid[0]) - 1)
        y = randint(0, len(self.grid) - 1)

        while self.isIllegalClick(x, y):
            x = randint(0, len(self.grid[0]) - 1)
            y = randint(0, len(self.grid) - 1)

        print("({}, {}) chosen at random".format(x, y))
        return (x, y, False)

    def isIllegalClick(self, x, y):
        # Out of bounds
        if x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid):
            return True

        # Tile already uncovered
        if self.grid[y][x].uncovered:
            return True

        # Can't uncover a flagged tile
        if self.grid[y][x].is_flagged:
            return True

        return False

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

        self.pruneSureMoves()
        self.segregated_frontier_tiles_and_fringes = []

    '''
        Gets rid of sure moves that have been automatically uncovered
        because of a previous move.
    '''
    def pruneSureMoves(self):
        moves_to_remove = []
        for move in self.sure_moves_not_played_yet:
            (x, y, _) = move

            if self.grid[y][x].uncovered:
                moves_to_remove.append(move)
        
        # Can't remove from set while iterating through it (throws a
        # 'RuntimeError: Set changed size during iteration').
        for move in moves_to_remove:
            self.sure_moves_not_played_yet.remove(move)

    def onGameBegin(self):
        self.sure_moves_not_played_yet = set()

    def highlightTiles(self):
        tiles_to_highlight = []
        
        # for move in self.sure_moves_not_played_yet:
        #     (x, y, toggle_flag) = move
            
        #     if toggle_flag:
        #         tile_highlight = (x, y, 3)
        #     else:
        #         tile_highlight = (x, y, 1)
                
        #     tiles_to_highlight.append(tile_highlight)

        # for tile in self.frontier_tiles:
        #     tile_highlight = (tile.x, tile.y, 2)
        #     tiles_to_highlight.append(tile_highlight)

        i = -1
        codes = list(range(1, 12))
        for (frontier, fringe) in self.segregated_frontier_tiles_and_fringes:
            i = ((i + 1) % len(codes))

            for (x, y) in frontier:
                tile_highlight = (x, y, codes[i])
                tiles_to_highlight.append(tile_highlight)
            
            for (x, y, _) in fringe:
                tile_highlight = (x, y, codes[i])
                tiles_to_highlight.append(tile_highlight)


        return tiles_to_highlight

    
    def cheekyHighlight(self, tiles, code):
        tiles_to_highlight = []
        if not isinstance(tiles, list):
            tiles = [tiles]

        for obj in tiles:
            if isinstance(obj, tuple):
                (x, y) = obj[:2]
                obj = self.grid[y][x]

            tiles_to_highlight.append((obj, code))
                
        self.renderer.highlightTilesAndDraw(tiles_to_highlight)

    def removeHighlights(self, tiles, code):
        tiles_to_unhighlight = []
        if not isinstance(tiles, list):
            tiles = [tiles]

        for obj in tiles:
            if isinstance(obj, tuple):
                (x, y) = obj[:2]
                obj = self.grid[y][x]

            tiles_to_unhighlight.append((obj, code))
        
        self.renderer.removeHighlightsAndDraw(tiles_to_unhighlight)

    def feedRenderer(self, renderer):
        self.renderer = renderer