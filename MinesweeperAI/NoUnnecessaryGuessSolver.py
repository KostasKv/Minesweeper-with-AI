from Agent import Agent
from Game import Game
from random import Random
from itertools import chain, combinations
from sympy import Matrix, pprint, ImmutableMatrix
from iteration_utilities import deepflatten
from copy import copy
from cp_solver import CpSolver


class NoUnnecessaryGuessSolver(Agent):
    def __init__(self, seed=0, use_optimised=1):
        self.use_optimised = use_optimised
        self.sure_moves_not_played_yet = set()
        self.frontier_tiles = []
        self.disjoint_frontiers_and_fringes = []
        self.SAMPLE_SIZE = (6, 6)
        self.sample_pos = None
        self.samples_considered_already = set()
        self.renderer = None
        self.cp_solver = CpSolver()
        self.seed = seed
        self.random = Random(seed)

    def nextMove(self):
        if self.game_state == Game.State.START:
            move = self.clickRandom()
        elif self.sure_moves_not_played_yet:
            move = self.sure_moves_not_played_yet.pop()
        else:
            if self.use_optimised == 1:
                sure_moves = self.method1(self.SAMPLE_SIZE)
            elif self.use_optimised == 2:
                sure_moves = self.method2(self.SAMPLE_SIZE)
            elif self.use_optimised == 3:
                sure_moves = self.method3(self.SAMPLE_SIZE)
            else:
                raise ValueError("No optimisation method provides".format(self.use_optimised))
            
            if sure_moves:
                move = sure_moves.pop()
                self.sure_moves_not_played_yet.update(sure_moves)
                # s = self.getSampleAtPosition(self.saadjacent_mines_constraintsSIZE)
                # self.cheekySampleHighlight(s, 2)
            else:
                move = self.clickRandom()
        
        if self.sure_moves_not_played_yet:
            h = self.sureMovesToHighlights(self.sure_moves_not_played_yet)
            self.cheekyGridHighlight(h)
    

        self.cheekyGridHighlight(self.sureMovesToHighlights([move]))
        self.cheekyGridHighlight(move, 6)

        return move

    def method1(self, SAMPLE_SIZE):
        sure_moves = self.lookForSureMovesFromSamplesOfGridFocussingOnFrontierTiles(SAMPLE_SIZE)

        if not sure_moves:
            sure_moves = self.lookForSureMovesFromAllPossibleSamplesOfGrid(SAMPLE_SIZE)

        return sure_moves

    def method2(self, SAMPLE_SIZE):
        return self.lookForSureMovesFromAllPossibleSamplesOfGrid(SAMPLE_SIZE)

    def method3(self, SAMPLE_SIZE):
        sure_moves = self.lookForSureMovesFromSamplesOfGridFocussingOnFrontierTiles(SAMPLE_SIZE)

        if not sure_moves:
            sure_moves = self.lookForSureMovesFromAllPossibleSamplesWhichCouldLeadToASolution(SAMPLE_SIZE)
        
        return sure_moves

    def lookForSureMovesFromAllPossibleSamplesOfGrid(self, size):
        samples = self.getSampleAreasFromGrid(size)
        sure_moves = set()
        count = 0

        for (sample, sample_pos) in samples:
            sample_hash = self.getSampleHash(sample, sample_pos)
            if sample_hash in self.samples_considered_already:
                count += 1
                continue

            self.highlightSample(sample)
            sure_moves = self.matrixAndBruteForceStrategies(sample)

            self.samples_considered_already.add(sample_hash)

            # if sure_moves:
            #     h = self.sureMovesToHighlights(sure_moves)
            #     self.cheekySampleHighlight(h)
            #     self.removeSampleHighlight(h)

            self.removeAllSampleHighlights(sample)
            if sure_moves:
                sure_moves = self.sampleMovesToGridMoves(sure_moves, sample_pos)
                sure_moves = self.pruneIllegalSureMoves(sure_moves)

                if sure_moves:
                    break
        
        return sure_moves

    def lookForSureMovesFromAllPossibleSamplesWhichCouldLeadToASolution(self, size):
        samples = self.getSampleAreasFromGridOPTIMISED(size)
        # samples = self.getSampleAreasFromGrid(size)
        sure_moves = set()
        count = 0

        for (sample, sample_pos) in samples:
            sample_hash = self.getSampleHash(sample, sample_pos)
            if sample_hash in self.samples_considered_already:
                count += 1
                continue

            self.highlightSample(sample)
            sure_moves = self.matrixAndBruteForceStrategies(sample)

            self.samples_considered_already.add(sample_hash)

            # if sure_moves:
            #     h = self.sureMovesToHighlights(sure_moves)
            #     self.cheekySampleHighlight(h)
            #     self.removeSampleHighlight(h)

            self.removeAllSampleHighlights(sample)
            if sure_moves:
                sure_moves = self.sampleMovesToGridMoves(sure_moves, sample_pos)
                sure_moves = self.pruneIllegalSureMoves(sure_moves)

                if sure_moves:
                    break
        
        return sure_moves

    def lookForSureMovesFromSamplesOfGridFocussingOnFrontierTiles(self, size):
        sure_moves = set()
        count = 0

        frontier_tile_coords = self.getAllFrontierTiles()

        for (x, y) in frontier_tile_coords:
            (sample, sample_pos) = self.getSampleCenteredOnTile(x, y, size)

            sample_hash = self.getSampleHash(sample, sample_pos)
            if sample_hash in self.samples_considered_already:
                count += 1
                continue

            self.highlightSample(sample)
            sure_moves = self.matrixAndBruteForceStrategies(sample)

            self.samples_considered_already.add(sample_hash)

            self.removeAllSampleHighlights(sample)
            if sure_moves:
                sure_moves = self.sampleMovesToGridMoves(sure_moves, sample_pos)
                sure_moves = self.pruneIllegalSureMoves(sure_moves)

                if sure_moves:
                    break

        return sure_moves

    def getSampleCenteredOnTile(self, tile_x, tile_y, sample_size):
        # Tile is either at exact center of sample, or slightly left/top of center
        x_offset = ((sample_size[0] - 1) // 2)
        y_offset = ((sample_size[1] - 1) // 2)
        x = tile_x - x_offset
        y = tile_y - y_offset

        # Bound sample to include at most 1 tile thick outside-grid wall tiles.
        max_sample_x = len(self.grid[0]) - sample_size[0] + 1
        max_sample_y = len(self.grid) - sample_size[1] + 1
        x = min(max(x, -1), max_sample_x)
        y = min(max(y, -1), max_sample_y)

        sample_pos = (x, y)
        sample = self.getSampleAtPosition(sample_pos, sample_size, self.grid)

        return (sample, sample_pos)

    @staticmethod
    def getSampleHash(sample, sample_pos):
        tiles = chain.from_iterable(sample)

        # Wall tiles      --> None
        # Uncovered tiles --> Num adjacent mines
        # Covered tiles   --> -1
        simpler_sample = tuple(None if tile is None else tile.num_adjacent_mines if tile.uncovered else -1 for tile in tiles)

        return hash((simpler_sample, sample_pos))

    def getSampleAreasFromGrid(self, size):
        # Note that these ranges will include the outside grid wall (1 tile thick at most)
        # in the samples. This is required to be sure that the solver will not make unecessary
        # guesses in a turn as knowing tiles reside next to the grid boundary is useful info
        # in certain scenarios.
        min_x = -1
        min_y = -1
        max_x = len(self.grid[0]) - size[0] + 1
        max_y = len(self.grid) - size[1] + 1

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                pos = (x, y)
                sample = self.getSampleAtPosition(pos, size, self.grid)
                yield (sample, pos)

    def getSampleAreasFromGridOPTIMISED(self, size):
        # Note that these ranges will include the outside grid wall (1 tile thick at most)
        # in the samples. This is required to be sure that the solver will not make unecessary
        # guesses in a turn as knowing tiles reside next to the grid boundary is useful info
        # in certain scenarios.
        min_x = -1
        min_y = -1
        max_x = len(self.grid[0]) - size[0] + 1
        max_y = len(self.grid) - size[1] + 1

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                pos = (x, y)

                # Only return samples that could possibly lead to uncovering a hidden tile
                if self.existsHiddenTileInOrJustOutsideSample(pos, size):
                    sample = self.getSampleAtPosition(pos, size, self.grid)
                    yield (sample, pos)

    def existsHiddenTileInOrJustOutsideSample(self, pos, size):
        for x in range(pos[0] - 1, pos[0] + size[0] + 1):
            for y in range(pos[1] - 1, pos[1] + size[1] + 1):
                # Out of bounds
                if x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid):
                    continue

                if not self.grid[y][x].uncovered and not self.grid[y][x].is_flagged:
                    return True
        
        return False

    def getSampleAtPosition(self, pos, size, grid):
        self.sample_pos = pos
        (x, y) = pos
        (columns, rows) = size

        # Calculate how many wall tiles to include (which can happen when part of sample lies
        # outside game grid).
        num_rows_of_wall_tiles_above = max(0 - y, 0)
        num_rows_of_wall_tiles_below = max(y + rows - len(grid), 0)
        num_wall_tiles_on_left_side = max((0 - x), 0)
        num_wall_tiles_on_right_side = max((x + columns - len(grid[0])), 0)

        row_of_wall_tiles = [None] * columns

        sample = [row_of_wall_tiles] * num_rows_of_wall_tiles_above
        sample_end = [row_of_wall_tiles] * num_rows_of_wall_tiles_below
        row_start = [None] * num_wall_tiles_on_left_side
        row_end = [None] * num_wall_tiles_on_right_side

        # Slices which will get tiles where sample overlaps with game grid.
        rows_slice = slice(max(y, 0), (y + rows))
        columns_slice = slice(max(x, 0), (x + columns))

        sample.extend([row_start + tile_row[columns_slice] + row_end for tile_row in grid[rows_slice]])
        sample.extend(sample_end)

        return sample

    @staticmethod
    def getAdjacentTilesInSample(tile_sample_coords, sample):
        max_x = len(sample[0]) - 1
        max_y = len(sample) - 1

        (x, y) = tile_sample_coords
        adjacents = []

        for i in [-1, 0, 1]:
            new_x = x + i
            
            if new_x < 0 or new_x > max_x:
                continue

            for j in [-1, 0, 1]:
                new_y = y + j

                if new_y < 0 or new_y > max_y:
                    continue

                # We want adjacent tiles, not the tile itself
                if new_x == x and new_y == y:
                    continue

                adjacent = sample[new_y][new_x]
                adjacents.append(adjacent)

        return adjacents

    def matrixAndBruteForceStrategies(self, sample):
        disjoint_sections = self.getDisjointSections(sample)
        all_possible_sure_moves = set()
        bruteforce_candidates = []

        for (brute, non_brute, fringe) in disjoint_sections:
            # Convert to a list so that tiles are ordered. That way we can reference
            # which matrix column refers to which tile (i'th column in matrix represents
            # i'th tile in list). Specifically placing bruteable tiles last
            # so they end up being the rightmost columns in the matrix.
            frontier = list(non_brute) + list(brute)

            matrix = self.createConstraintMatrixOfSample(frontier, fringe, self.mines_left)
            # if len(fringe) > 1:
            #     self.cheekySampleHighlight(sample, 2)
            #     self.cheekySampleHighlight(non_brute, 7)
            #     self.cheekySampleHighlight(brute, 3)
            #     self.cheekySampleHighlight(fringe, 4)
            #     pprint(matrix)
            #     print()
            # matrix = matrix.rref(pivots=False)

            # # Look for solutions that can be extracted quickly from the matrix (without
            # # resorting to bruteforcing all possible mine configurations)
            # quick_sure_moves = self.matrixQuickerSearch(matrix, frontier)

            # # solutions = self.minMaxBoundarySolutionSearch(matrix)
            # # quick_sure_moves = {(*frontier[i], is_mine) for (i, is_mine) in solutions}
                

            # # pprint(matrix)
            # # col_solutions = [frontier.index((x, y)), i for (x, y, _) in sure]
            # (adjusted_matrix, cols_deleted) = self.updateMatrixWithSolutions(matrix, frontier, quick_sure_moves)
            # # pprint(adjusted_matrix)
            # # print(frontier)
            # adjusted_frontier = [coords for (i, coords) in enumerate(frontier) if i not in cols_deleted]
            # # print(adjusted_frontier)

            # bruted_sure_moves = set()
            # # brute_start_index = self.findBruteStartIndex(adjusted_frontier, brute)
            # adjusted_matrix = matrix
            # quick_sure_moves = set()
            # adjusted_frontier = frontier
            # Exhaustive brute force search if there are bruteable tiles without solutions left.
            # if adjusted_matrix.rows > 1:
            indices_of_inside_frontier = [i for (i, (x, y)) in enumerate(frontier) if x >= 0 and y >= 0 and x < len(sample[0]) and y < len(sample)]

            bruted_sure_moves = self.matrixBruteForceSearch(matrix, frontier, indices_of_inside_frontier, self.mines_left)
            
            all_possible_sure_moves.update(bruted_sure_moves)

        return all_possible_sure_moves

    def matrixQuickerSearch(self, matrix, frontier):
        sure_moves = set()
        finished_searching = False
        tiles_removed = []

        # Don't want to affect original frontier list
        frontier = copy(frontier)

        while not finished_searching:
            solutions = self.minMaxBoundarySolutionSearch(matrix)

            if solutions:
                found_sure_moves = {(*frontier[i], is_mine) for (i, is_mine) in solutions}
                sure_moves.update(found_sure_moves)

                # pprint(matrix)
                (matrix, cols_deleted) = self.updateMatrixWithSolutions(matrix, frontier, found_sure_moves)
                
                # for i in cols_deleted:
                #     removed = frontier.pop(i)
                #     tiles_removed.append(removed)

                # Check that claimed cols deleted match up with the cols that were expected to be deleted.
                assert(all(col_to_delete in cols_deleted for (col_to_delete, _) in solutions))
                
                # Delete (and record) frontier tiles from list that
                # represented the columns that were deleted from the matrix
                for i in range(len(frontier) - 1, -1, -1):
                    if i in cols_deleted:
                        tiles_removed.append(frontier[i])
                        del frontier[i]

                # print("\nremoved {} to get:\n".format(cols_deleted))
                # pprint(matrix)
                # print("\n\n")
            else:
                finished_searching = True
        
        # if matrix.rows > 1:
        #     pprint(matrix)
        #     print()

        return sure_moves

    @staticmethod
    def minMaxBoundarySolutionSearch(matrix):
        solutions = set()

        for i in range(matrix.rows):
            negatives = []
            positives = []
            min_bound = 0
            max_bound = 0

            for j in range(matrix.cols - 1):
                if matrix[i, j] < 0:
                    negatives.append(j)
                    min_bound += matrix[i, j]
                elif matrix[i, j] > 0:
                    positives.append(j,)
                    max_bound += matrix[i, j]

            min_bound_solution_exists = (matrix[i, -1] == min_bound)
            max_bound_solution_exists = (matrix[i, -1] == max_bound)

            if min_bound_solution_exists or max_bound_solution_exists:
                for j in negatives:
                    solutions.add((j, min_bound_solution_exists))
                for j in positives:
                    solutions.add((j, max_bound_solution_exists))

        return solutions

    @staticmethod
    def updateMatrixWithSolutions(matrix, frontier, sure_moves):
        if not sure_moves:
            return matrix, []

        for (x, y, is_mine) in sure_moves:
            column = frontier.index((x, y))

            for i in range(matrix.rows):
                if is_mine and matrix[i, column] != 0:
                    matrix[i, matrix.cols - 1] -= matrix[i, column]

                matrix[i, column] = 0

        # Get rid of rows and columns with just zero entries
        rows, cols = matrix.shape
        nonzero_rows = [i for i in range(rows) if any(matrix[i, j] != 0 for j in range(cols))]

        nonzero_cols = []
        cols_deleted = []
        for j in range(cols - 1):
            if all(matrix[i, j] == 0 for i in range(rows)):
                cols_deleted.append(j)
            else:
                nonzero_cols.append(j)
        
        matrix = matrix[nonzero_rows, nonzero_cols + [cols - 1]]
        return (matrix, cols_deleted)

    @staticmethod
    def findBruteStartIndex(frontier, brute):
        ''' Inputs are an ordered collection of frontier tiles (non_brutes followed by brutes)
            and an unordered collection of brute tiles.'''
        i = None
        for (i, tile) in enumerate(reversed(frontier)):
            if tile not in brute:
                break
        
        if i is None or i == 0:
            # No brute tile found
            index = None
        else:
            index = len(frontier) - i
        
        return index

    def matrixBruteForceSearch(self, matrix, frontier, indices_of_frontier_tiles_inside_sample, mines_left_in_entire_board):
        matrix_row_constraints = [list(map(int, list(matrix[i, :]))) for i in range(matrix.rows)]
        total_mines_constraint = [1 if i in indices_of_frontier_tiles_inside_sample else 0 for i in range(len(frontier))] + [mines_left_in_entire_board]
        definite_solutions = self.cp_solver.searchForDefiniteSolutions(matrix_row_constraints, total_mines_constraint)

        sure_moves = set()

        for (index, is_mine) in definite_solutions:
            coords = frontier[index]
            sure_moves.add((*coords, is_mine,))

        return sure_moves

    def getAllFrontierTiles(self):
        frontier_tile_coords = set()

        for (y, row) in enumerate(self.grid):
            for (x, tile) in enumerate(row):
                if tile.uncovered or tile.is_flagged:
                    continue

                if self.isCoveredTileAFrontierTile(x, y):
                    frontier_tile_coords.add((x, y))

        return frontier_tile_coords

    def isCoveredTileAFrontierTile(self, tile_x, tile_y):
        '''Assumes input is a non-flagged covered tile '''

        adjacent_coords = [((tile_x + i), (tile_y + j)) for i in (-1, 0, 1) for j in (-1, 0, 1)]

        for (x, y) in adjacent_coords:
            if x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid):
                continue

            if self.grid[y][x].uncovered:
                return True
        
        return False

    def getDisjointSections(self, sample):
        disjoint_sections = []

        for (y, row) in enumerate(sample):
            for (x, tile) in enumerate(row):
                # Only consider uncovered tiles with a number larger than 0, as those are the only
                # ones from which useful constraints can be made (and whether or not those constraints are disjoint
                # determines whether or not the sections are disjoint).
                if not tile or not tile.uncovered or tile.num_adjacent_mines == 0:
                    continue
                
                disjoint_sections = self.updateSampleDisjointSectionsBasedOnUncoveredTile(sample, disjoint_sections, (x, y), tile.num_adjacent_mines)

        return disjoint_sections

    def updateSampleDisjointSectionsBasedOnUncoveredTile(self, sample, disjoint_sections, tile_coords, num_adjacent_mines):
        adjacent_section = self.getAdjacentSectionForSampleTile(sample, tile_coords, num_adjacent_mines)
        (brute, non_brute) = adjacent_section[:2]
           
        # Can only merge or find new sections based on frontier tiles in the adjacent section
        if brute or non_brute:
            disjoint_sections = self.updateDisjointSectionBasedOnAdjacentSection(disjoint_sections, adjacent_section)

        return disjoint_sections

    def getAdjacentSectionForSampleTile(self, sample, tile_coords, num_adjacent_mines):
        adjacent_coords = [(tile_coords[0] + i, tile_coords[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
        mines_left_around_tile = num_adjacent_mines
        non_brute = set()
        brute = set()

        while adjacent_coords:
            (x, y) = adjacent_coords.pop()
            
            # If tile is outside sample, then you can't brute it.
            if x < 0 or y < 0 or x >= len(sample[0]) or y >= len(sample):
                non_brute.add((x, y))
                continue
            
            adjacent = sample[y][x]

            # If we know, or can figure out, that a tile is a wall tile then
            # exclude it from the section (even if that tile is outside the sample,
            # since we don't want to add constraints that shouldn't be there).
            if adjacent is None:
                # If dim==0, wall is a row at (wall_dim_location, y), for any y.
                # If dim==1, wall is a colum at (x, wall_dim_location), for any x.
                (wall_dim_location, dim) = self.getWallLocation(tile_coords, (x, y))

                # If it's known where the entire wall is, remove all coordinates belonging to that wall.
                if wall_dim_location is not None:
                    non_brute = set(coord for coord in non_brute if coord[dim] != wall_dim_location)
                    adjacent_coords = [coord for coord in adjacent_coords if coord[dim] != wall_dim_location]

                continue

            if adjacent.uncovered:
                continue

            if adjacent.is_flagged:
                mines_left_around_tile -= 1
            elif self.isBruteforceableSampleTile(sample, (x, y)):
                brute.add((x, y))
            else:
                non_brute.add((x, y))

        fringe = {(*tile_coords, mines_left_around_tile)}

        return (brute, non_brute, fringe)

    def updateDisjointSectionBasedOnAdjacentSection(self, disjoint_sections, adjacent_section):
        (brute, non_brute, fringe) = adjacent_section

        updated_disjoint_sections = []
        sections_to_merge = []

        # 'Disjoint' sections that share any of the frontier tiles are not really
        # disjoint; they should be merged.
        for section in disjoint_sections:
            section_frontier = set.union(*section[:2])
            section_is_disjoint_from_adjacent = (brute.isdisjoint(section_frontier) and non_brute.isdisjoint(section_frontier))

            if section_is_disjoint_from_adjacent:
                updated_disjoint_sections.append(section)
            else:
                sections_to_merge.append(section)

        if sections_to_merge:
            sections_to_merge.append(adjacent_section)
            section = self.mergeSections(sections_to_merge)
        else:
            # New disjoint section discovered
            section = adjacent_section
        
        updated_disjoint_sections.append(section)

        return updated_disjoint_sections

    @staticmethod
    def mergeSections(sections_to_merge):
        (all_brute_sets, all_non_brute_sets, all_fringe_sets) = zip(*sections_to_merge)
        return (set.union(*all_brute_sets), set.union(*all_non_brute_sets), set.union(*all_fringe_sets))

    @staticmethod
    def getWallLocation(tile_pos, adjacent_tile_pos):
        (tile_x, tile_y) = tile_pos
        (adj_x, adj_y) = adjacent_tile_pos

        is_x_off_center = abs(tile_x - adj_x)
        is_y_off_center = abs(tile_y - adj_y)
        
        if is_x_off_center and is_y_off_center:
            # Corner adjacents on their own aren't enough information
            # to know whether it's a row or column of wall tiles
            wall_dim_position = None
            dim = None
        elif is_x_off_center:
            # Wall is a column at adj_x
            wall_dim_position = adj_x
            dim = 0
        elif is_y_off_center:
            # Wall is a row at adj_y
            wall_dim_position = adj_y
            dim = 1
        else:
            raise ArithmeticError("Something went wrong in the wall-tile deduction. Did you accidentally include the tile itself as its own adjacent? Or maybe the coordinates aren't correct.")
        
        return (wall_dim_position, dim)

    '''
        Assumes input tile is a covered sample tile.

        Tile is bruteforceable iff. none of its adjacent tiles are an unknown outside tile.
        Only covered tiles in the inner region of a sample are bruteforceable. Covered tiles
        on the border of the sample will always have an adjacent unknown tile.
    '''
    @staticmethod
    def isBruteforceableSampleTile(sample, tile_pos):
        (x, y) = tile_pos
        return x >= 1 and y >= 1 and x <= (len(sample[0]) - 2) and y <= (len(sample) - 2)

    @staticmethod
    def createConstraintMatrixOfSample(frontier, fringe, mines_left):
        matrix = []

        # Build up matrix of row equations.
        for (fringe_x, fringe_y, num_unflagged_adjacent_mines_around_tile) in fringe:
            matrix_row = []

            # Build equation's left-hand-side of variables
            for (frontier_x, frontier_y) in frontier:
                # If frontier tile is adjacent to fringe tile, then it has an effect
                # on the fringe tile's adjacent mine constraint. Include it in the equation
                # by giving it a coefficient of 1, otherwise exclude it with a
                # coefficient of 0.
                if abs(frontier_x - fringe_x) <= 1 and abs(frontier_y - fringe_y) <= 1:
                    matrix_row.append(1)
                else:
                    matrix_row.append(0)

            # Append equation's right-hand-side answer/constraint
            matrix_row.append(num_unflagged_adjacent_mines_around_tile)

            matrix.append(matrix_row)

        return Matrix(matrix)

    @staticmethod
    def sampleMovesToGridMoves(sample_moves, sample_pos):
        (x, y) = sample_pos
        return list(map(lambda move: ((move[0] + x), (move[1] + y), move[2]), sample_moves))

    def clickRandom(self):
        x = self.random.randint(0, len(self.grid[0]) - 1)
        y = self.random.randint(0, len(self.grid) - 1)

        while self.isIllegalClick(x, y):
            x = self.random.randint(0, len(self.grid[0]) - 1)
            y = self.random.randint(0, len(self.grid) - 1)

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
        # Removes coordinates from every tile. A sample's coordinates will be inferred
        # from the element's position in the sample.
        converted_grid = [[SampleTile(tile) for tile in row] for row in grid]
        self.grid = converted_grid

        self.mines_left = mines_left
        self.game_state = game_state

        # Last played move could have uncovered multiple tiles, making some sure moves now illegal moves.
        self.sure_moves_not_played_yet = self.pruneIllegalSureMoves(self.sure_moves_not_played_yet)

    def pruneIllegalSureMoves(self, sure_moves):
        ''' Gets rid of sure moves that cannot actually be played. '''
        valid_sure_moves = set()

        for move in sure_moves:
            (x, y, _) = move

            tile_is_in_grid = (x >= 0 and y >= 0 and x < len(self.grid[0]) and y < len(self.grid))

            if tile_is_in_grid and not self.grid[y][x].uncovered and not self.grid[y][x].is_flagged:
                valid_sure_moves.add(move)

        return valid_sure_moves

    def onGameBegin(self):
        self.random = Random(self.seed)
        self.sure_moves_not_played_yet = set()
        self.samples_considered_already = set()

    @staticmethod
    def disjointSectionsToHighlights(sections):
        ''' 
            Each disjoint section is given a certain highlight code. This highlight
            cycles through a number of consecutive numbered highlights, stepping through
            each highlight in the cycle once per disjoint section.
        '''
        START_HIGHLIGHT_NUM = 7
        END_HIGHLIGHT_NUM = 12
        
        highlights = []

        code_i = START_HIGHLIGHT_NUM

        for (brute, non_brute, fringe) in sections:
            all_section_tiles = brute.union(non_brute.union(fringe))
            code = code_i + 1
            highlights.extend((tile, code) for tile in all_section_tiles)
            code_i = (code_i + 1) % END_HIGHLIGHT_NUM
        
        return highlights

    @staticmethod
    def sureMovesToHighlights(sure_moves):
        FLAG = 12
        SAFE = 11
        return [((x, y), FLAG) if is_mine else ((x, y), SAFE) for (x, y, is_mine) in sure_moves]

    def highlightSample(self, sample):
        HIGHLIGHT_CODE = 2
        
        # Convert to (sample_coords, code). Ignores None tiles (which are wall tiles that are out of bounds).
        h = [((x, y), HIGHLIGHT_CODE) for (y, row) in enumerate(sample) for (x, tile) in enumerate(row) if tile is not None]

        self.cheekyHighlight(h)

    def cheekyGridHighlight(self, *args, transform=None):
        self.handleHighlights(*args, add_highlights=True, transform=transform)

    def removeGridHighlight(self, *args, transform=None):
        self.handleHighlights(*args, add_highlights=False, transform=transform)

    def cheekyHighlight(self, *args):
        self.handleHighlights(*args, add_highlights=True, transform=self.sample_pos)

    def removeHighlight(self, *args):
        self.handleHighlights(*args, add_highlights=False, transform=self.sample_pos)

    def removeAllSampleHighlights(self, sample):
        if not self.renderer:
            return
        elif not self.sample_pos:
            raise ValueError("self.sample_pos has not been assigned a position.")
    
        tile_coords = [(x, y) for (y, row) in enumerate(sample) for (x, tile) in enumerate(row) if tile is not None]
        tile_coords = self.transformCoords(tile_coords, transform=self.sample_pos)

        self.renderer.removeAllTileHighlights(tile_coords)

    def handleHighlights(self, *args, add_highlights=None, transform=None):
        if not self.renderer:
            return
        
        tile_coords_with_code = self.prepHighlights(*args, transform=transform)
        
        if not tile_coords_with_code:
            return
        
        self.renderer.highlightTilesAndDraw(tile_coords_with_code, add_highlights=add_highlights)

    def prepHighlights(self, *args, transform=None):
        if len(args) not in [1, 2]:
            raise TypeError("Expected 1 or 2 positional arguments. Received {}.".format(len(args)))

        (tile_coords, codes) = self.getTilesAndCodesSeperate(*args)

        if transform:
            tile_coords = self.transformCoords(tile_coords, transform=transform)

            # Remove coordinates that end up out of bounds after transformation
            for (i, (x, y)) in enumerate(tile_coords[::-1]):
                inside_grid_bounds = (x >= 0 and y >= 0 and x < len(self.grid[0]) and y < len(self.grid))

                if not inside_grid_bounds:
                    index = len(tile_coords) - 1 - i
                    tile_coords.pop(index)
                    codes.pop(index)
    
        return list(zip(tile_coords, codes))

    def getTilesAndCodesSeperate(self, *args):
        if not args[0]:
            return ([], [])

        args = list(args)

        # Make sure tiles argument is put inside a list
        try:
            # A tuple is assumed to contain coordinates
            if isinstance(args[0], tuple):
                raise TypeError

            # Check if its iterable
            iter(args[0])
        except TypeError:
            # Not an iterable (or is tuple). So place it inside a list.
            args[0] = [args[0]]

        if len(args) == 2:
            (tile_coords, code) = args
            code = str(code)

            # Flatten tiles down to 1D.
            tile_coords = list(deepflatten(tile_coords, types=(list, set)))
            codes = [code] * len(tile_coords)
        else:
            tiles_with_codes = args[0]

            try:
                # NOTE: this ain't the best check. This will fail to catch out anything erroneous
                # that happens to have its first element as length 2, e.g. [[tile_1, tile_2]]
                if len(tiles_with_codes[0]) != 2:
                    raise TypeError
            except:
                raise TypeError("Expected {} to be of form (Tile_like, highlight_code). Did you forget to pass the highlight code as a second parameter?".format(tiles_with_codes[0]))

            tile_coords, codes = zip(*tiles_with_codes)
            codes = list(map(str, codes))

        # Truncate tuples (x, y, z, ...,) down to (x, y), which should be the tile coordinates.
        tile_coords = list(map(lambda x: x[:2], tile_coords))

        for (x, y) in tile_coords:
            x = int(x)
            y = int(y)

        return (tile_coords, codes)

    def transformCoords(self, coords, transform):
        return list(map(lambda c: (c[0] + transform[0], c[1] + transform[1]), coords))

    def feedRenderer(self, renderer):
        self.renderer = renderer


class SampleTile():
    def __init__(self, tile):
        self.uncovered = tile.uncovered
        self.is_flagged = tile.is_flagged
        self.num_adjacent_mines = tile.num_adjacent_mines

class SampleTile2():
    def __init__(self, tile, coords):
        self.x = coords[0]
        self.y = coords[1]
        self.uncovered = tile.uncovered
        self.is_flagged = tile.is_flagged
        self.num_adjacent_mines = tile.num_adjacent_mines