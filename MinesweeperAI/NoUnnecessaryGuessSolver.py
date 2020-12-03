from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from Agent import Agent
from Game import Game
from random import randint
from itertools import chain, combinations
from sympy import Matrix, pprint, ImmutableMatrix
from iteration_utilities import deepflatten
from copy import copy
# import cp_solver

from ortools.sat.python import cp_model


class NoUnnecessaryGuessSolver(Agent):
    def __init__(self):
        self.sure_moves_not_played_yet = set()
        self.frontier_tiles = []
        self.disjoint_frontiers_and_fringes = []
        self.SAMPLE_SIZE = (5, 5)
        self.sample_pos = None
        self.samples_considered_already = set()
        self.renderer = None

    def nextMove(self):
        if self.game_state == Game.State.START:
            move = self.clickRandom()
        elif self.sure_moves_not_played_yet:
            move = self.sure_moves_not_played_yet.pop()
        else:
            sure_moves = self.lookForSureMovesFromSamplesOfGrid(self.SAMPLE_SIZE)

            if sure_moves:
                move = sure_moves.pop()
                self.sure_moves_not_played_yet.update(sure_moves)
                # s = self.getSampleAtPosition(self.sample_pos, self.SAMPLE_SIZE)
                # self.cheekySampleHighlight(s, 2)
            else:
                move = self.clickRandom()
        
        if self.sure_moves_not_played_yet:
            h = self.sureMovesToHighlights(self.sure_moves_not_played_yet)
            self.cheekyHighlight(h)
    

        self.cheekyHighlight(self.sureMovesToHighlights([move]))
        self.cheekyHighlight(move, 6)

        return move

    def lookForSureMovesFromSamplesOfGrid(self, size):
        max_op = 4
        ops = [None] + [i for i in range(1, max_op + 1)]

        # # TRY ALL METHODS
        # all_samples = [self.getSampleAreasFromGrid(size, optimisation=op) for op in ops]
        # all_samples.append(self.getSampleAreasFromGridOPTIMISED(size))

        # COMPARE JUST getSampleAreasFromGrid optimisations
        all_samples = [
            self.getSampleAreasFromGrid(size, optimisation=None),
            self.getSampleAreasFromGridOPTIMISED(size),
            self.getSampleAreasFromGridOPTIMISED2(size),
            self.getSampleAreasFromGridOPTIMISED3(size),
            ]

        for samples in all_samples:
            for (sample, sample_pos) in samples:
                pass

        count = 0
        sure_moves = set()
            # sample_hash = self.getSampleHash(sample)
            # if sample_hash in self.samples_considered_already:
            #     count += 1
            #     continue

            # self.cheekySampleHighlight(sample, 2)
            # sure_moves = self.matrixAndBruteForceStrategies(sample)

            # self.samples_considered_already.add(sample_hash)

            # # if sure_moves:
            # #     h = self.sureMovesToHighlights(sure_moves)
            # #     self.cheekySampleHighlight(h)
            # #     self.removeSampleHighlight(h)

            # self.removeAllSampleHighlights(sample)
            # if sure_moves:
            #     sure_moves = self.sampleMovesToGridMoves(sure_moves, sample_pos)
            #     sure_moves = self.pruneSureMoves(sure_moves)

            #     if sure_moves:
            #         break
        
        return sure_moves

    @staticmethod
    def getSampleHash(sample):
        tiles = chain.from_iterable(sample)
        simpler_sample = tuple(None if tile is None else tile.num_adjacent_mines if tile.uncovered else -1 for tile in tiles)
        return hash(simpler_sample)

    '''
        Note that (shallow) copies of the grid's Tile objects are
        used. This is so that a change in a sample's tile doesn't accidentally make a
        a change to the grid itself. This way the solving algorithms can
        'mark' a sample with solutions while keeping unique samples independant from eachother.
    '''

    def getSampleAreasFromGrid(self, size, optimisation=None):
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

                if optimisation is None:
                    sample = self.getSampleAtPosition(pos, size)
                elif optimisation == 1:
                    sample = self.getSampleAtPositionOPTIMISED(pos, size)
                elif optimisation == 2:
                    sample = self.getSampleAtPositionOPTIMISED2(pos, size)
                elif optimisation == 3:
                    sample = self.getSampleAtPositionOPTIMISED3(pos, size)
                elif optimisation == 4:
                    sample = self.getSampleAtPositionOPTIMISED4(pos, size)
                else:
                    raise ValueError("No optimisation for {} given.".format(optimisation))
                
                self.sample_pos = pos
                yield (sample, pos)

    '''
        Each tile in sample has its relative sample coordinates, rather
        than its real grid coordinates.
    '''
    def getSampleAtPosition(self, pos, size):
        (x, y) = pos
        (columns, rows) = size

        # Calculate how many wall tiles to include (which can happen when part of sample lies
        # outside game grid).
        num_rows_of_wall_tiles_above = max(0 - y, 0)
        num_rows_of_wall_tiles_below = max(y + rows - len(self.grid), 0)
        num_wall_tiles_on_left_side = max((0 - x), 0)
        num_wall_tiles_on_right_side = max((x + columns - len(self.grid[0])), 0)

        row_of_wall_tiles = [None] * columns

        sample = [row_of_wall_tiles] * num_rows_of_wall_tiles_above
        sample_end = [row_of_wall_tiles] * num_rows_of_wall_tiles_below
        row_start = [None] * num_wall_tiles_on_left_side
        row_end = [None] * num_wall_tiles_on_right_side

        # Slices which will get tiles where sample overlaps with game grid.
        rows_slice = slice(max(y, 0), (y + rows))
        columns_slice = slice(max(x, 0), (x + columns))

        for sample_y, tile_row in enumerate(self.grid[rows_slice], num_rows_of_wall_tiles_above):
            # Don't want a reference to row_start since row will be modified. Just want it's values.
            row = copy(row_start)

            for sample_x, tile in enumerate(tile_row[columns_slice], num_wall_tiles_on_left_side):
                copied_tile = copy(tile)
                copied_tile.x = sample_x
                copied_tile.y = sample_y

                row.append(copied_tile)

            row.extend(row_end)
            sample.append(row)

        sample.extend(sample_end)

        return sample

    def getSampleAreasFromGridOPTIMISED(self, size):
        converted_grid = []
        for row in self.grid:
            row = ['@' if tile is None else str(tile.num_adjacent_mines) if tile.uncovered else '-' for tile in row]
            converted_grid.append(row)

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
                sample = self.getSampleAtPositionOPTIMISED5(pos, size, converted_grid)
                self.sample_pos = pos
                yield (sample, pos)

    def getSampleAreasFromGridOPTIMISED2(self, size):
        converted_grid = [[SampleTile(tile) for tile in row] for row in self.grid]

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
                sample = self.getSampleAtPositionOPTIMISED5(pos, size, converted_grid)
                self.sample_pos = pos
                yield (sample, pos)

    def getSampleAreasFromGridOPTIMISED3(self, size):
        converted_grid = [[copy(tile) for tile in row] for row in self.grid]

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
                sample = self.getSampleAtPositionOPTIMISED5(pos, size, converted_grid)
                self.sample_pos = pos
                yield (sample, pos)

    def getSampleAtPositionOPTIMISED(self, pos, size):
        (x, y) = pos
        (columns, rows) = size

        # Calculate how many wall tiles to include (which can happen when part of sample lies
        # outside game grid).
        num_rows_of_wall_tiles_above = max(0 - y, 0)
        num_rows_of_wall_tiles_below = max(y + rows - len(self.grid), 0)
        num_wall_tiles_on_left_side = max((0 - x), 0)
        num_wall_tiles_on_right_side = max((x + columns - len(self.grid[0])), 0)

        row_of_wall_tiles = [None] * columns

        sample = [row_of_wall_tiles] * num_rows_of_wall_tiles_above
        sample_end = [row_of_wall_tiles] * num_rows_of_wall_tiles_below
        row_start = [None] * num_wall_tiles_on_left_side
        row_end = [None] * num_wall_tiles_on_right_side

        # Slices which will get tiles where sample overlaps with game grid.
        rows_slice = slice(max(y, 0), (y + rows))
        columns_slice = slice(max(x, 0), (x + columns))

        for sample_y, tile_row in enumerate(self.grid[rows_slice], num_rows_of_wall_tiles_above):
            # Don't want a reference to row_start since row will be modified. Just want it's values.
            row = [] + row_start

            for sample_x, tile in enumerate(tile_row[columns_slice], num_wall_tiles_on_left_side):
                copied_tile = SampleTile((sample_x, sample_y), tile)
                row.append(copied_tile)

            row.extend(row_end)
            sample.append(row)

        sample.extend(sample_end)

        return sample

    def getSampleAtPositionOPTIMISED2(self, pos, size):
        (x, y) = pos
        (columns, rows) = size

        # Calculate how many wall tiles to include (which can happen when part of sample lies
        # outside game grid).
        num_rows_of_wall_tiles_above = max(0 - y, 0)
        num_rows_of_wall_tiles_below = max(y + rows - len(self.grid), 0)
        num_wall_tiles_on_left_side = max((0 - x), 0)
        num_wall_tiles_on_right_side = max((x + columns - len(self.grid[0])), 0)

        row_of_wall_tiles = [None] * columns

        sample = [row_of_wall_tiles] * num_rows_of_wall_tiles_above
        sample_end = [row_of_wall_tiles] * num_rows_of_wall_tiles_below
        row_start = [None] * num_wall_tiles_on_left_side
        row_end = [None] * num_wall_tiles_on_right_side

        # Slices which will get tiles where sample overlaps with game grid.
        rows_slice = slice(max(y, 0), (y + rows))
        columns_slice = slice(max(x, 0), (x + columns))

        for sample_y, tile_row in enumerate(self.grid[rows_slice], num_rows_of_wall_tiles_above):
            # Don't want a reference to row_start since row will be modified. Just want it's values.
            row = [] + row_start

            for sample_x, tile in enumerate(tile_row[columns_slice], num_wall_tiles_on_left_side):
                if tile is None:
                    value = '@'
                elif tile.uncovered:
                    value = str(tile.num_adjacent_mines)
                else:
                    value = '-'
                    
                copied_tile = (value, (sample_x, sample_y))
                row.append(copied_tile)

            row.extend(row_end)
            sample.append(row)

        sample.extend(sample_end)

        return sample

    def getSampleAtPositionOPTIMISED3(self, pos, size):
        (x, y) = pos
        (columns, rows) = size

        # Calculate how many wall tiles to include (which can happen when part of sample lies
        # outside game grid).
        num_rows_of_wall_tiles_above = max(0 - y, 0)
        num_rows_of_wall_tiles_below = max(y + rows - len(self.grid), 0)
        num_wall_tiles_on_left_side = max((0 - x), 0)
        num_wall_tiles_on_right_side = max((x + columns - len(self.grid[0])), 0)

        row_of_wall_tiles = [None] * columns

        sample = [row_of_wall_tiles] * num_rows_of_wall_tiles_above
        sample_end = [row_of_wall_tiles] * num_rows_of_wall_tiles_below
        row_start = [None] * num_wall_tiles_on_left_side
        row_end = [None] * num_wall_tiles_on_right_side

        # Slices which will get tiles where sample overlaps with game grid.
        rows_slice = slice(max(y, 0), (y + rows))
        columns_slice = slice(max(x, 0), (x + columns))

        for tile_row in self.grid[rows_slice]:
            # Don't want a reference to row_start since row will be modified. Just want it's values.
            row = [] + row_start

            for tile in tile_row[columns_slice]:
                if tile is None:
                    value = '@'
                elif tile.uncovered:
                    value = str(tile.num_adjacent_mines)
                else:
                    value = '-'

                row.append(value)

            row.extend(row_end)
            sample.append(row)

        sample.extend(sample_end)

        return sample

    def getSampleAtPositionOPTIMISED4(self, pos, size):
        (x, y) = pos
        (columns, rows) = size

        # Calculate how many wall tiles to include (which can happen when part of sample lies
        # outside game grid).
        num_rows_of_wall_tiles_above = max(0 - y, 0)
        num_rows_of_wall_tiles_below = max(y + rows - len(self.grid), 0)
        num_wall_tiles_on_left_side = max((0 - x), 0)
        num_wall_tiles_on_right_side = max((x + columns - len(self.grid[0])), 0)

        row_of_wall_tiles = [None] * columns

        sample = [row_of_wall_tiles] * num_rows_of_wall_tiles_above
        sample_end = [row_of_wall_tiles] * num_rows_of_wall_tiles_below
        row_start = [None] * num_wall_tiles_on_left_side
        row_end = [None] * num_wall_tiles_on_right_side

        # Slices which will get tiles where sample overlaps with game grid.
        rows_slice = slice(max(y, 0), (y + rows))
        columns_slice = slice(max(x, 0), (x + columns))

        for tile_row in self.grid[rows_slice]:
            # Concatenating to empty list because we don't want a reference to row_start since row will be modified.
            row = [] + row_start
            row.extend('@' if tile is None else str(tile.num_adjacent_mines) if tile.uncovered else '-' for tile in tile_row[columns_slice])
            sample.append(row)

        sample.extend(sample_end)

        return sample

    def getSampleAtPositionOPTIMISED5(self, pos, size, grid):
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

        sample.extend([row_start + tile_row[columns_slice] for tile_row in grid[rows_slice]])
        sample.extend(sample_end)

        return sample


    # def singlePointStrategy(self, sample):
    #     all_sure_moves_found = set()
    #     tiles_and_adjacents_of_interest = []

    #     for (sample_y, row) in enumerate(sample[1 : -1]):
    #         for (sample_x, tile) in enumerate(row[1 : -1]):
    #             # Skip tiles that can't be used to determine if neighbouring
    #             # tiles are/aren't mines using SPS.
    #             if not tile or not tile.uncovered:
    #                 continue

    #             adjacent_tiles = self.getAdjacentTilesInSample((sample_x, sample_y), sample)
    #             self.cheekyHighlight(adjacent_tiles)
    #             sure_moves = self.singlePointStrategyOnTileAndAdjacents(tile, adjacent_tiles)

    #             if sure_moves:
    #                 all_sure_moves_found.update(sure_moves)
    #             else:
    #                 # Incase SPS needs to be repeated multiple times for this sample.
    #                 # Only uncovered inside tiles which haven't had sure moves yet
    #                 # could possible have sure moves later (after nearby sure moves are found first)
    #                 tiles_and_adjacents_of_interest.append((tile, adjacent_tiles))
    #             self.removeHighlight(adjacent_tiles)

    #     moves_found = (len(all_sure_moves_found) > 0)

    #     # Sure moves found after an iteration can lead to the discovery of new sure moves in the same sample.
    #     # Therefore, SPS should be repeated a bunch until it can no longer find any more sure moves in the sample.
    #     while moves_found:
    #         moves_found = False

    #         for (tile, adjacent_tiles) in tiles_and_adjacents_of_interest:
    #             sure_moves = self.singlePointStrategyOnTileAndAdjacents(tile, adjacent_tiles)

    #             if sure_moves:
    #                 moves_found = True
    #                 all_sure_moves_found.update(sure_moves)

    #                 # Once sure moves are found around a tile, it can't give us any more sure moves.
    #                 tiles_and_adjacents_of_interest.remove((tile, adjacent_tiles))

    #     return all_sure_moves_found

    # '''
    #     Side effect: for every solution it finds, this method marks the tile with
    #     that solution. This means the sample can be affected.
    # '''
    # def singlePointStrategyOnTileAndAdjacents(self, tile, adjacent_tiles):
    #     sure_moves = set()
    #     num_flagged = 0
    #     adjacent_covered_tiles = []

    #     for adjacent in adjacent_tiles:
    #         if adjacent.uncovered:
    #             continue

    #         if adjacent.is_flagged:
    #             num_flagged += 1
    #         else:
    #             adjacent_covered_tiles.append(adjacent)

    #     if adjacent_covered_tiles:
    #         self.cheekyHighlight(tile, 4)
    #         self.cheekyHighlight(adjacent_covered_tiles, 1)

    #         adjacent_mines_not_flagged = tile.num_adjacent_mines - num_flagged

    #         if adjacent_mines_not_flagged == 0:
    #             sure_moves = self.formIntoSureMovesAndUpdateTilesWithSolution(adjacent_covered_tiles, is_mine=True)
    #         elif adjacent_mines_not_flagged == len(adjacent_covered_tiles):
    #             sure_moves = self.formIntoSureMovesAndUpdateTilesWithSolution(adjacent_covered_tiles, is_mine=True)

    #         self.removeHighlight(tile, 4)
    #         self.removeHighlight(adjacent_covered_tiles, 1)

    #     # # DEBUG
    #     # for (x, y, is_mine) in sure_moves_found:
    #     #     if is_mine:
    #     #         code = 12
    #     #     else:
    #     #         code = 11

    #     #     self.removeHighlight((x, y), code)

    #     return sure_moves

    # def formIntoSureMovesAndUpdateTilesWithSolution(self, adjacent_covered_tiles, is_mine=True):
    #     sure_moves = set()

    #     for tile in adjacent_covered_tiles:
    #         move = (tile.x, tile.y, is_mine)
    #         sure_moves.add(move)

    #         # Mark solution on sample's tile itself
    #         if isinstance(tile, SampleOutsideTile):
    #             tile.setIsMine(is_mine)
    #         else:
    #             tile.is_flagged = is_mine

    #         # DEBUG
    #         if is_mine:
    #             code = 12
    #         else:
    #             code = 11
    #         self.cheekyHighlight(tile, code)

    #     return sure_moves

    # @staticmethod
    # def getAdjacentTilesInSample(tile_sample_coords, sample, include_outside=False):
    #     max_x = len(sample[0]) - 1
    #     max_y = len(sample) - 1

    #     (x, y) = tile_sample_coords
    #     adjacents = []
    #     is_outside = False

    #     for i in [-1, 0, 1]:
    #         new_x = x + i
            
    #         if new_x < 0 or new_x > max_x:
    #             if include_outside:
    #                 is_outside = True
    #             else:
    #                 continue

    #         for j in [-1, 0, 1]:
    #             new_y = y + j

    #             if new_y < 0 or new_y > max_y:
    #                 if include_outside:
    #                     is_outside = True
    #                 else:
    #                     continue

    #             # We want adjacent tiles, not the tile itself
    #             if new_x == x and new_y == y:
    #                 continue
                
    #             if is_outside:
    #                 adjacent = (new_x, new_y)
    #             else:
    #                 adjacent = sample[new_y][new_x]

    #             adjacents.append(adjacent)

    #     return adjacents

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

    # @staticmethod
    # def updateSampleWithSureMoves(sample, sure_moves_found):
    #     for (x, y, is_mine) in sure_moves_found:
    #         sample[y][x].setIsMine(is_mine)

    #     return sample

    # # Implementation from
    # # https://docs.python.org/3/library/itertools.html#recipes
    # @staticmethod
    # def powerset(iterable):
    #     "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    #     s = list(iterable)
    #     return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

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
            # (adjusted_matrix, adjusted_frontier, quick_sure_moves) = self.matrixQuickerSearch(matrix, frontier)

            bruted_sure_moves = set()
            # # brute_start_index = self.findBruteStartIndex(adjusted_frontier, brute)
            adjusted_matrix = matrix
            quick_sure_moves = set()
            adjusted_frontier = frontier
            # Exhaustive brute force search if there are bruteable tiles without solutions left.
            if adjusted_matrix.rows > 1:
                bruted_sure_moves = self.matrixBruteForceSearch(adjusted_matrix, adjusted_frontier)
            
            all_possible_sure_moves.update(quick_sure_moves | bruted_sure_moves)

        return all_possible_sure_moves

    def matrixQuickerSearch(self, matrix, frontier):
        sure_moves = set()
        finished_searching = False
        tiles_removed = []

        while not finished_searching:
            solutions = self.minMaxBoundarySolutionSearch(matrix)

            if solutions:
                for (frontier_index, is_mine) in solutions:
                    (x, y) = frontier[frontier_index]
                    sure_moves.add((x, y, is_mine))

                # pprint(matrix)
                (matrix, cols_deleted) = self.updateMatrixWithSolutions(matrix, solutions)
                
                # for i in cols_deleted:
                #     removed = frontier.pop(i)
                #     tiles_removed.append(removed)

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

        return (matrix, frontier, sure_moves)

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
    def updateMatrixWithSolutions(matrix, solutions):
        for (column, is_mine) in solutions:
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

    def matrixBruteForceSearch(self, matrix, frontier):
        matrix_row_constraints = [list(map(int, list(matrix[i, :]))) for i in range(matrix.rows)]
        definite_solutions = self.searchForDefiniteSolutionsUsingCpSolver(matrix_row_constraints)

        sure_moves = set()

        for (index, is_mine) in definite_solutions:
            coords = frontier[index]
            sure_moves.add((*coords, is_mine,))

        return sure_moves


    # def matrixBruteForceSearch(self, matrix, frontier, brute_start_index):
    #     brute_indexes = range(brute_start_index, matrix.cols - 1)
    #     combinations = self.powerset(brute_indexes)

    #     num_mines_over_all_configs = [0] * (matrix.cols - 1)
    #     # can_be_definite_move = [True] * (adjusted_matrix.cols - 1)

    #     for combo in combinations:
    #         mines_in_current_config = [0] * (matrix.cols - 1)
    #         valid_config = True

    #         for i in range(0, brute_start_index):
    #             brute_sum = sum(matrix[i, j] for j in combo)
    #             remaining_total = matrix[i, matrix.cols - 1] - brute_sum

    #             if remaining_total in [0, 1]:
    #                 mines_in_current_config[i] = remaining_total
    #             else:
    #                 valid_config = False
    #                 break

    #         if not valid_config:
    #             continue

    #         for i in range(brute_start_index + 1, matrix.rows):
    #             brute_sum = sum(matrix[i, j] for j in combo)
    #             remaining_total = matrix[i, matrix.cols - 1] - brute_sum

    #             if remaining_total == 0:
    #                 mines_in_current_config[i] = remaining_total
    #             else:
    #                 valid_config = False
    #                 break

    #         if not valid_config:
    #             continue

    #         for i in combo:
    #             mines_in_current_config[i] = 1

    #         for i, num_mines in enumerate(mines_in_current_config):
    #             num_mines_over_all_configs[i] += num_mines

    #     sure_moves = set()
    #     num_configs = 2 ** len(brute_indexes)

    #     for (i, num_mines) in enumerate(num_mines_over_all_configs):
    #         # If a frontier tile exclusively has a mine in all possible valid configuartions
    #         # or vice versa, then it is a definite safe move (no guessing
    #         # required).
    #         if num_mines in [0, num_configs]:
    #             (x, y) = frontier[i]
    #             is_mine = (num_mines == num_configs)
    #             sure_moves.add((x, y, is_mine))

    #     return sure_moves

    def getDisjointSections(self, sample):
        disjoint_sections = []

        for (sample_y, row) in enumerate(sample):
            for (sample_x, tile) in enumerate(row):
                # Only consider uncovered tiles with a number larger than 0, as those are the only
                # ones from which useful constraints can be made (and whether or not those constraints are disjoint
                # determines whether or not the sections are disjoint).
                if not tile or not tile.uncovered or tile.num_adjacent_mines == 0:
                    continue
                
                disjoint_sections = self.updateSampleDisjointSectionsBasedOnUncoveredTile(sample, disjoint_sections, tile)

        return disjoint_sections

    def updateSampleDisjointSectionsBasedOnUncoveredTile(self, sample, disjoint_sections, tile):
        adjacent_section = self.getAdjacentSectionForSampleTile(sample, tile)
        (brute, non_brute) = adjacent_section[:2]
           
        # Can only merge or find new sections based on frontier tiles in the adjacent section
        if brute or non_brute:
            disjoint_sections = self.updateDisjointSectionBasedOnAdjacentSection(disjoint_sections, adjacent_section)

        return disjoint_sections

    def getAdjacentSectionForSampleTile(self, sample, tile):
        adjacent_coords = [(tile.x + i, tile.y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
        mines_left_around_tile = tile.num_adjacent_mines
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
                (wall_dim_location, dim) = self.getWallLocation((tile.x, tile.y), (x, y))

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

        fringe = {(tile.x, tile.y, mines_left_around_tile)}

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
                if abs(frontier_x -
                       fringe_x) <= 1 and abs(frontier_y - fringe_y) <= 1:
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

        self.sure_moves_not_played_yet = self.pruneSureMoves(self.sure_moves_not_played_yet)

    '''
        Gets rid of sure moves that have been automatically uncovered
        because of a previous move.
    '''

    def pruneSureMoves(self, sure_moves):
        valid_sure_moves = set()

        for move in sure_moves:
            (x, y, _) = move

            tile_is_in_grid = (x >= 0 and y >= 0 and x < len(self.grid[0]) and y < len(self.grid))

            if tile_is_in_grid and not self.grid[y][x].uncovered and not self.grid[y][x].is_flagged:
                valid_sure_moves.add(move)

        return valid_sure_moves

    def onGameBegin(self):
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

    def cheekyHighlight(self, *args, transform=None):
        self.handleHighlights(*args, add_highlights=True, transform=transform)

    def removeHighlight(self, *args, transform=None):
        self.handleHighlights(*args, add_highlights=False, transform=transform)

    def cheekySampleHighlight(self, *args):
        self.handleHighlights(*args, add_highlights=True, transform=self.sample_pos)

    def removeSampleHighlight(self, *args):
        self.handleHighlights(*args, add_highlights=False, transform=self.sample_pos)

    def removeAllSampleHighlights(self, tiles):
        if not self.renderer:
            return
        
        tiles = deepflatten(tiles, depth=1)
        tile_coords = self.convertTilesToCoords(tiles, transform=self.sample_pos, return_removed_indexes=False)
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

        (tiles, codes) = self.getTilesAndCodesSeperate(*args)
        (coords, removed_indexes) = self.convertTilesToCoords(tiles, transform=transform)
        
        # Remove codes that were associated with the removed tiles
        codes = [code for (i, code) in enumerate(codes) if i not in removed_indexes]
        
        return list(zip(coords, codes))

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
            (tiles, code) = args
            code = str(code)

            # Flatten tiles down to 1D.
            tiles = list(deepflatten(tiles, types=(list, set)))
            codes = [code] * len(tiles)
        else:
            tiles_with_codes = args[0]

            try:
                # NOTE: this ain't the best check. This will fail to catch out anything erroneous
                # that happens to have its first element as length 2, e.g. [[tile_1, tile_2]]
                if len(tiles_with_codes[0]) != 2:
                    raise TypeError
            except:
                raise TypeError("Expected {} to be of form (Tile_like, highlight_code). Did you forget to pass the highlight code as a second parameter?".format(tiles_with_codes[0]))

            tiles, codes = zip(*tiles_with_codes)
            codes = list(map(str, codes))

        return (tiles, codes)

    def convertTilesToCoords(self, tiles, transform=None, return_removed_indexes=True):
        all_coords = []
        removed_indexes = []

        for i, tile in enumerate(tiles):
            # Filter out None-type tiles (which represent outside-grid walls).
            if not tile:
                removed_indexes.append(i)
                continue    

            try:
                if isinstance(tile, tuple):
                    coords = tile[:2]
                else:
                    coords = (tile.x, tile.y)
                
                (x, y) = map(int, coords)
            except:
                raise TypeError("Object {} is not Tile-like. It should either have x and y attributes, or be a tuple (x, y, ...), where x and y can be converted to int.".format(tile))
            
            if transform:
                x += transform[0]
                y += transform[1]
                
            inside_grid_bounds = (x >= 0 and y >= 0 and x < len(self.grid[0]) and y < len(self.grid))

            if inside_grid_bounds:
                all_coords.append((x, y))
            else:
                removed_indexes.append(i)

        if return_removed_indexes:
            return (all_coords, removed_indexes)
        else:
            return all_coords

    def feedRenderer(self, renderer):
        self.renderer = renderer


    class SolutionTracker(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.__variables = variables
            self.__solution_count = 0
            self.count = [0] * len(variables)

        def on_solution_callback(self):
            self.__solution_count += 1

            for (i, v) in enumerate(self.__variables):
                self.count[i] += self.Value(v)

        def result(self):
            # print(self.__solution_count, self.count)
            definite_solutions = []

            for (i, x) in enumerate(self.count):
                if x == 0:
                    definite_solutions.append((i, False))
                elif x == self.__solution_count:
                    definite_solutions.append((i, True))

            return definite_solutions


    def searchForDefiniteSolutionsUsingCpSolver(self, matrix_row_constraints):
        if not matrix_row_constraints:
            return []

        """Showcases calling the solver to search for all solutions."""
        # Creates the model.
        model = cp_model.CpModel()

        # Create the variables
        variables = [model.NewBoolVar(str(i)) for i in range(len(matrix_row_constraints[0]) - 1)]

        c = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]

        # Create the constraints.
        for constraint in matrix_row_constraints:
            x = [int(constraint[i]) * variables[i] for i in range(len(constraint) - 1) if constraint[i] != 0]
            sum_value = constraint[-1]
            model.Add(sum(x) == sum_value)
        
        # Create a solver and solve.
        solver = cp_model.CpSolver()
        solution_tracker = self.SolutionTracker(variables)
        status = solver.SearchForAllSolutions(model, solution_tracker)

        return solution_tracker.result()

    # c = [[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2],
    # [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1],
    # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, -1, -1, -1, 0, 0, -1],
    # [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 3],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, -1, 1, 0, 0]]

   

    # print(searchForDefiniteSolutionsUsingCpSolver(c))

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