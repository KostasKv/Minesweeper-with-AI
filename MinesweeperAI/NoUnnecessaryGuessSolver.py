from Agent import Agent
from Game import Game
from random import randint
from itertools import chain, combinations
from sympy import Matrix, pprint
from iteration_utilities import deepflatten
from copy import copy


class NoUnnecessaryGuessSolver(Agent):
    def __init__(self):
        self.sure_moves_not_played_yet = set()
        self.frontier_tiles = []
        self.disjoint_frontiers_and_fringes = []
        self.SAMPLE_SIZE = (5, 5)
        self.sample_pos = None

    def nextMove(self):
        if self.game_state == Game.State.START:
            move = self.clickRandom()
        elif self.sure_moves_not_played_yet:
            move = self.sure_moves_not_played_yet.pop()
        else:
            sure_moves = self.lookForSureMovesFromSamplesOfGrid(
                self.SAMPLE_SIZE)

            if sure_moves:
                move = sure_moves.pop()
                self.sure_moves_not_played_yet.update(sure_moves)
            else:
                move = self.clickRandom()

        return move

    def lookForSureMovesFromSamplesOfGrid(self, size):
        samples = self.getSampleAreasFromGrid(size)

        for (sample, sample_pos) in samples:
            # sure_moves_found = self.singlePointStrategy(sample)
            # extra_sure_moves_found = self.matrixAndBruteForceStrategies(sample)
            # sure_moves_found.update(extra_sure_moves_found)
            sure_moves_found = self.matrixAndBruteForceStrategies(sample)

            # DEBUG
            if sure_moves_found:
                self.cheekySampleHighlight(self.sampleToHighlights(sample))

                highlights = []
                for (x, y, is_mine) in sure_moves_found:
                    if is_mine:
                        code = 12
                    else:
                        code = 11
                    highlights.append(((x, y), code))
                self.cheekySampleHighlight(highlights)

                highlights = []
                for (x, y, is_mine) in sure_moves_found:
                    if is_mine:
                        code = 12
                    else:
                        code = 11
                    highlights.append(((x, y), code))
                self.removeHighlight(highlights)

                self.removeHighlight(self.sampleToHighlights(sample))

            self.removeAllSampleHighlights(sample)
            # if sure_moves_found:
            #     break

        # self.removeHighlights(sample, 2)

        return sure_moves_found

    '''
        Note that (shallow) copies of the grid's Tile objects are
        used. This is so that a change in a sample's tile doesn't accidentally make a
        a change to the grid itself. This way the solving algorithms can
        'mark' a sample with solutions while keeping unique samples independant from eachother.
    '''

    def getSampleAreasFromGrid(self, size):
        max_x = len(self.grid[0]) - size[0]
        max_y = len(self.grid) - size[1]

        for y in range(max_y + 1):
            for x in range(max_x + 1):
                pos = (x, y)
                sample = self.getSampleAtPosition(pos, size)
                self.sample_pos = pos
                yield (sample, pos)

    def getSampleAtPosition(self, pos, size):
        (x, y) = pos
        (rows, columns) = size
        rows_slice = slice(y, (y + columns))
        columns_slice = slice(x, (x + rows))

        # Note that sample will be surrounded with unknown tiles.
        max_x = columns + 1
        max_y = rows + 1

        sample = [[SampleOutsideTile(x, 0) for x in range(max_x + 1)]]

        # Create row of shallow-copy grid tiles, with unknown tiles on the
        # outsides of the row
        for sample_y, tile_row in enumerate(self.grid[rows_slice], 1):
            row = [SampleOutsideTile(0, sample_y)]

            for sample_x, tile in enumerate(tile_row[columns_slice], 1):
                copied_tile = copy(tile)
                copied_tile.x = sample_x
                copied_tile.y = sample_y
                row.append(copied_tile)

            row.append(SampleOutsideTile(max_x, sample_y))

            sample.append(row)

        sample.append([SampleOutsideTile(x, max_y) for x in range(max_x + 1)])

        return sample

    # def singlePointStrategy(self, sample):
    #     all_sure_moves_found = set()
    #     tiles_and_adjacents_of_interest = []

    #     for (sample_y, row) in enumerate(sample[1 : -1]):
    #         for (sample_x, tile) in enumerate(row[1 : -1]):
    #             # Skip tiles that can't be used to determine if neighbouring
    #             # tiles are/aren't mines using SPS.
    #             if not tile or isinstance(tile, SampleOutsideTile) or not tile.uncovered:
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

    @staticmethod
    def getAdjacentTilesInSample(
            tile_sample_coords, sample, return_sample_coords=False):
        max_x = len(sample[0]) - 1
        max_y = len(sample) - 1

        (x, y) = tile_sample_coords
        adjacents = []

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

                adjacent = sample[new_y][new_x]

                if return_sample_coords:
                    adjacent = (adjacent, (new_x, new_y))

                adjacents.append(adjacent)

        return adjacents

    # @staticmethod
    # def updateSampleWithSureMoves(sample, sure_moves_found):
    #     for (x, y, is_mine) in sure_moves_found:
    #         sample[y][x].setIsMine(is_mine)

    #     return sample

    # Implementation from
    # https://docs.python.org/3/library/itertools.html#recipes
    @staticmethod
    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r)
                                   for r in range(len(s) + 1))

    def matrixAndBruteForceStrategies(self, sample):
        sure_moves = set()
        self.disjoint_frontiers_and_fringes = \
            self.getDisjointFrontiersAndFringes(sample)

        
        # i = 0
        # highlights = []
        # for (frontier, fringe) in self.disjoint_frontiers_and_fringes:
        #     code = i + 1
        #     i = (i + 1) % len(self.disjoint_frontiers_and_fringes)

        #     x = list(frontier) + list(fringe)
        #     highlights.append((x, code))

        # self.cheekySampleHighlight(highlights)
        # self.removeSampleHighlight(highlights)

        bruteforce_candidates = []

        for (frontier, fringe) in self.disjoint_frontiers_and_fringes:
            # Case of 1 fringe tile has already been tried by single point
            # strategy
            if len(fringe) <= 1:
                continue

            self.cheekyHighlight(frontier, 2)
            self.cheekyHighlight(fringe, 4)

            # Tiles need an order as a way to reference which column of the
            # matrix represents which frontier tile. (i'th column in
            # matrix represents i'th tile in list)
            frontier = list(frontier)

            matrix = self.frontierAndFringeInSampleToMatrix(frontier, fringe)
            matrix = matrix.rref(pivots=False)

            (sure_moves, adjusted_matrix) = self.matrixQuickerSearch(frontier,
                                                                     matrix)

            # Bruteforce later if necessary
            if not sure_moves:
                bruteforce_candidates.append((frontier, adjusted_matrix))

            self.removeHighlight(frontier, 2)
            self.removeHighlight(fringe, 4)

        while (not sure_moves) and bruteforce_candidates:
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

        # DEBUG
        self.removeHighlight([self.grid[y][x] for (x, y, is_mine) in sure_moves if is_mine], 12)
        self.removeHighlight([self.grid[y][x] for (x, y, is_mine) in sure_moves if not is_mine], 11)

        return sure_moves, matrix

    @staticmethod
    def minMaxBoundarySolutionSearch(matrix):
        solutions = set()

        for i in range(matrix.rows):
            min_bound = 0
            max_bound = 0
            row_start_index = i * matrix.cols

            for j in range(matrix.cols - 1):
                min_bound = min(
                    min_bound + matrix[row_start_index + j], min_bound)
                max_bound = max(
                    max_bound + matrix[row_start_index + j], max_bound)

            min_bound_solution_exists = matrix[row_start_index +
                                               matrix.cols - 1] == min_bound
            max_bound_solution_exists = matrix[row_start_index +
                                               matrix.cols - 1] == max_bound

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
        nonzero_rows = [i for i in range(rows) if any(
            matrix[i, j] != 0 for j in range(cols))]
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

        brute_indexes = list(
            range(
                bruteforce_leftmost_col,
                adjusted_matrix.cols - 1))
        combinations = self.powerset(brute_indexes)
        num_mines_over_all_configs = [0] * (adjusted_matrix.cols - 1)
        can_be_definite_move = [True] * (adjusted_matrix.cols - 1)

        for combo in combinations:
            mines_in_current_config = [0] * (adjusted_matrix.cols - 1)
            valid_config = True

            for i in range(0, bruteforce_leftmost_col):
                brute_sum = sum(adjusted_matrix[i, j] for j in combo)
                remaining_total = adjusted_matrix[i,
                                                  adjusted_matrix.cols - 1] - brute_sum

                if remaining_total in [0, 1]:
                    mines_in_current_config[i] = remaining_total
                else:
                    valid_config = False
                    break

            if not valid_config:
                continue

            for i in range(bruteforce_leftmost_col + 1, adjusted_matrix.rows):
                brute_sum = sum(adjusted_matrix[i, j] for j in combo)
                remaining_total = adjusted_matrix[i,
                                                  adjusted_matrix.cols - 1] - brute_sum

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
            # or vice versa, then it is a definite safe move (no guessing
            # required).
            if num_mines in [0, num_configs]:
                (x, y) = frontier[i]
                is_mine = (num_mines == num_configs)
                sure_moves.add((x, y, is_mine))

        return sure_moves

    def getDisjointFrontiersAndFringes(self, sample):
        frontiers_and_associated_fringes = []

        self.cheekySampleHighlight(self.sampleToHighlights(sample))

        # For each uncovered tile, get its adjacent frontier tiles and decide whether those
        # belong in a known segregated group, or whether to create a new group
        # for them.
        for (sample_y, row) in enumerate(sample[1: -1], 1):
            for (sample_x, tile) in enumerate(row[1: -1], 1):
                if not tile.uncovered or isinstance(tile, SampleOutsideTile):
                    continue

                adjacent_tiles_with_coords = self.getAdjacentTilesInSample(
                    (sample_x, sample_y), sample, return_sample_coords=True)

                self.cheekySampleHighlight(tile, 4)
                self.cheekySampleHighlight([sample_coords for (_, sample_coords) in adjacent_tiles_with_coords])
                mines_left_around_tile = tile.num_adjacent_mines
                adjacent_frontier_tiles = []
                adjacent_frontier_tiles_that_are_bruteforceable = []

                for (adjacent, sample_coords) in adjacent_tiles_with_coords:
                    # Uncovered tiles can't be frontier tiles or be flagged, so
                    # skip.
                    if adjacent.uncovered:
                        continue

                    if adjacent.is_flagged:
                        mines_left_around_tile -= 1
                    else:
                        adjacent_frontier_tiles.append(
                            (adjacent.x, adjacent.y))

                        if self.isBruteforceableSampleTile(
                                sample, sample_coords):
                            adjacent_frontier_tiles_that_are_bruteforceable.append(
                                (adjacent.x, adjacent.y))

                if not adjacent_frontier_tiles:
                    continue

                common_frontiers_and_fringes = []

                # Looks for frontiers that shares any of the bruteforcable
                # adjacent frontier tiles
                for bruteforceable_tile in adjacent_frontier_tiles_that_are_bruteforceable:
                    for (frontier, fringe) in frontiers_and_associated_fringes:
                        if (bruteforceable_tile in frontier) and (
                                (frontier, fringe) not in common_frontiers_and_fringes):
                            common_frontiers_and_fringes.append(
                                (frontier, fringe))

                if common_frontiers_and_fringes:
                    # All frontiers that share any of the brutefoceable adjacent frontier tiles should be
                    # one frontier so merge them, and their associated fringe
                    # tiles too.
                    (some_group, some_fringe) = common_frontiers_and_fringes.pop()

                    # Add adjacents and fringe tile
                    some_group.update(adjacent_frontier_tiles)
                    some_fringe.add((tile.x, tile.y, mines_left_around_tile))

                    # Merge groups and merge fringe tiles
                    for group_and_fringe in common_frontiers_and_fringes:
                        some_group.update(group_and_fringe[0])
                        some_fringe.update(group_and_fringe[1])
                        frontiers_and_associated_fringes.remove(
                            group_and_fringe)
                else:
                    # New frontier discovered.
                    new_frontier = set(adjacent_frontier_tiles)
                    fringe = set()
                    fringe.add((tile.x, tile.y, mines_left_around_tile))
                    frontiers_and_associated_fringes.append(
                        (new_frontier, fringe))

        self.removeSampleHighlight(self.sampleToHighlights(sample))

        return frontiers_and_associated_fringes

    def isBruteforceableSampleTile(self, sample, tile_sample_coords):
        (x, y) = tile_sample_coords
        return (x > 1) and (y > 1) and (
            x < (len(sample[0]) - 2)) and (y < (len(sample) - 2))

    @staticmethod
    def frontierAndFringeInSampleToMatrix(frontier, fringe):
        matrix = []

        # Build up matrix of row equations.
        for (fringe_x, fringe_y,
             num_adjacent_unknown_mines_around_fringe_tile) in fringe:
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
        self.disjoint_frontiers_and_fringes = []

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
        for (frontier, fringe) in self.disjoint_frontiers_and_fringes:
            i = ((i + 1) % len(codes))

            for (x, y) in frontier:
                tile_highlight = (x, y, codes[i])
                tiles_to_highlight.append(tile_highlight)

            for (x, y, _) in fringe:
                tile_highlight = (x, y, codes[i])
                tiles_to_highlight.append(tile_highlight)

        return tiles_to_highlight

    def cheekyHighlight(self, *args, sample_pos=None):
        self.handleHighlights(*args, add_highlights=True, sample_pos=sample_pos)

    def cheekySampleHighlight(self, *args):
        self.handleHighlights(*args, add_highlights=True, sample_pos=self.sample_pos)

    def removeHighlight(self, *args, sample_pos=None):
        self.handleHighlights(*args, add_highlights=False, sample_pos=sample_pos)

    def removeSampleHighlight(self, *args):
        self.handleHighlights(*args, add_highlights=False, sample_pos=self.sample_pos)

    def removeAllSampleHighlights(self, tiles):
        tiles = deepflatten(tiles, depth=1)
        tile_coords = self.convertTilesToCoords(tiles)
        tile_coords = self.mapSampleCoordsToGridCoords(tile_coords, sample_pos=self.sample_pos)
        self.renderer.removeAllTileHighlights(tile_coords)

    def handleHighlights(self, *args, add_highlights=None, sample_pos=None):
        tile_coords_with_code = self.prepHighlights(*args, sample_pos=sample_pos)
        
        if not tile_coords_with_code:
            return
        
        self.renderer.highlightTilesAndDraw(tile_coords_with_code, add_highlights=add_highlights)

    def prepHighlights(self, *args, sample_pos=None):
        if len(args) not in [1, 2]:
            raise TypeError

        tiles_with_code = self.structureHighlights(*args)

        if not tiles_with_code:
            return []

        # Convert any Tile-like objects to an (x, y) tuple. Also removes any 'tiles' that are of None type.
        (tiles, codes) = zip(*tiles_with_code)
        coords = self.convertTilesToCoords(tiles)
        tile_coords_with_code = list(zip(coords, codes))

        if sample_pos and tile_coords_with_code:
            tile_coords_with_code = self.transformSampleCoordsWithCodeToGridCoordsWithCode(tile_coords_with_code, sample_pos)

        return tile_coords_with_code

    def structureHighlights(self, *args):
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
            # Pack tiles and code into a list of (tile, code) tuples
            (tiles, code) = args

            # Flatten tiles down to 1D.
            tiles = list(deepflatten(tiles, types=(list, set)))

            tiles_with_code = [(tile, code) for tile in tiles]
        else:
            tiles_with_code = args[0]

        return tiles_with_code

    def convertTilesToCoords(self, tiles):
        all_coords = []

        for tile in tiles:
            if not tile:
                continue

            if isinstance(tile, tuple):
                coords = tile[:2]
            else:
                coords = (tile.x, tile.y)

            all_coords.append(coords)

        return all_coords

    def transformSampleCoordsWithCodeToGridCoordsWithCode(self, coords_with_code, sample_pos):
        coords, codes = zip(*coords_with_code)
        grid_coords = self.mapSampleCoordsToGridCoords(coords, sample_pos)
        return list(zip(grid_coords, codes))


    def mapSampleCoordsToGridCoords(self, coords, sample_pos):
        (max_x, max_y) = len(self.grid[0]) - 1, len(self.grid) - 1
        (sample_x, sample_y) = sample_pos
        transformed_coords = []

        for (x, y) in coords:
            x += sample_x
            y += sample_y

            if x < 0 or y < 0 or x > max_x or y > max_y:
                continue
            
            transformed_coords.append((x, y))

        return transformed_coords

    def sampleToHighlights(self, sample):
        tiles_to_highlight = []

        for row in sample:
            for tile in row:
                # Don't try highlighting out-of-bound tiles.
                if tile.x < 0 or tile.y < 0 or tile.x >= len(self.grid[0]) or tile.y >= len(self.grid):
                    continue

                if isinstance(tile, SampleOutsideTile):
                    code = 6
                else:
                    code = 2

                tiles_to_highlight.append((tile, code))

        return tiles_to_highlight

    def feedRenderer(self, renderer):
        self.renderer = renderer


class SampleOutsideTile:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.unknown = True
        self.uncovered = True
        self.is_flagged = False

    def setIsFlagged(self, is_flagged):
        self.is_flagged = is_flagged
        self.uncovered = not is_flagged
        self.unknown = False
