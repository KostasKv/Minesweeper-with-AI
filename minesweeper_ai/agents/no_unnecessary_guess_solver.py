from random import Random
from copy import copy
from itertools import chain, combinations

from iteration_utilities import deepflatten

from .cp_solver import CpSolver
from .agent import Agent
from minesweeper_ai._game import _Game


class NoUnnecessaryGuessSolver(Agent):
    def __init__(self, sample_size=(5,5), use_num_mines_constraint=True, first_click_pos=None, seed=0):
        self.SAMPLE_SIZE = sample_size
        self.use_num_mines_constraint = use_num_mines_constraint
        self.first_click_pos = first_click_pos
        self.seed = seed

        self.random = Random(seed)  # This might need changing before experimenting. (agent always picks same first tile between games. Not so random)
        self.cp_solver = CpSolver()
        self.renderer = None
        self.sample_pos = None
        self.samples_considered_already = set()
        self.sample_selection_tiles_to_ignore = set()
        self.sure_moves_not_played_yet = set()

        # Used in games statistics
        self.sample_count = 0
        self.samples_with_solution_count = 0
        self.had_to_guess_this_game = False
        self.last_move_was_sure_move = None

    def nextMove(self):
        if self.game_state == _Game.State.START:
            move = self.getFirstMove()
            self.last_move_was_sure_move = False
        elif self.sure_moves_not_played_yet:
            move = self.sure_moves_not_played_yet.pop()
            self.last_move_was_sure_move = True
        else:
            sure_moves = self.lookForSureMovesFromGridSamplesFocussedOnGridSamples(self.SAMPLE_SIZE)

            if not sure_moves:
                # Try again, this time try every sample that could possibly give sure moves.
                sure_moves = self.lookForSureMovesFromAllUsefulGridSamples(self.SAMPLE_SIZE)
            
            if sure_moves:
                move = sure_moves.pop()
                self.sure_moves_not_played_yet.update(sure_moves)
                self.samples_with_solution_count += 1
                self.last_move_was_sure_move = True
            else:
                move = self.clickRandom()
                self.had_to_guess_this_game = True
                self.last_move_was_sure_move = False
        
        return move

    def getFirstMove(self):
        if self.first_click_pos is None:
            return self.clickRandom()
        else:
            return (*self.first_click_pos, False)

    # The method distinction between this and the one below helps with performance profiling since
    # program run time is then split in two between these two, rather than being stuck together under one
    # method name and so their seperate run times are hard to distinguish.
    def lookForSureMovesFromGridSamplesFocussedOnGridSamples(self, sample_size):
        temp_sample_hashes = set(self.samples_considered_already)
        moves1 = self.lookForSureMovesFromGridSamples(sample_size, limit_search_to_frontier=True)
        
        samples_considered_already_1 = set(self.samples_considered_already)

        self.samples_considered_already = set(temp_sample_hashes)
        moves2 = self.lookForSureMovesFromGridSamplesOPTIMISATIONTEST(sample_size, limit_search_to_frontier=True)
        self.samples_considered_already = set(samples_considered_already_1)

        if len(moves1) * len(moves2) == 0 and len(moves1) + len(moves2) > 0:
            raise Exception("One method didn't find sure moves the other did! Was there a mistake with the optimisation technique?")
        

        return moves1

    def lookForSureMovesFromAllUsefulGridSamples(self, sample_size):
        temp_sample_hashes = copy(self.samples_considered_already)
        moves1 = self.lookForSureMovesFromGridSamples(sample_size, limit_search_to_frontier=False)
        
        samples_considered_already_1 = copy(self.samples_considered_already)
        
        self.samples_considered_already = copy(temp_sample_hashes)
        moves2 = self.lookForSureMovesFromGridSamplesOPTIMISATIONTEST(sample_size, limit_search_to_frontier=False)
        self.samples_considered_already = copy(samples_considered_already_1)

        if len(moves1) * len(moves2) == 0 and len(moves1) + len(moves2) > 0:
            raise Exception("One method didn't find sure moves the other did! Was there a mistake with the optimisation technique?")

        return moves1
    
    def lookForSureMovesFromGridSamples(self, size, limit_search_to_frontier=False):
        samples = self.getUsefulSampleAreasFromGrid(size, limit_search_to_frontier=limit_search_to_frontier)

        for (sample, sample_pos) in samples:
            sample_hash = self.getSampleHash(sample, sample_pos)
            
            if sample_hash not in self.samples_considered_already:
                # self.highlightSample(sample)
                
                self.samples_considered_already.add(sample_hash)
                sure_moves = self.getAllSureMovesFromSample(sample, sample_pos)
                # self.removeAllSampleHighlights(sample)
                if sure_moves:
                    return sure_moves
                    
        # No sure moves found
        return set()

    def lookForSureMovesFromGridSamplesOPTIMISATIONTEST(self, size, limit_search_to_frontier=False):
        # samples = self.getUsefulSampleAreasFromGrid(size, limit_search_to_frontier=limit_search_to_frontier)
        samples = self.getUsefulSampleAreasFromGridOPTIMISATIONTEST(size, limit_search_to_frontier=limit_search_to_frontier)
        
        for (sample, sample_pos) in samples:
            sample_hash = self.getSampleHash(sample, sample_pos)
            
            if sample_hash not in self.samples_considered_already:
                # self.highlightSample(sample)
                
                self.samples_considered_already.add(sample_hash)
                sure_moves = self.getAllSureMovesFromSample(sample, sample_pos)
                # self.removeAllSampleHighlights(sample)
                if sure_moves:
                    return sure_moves
                    
        # No sure moves found
        return set()
    
    def getUsefulSampleAreasFromGrid(self, size, limit_search_to_frontier=False):
        # Note that these sample positions will include the outside grid wall (1 tile thick at most)
        # in the samples. Knowing a sample is beside a wall is useful info and can lead to sure moves.
        if limit_search_to_frontier:
            sample_positions = self.getAllSamplePosCenteredOnFrontier(size)
        else:
            sample_positions = self.getAllSamplePosOfSamplesWhichCouldGiveSolutions(size)

        for pos in sample_positions:
            sample = self.getSampleAtPosition(pos, size, self.grid)
            yield (sample, pos)

    def getAllSamplePosCenteredOnFrontier(self, size):
        frontier_tile_coords = self.getAllFrontierTileCoords()

        for (x, y) in frontier_tile_coords:
            yield self.getSamplePosOfSampleCenteredOnTile(x, y, size)


    def getUsefulSampleAreasFromGridOPTIMISATIONTEST(self, size, limit_search_to_frontier=False):
        # Note that these sample positions will include the outside grid wall (1 tile thick at most)
        # in the samples. Knowing a sample is beside a wall is useful info and can lead to sure moves.
        if limit_search_to_frontier:
            # sample_positions = self.getAllSamplePosCenteredOnFrontier(size)
            sample_positions = self.getAllSamplePosCenteredOnFrontierOPTIMISATIONTEST(size)
        else:
            sample_positions = self.getAllSamplePosOfSamplesWhichCouldGiveSolutions(size)

        for pos in sample_positions:
            sample = self.getSampleAtPosition(pos, size, self.grid)
            yield (sample, pos)

    def getAllSamplePosCenteredOnFrontierOPTIMISATIONTEST(self, size):
        # frontier_tile_coords = self.getAllFrontierTileCoords()
        disjoint_sections = self.getAllDisjointSections()
        

        for (frontier, fringe) in disjoint_sections:
            frontier = {(x - 1, y - 1) for (x, y) in frontier}
            fringe = {(x - 1, y - 1) for (x, y, _) in fringe}
            
            # sample_positions = self.getAllSamplesPosFullyContainingFrontier(frontier_coords, self.grid, size)
            sample_pos = self.getSampleThatContainsSectionEntirely((frontier, fringe), self.grid, size)


            if sample_pos:
                yield sample_pos

                self.sample_selection_tiles_to_ignore |= frontier | fringe

                # for pos in sample_positions:
                #     redundant_sample = self.getSampleAtPosition(pos, size, self.grid)
                #     sample_hash = self.getSampleHash(redundant_sample, pos)
                #     self.samples_considered_already.add(sample_hash)

                # yield sample_to_yield
            else:
                for (x, y) in frontier:
                    yield self.getSamplePosOfSampleCenteredOnTile(x, y, size)

    def sampleContainingWholeGridAndBoundary(self):
        boundary_row = [None] * (len(self.grid[0]) + 2)
        whole_grid_sample = []

        # Build full-grid containing sample
        whole_grid_sample.append(boundary_row)
        for row in self.grid:
            whole_grid_sample.append([None] + row + [None])
        whole_grid_sample.append(boundary_row)
        
        return whole_grid_sample

    def getAllDisjointSections(self):
        whole_grid_sample = self.sampleContainingWholeGridAndBoundary()
        self.sample_pos = (-1, -1)
        # self.highlightSample(whole_grid_sample)
        disjoint_sections = self.getDisjointSections(whole_grid_sample)
        # h = self.disjointSectionsToHighlights(disjoint_sections)
        # self.cheekyHighlight(h)
        # disjoint_frontiers = []

        # for (i, section) in enumerate(disjoint_sections, 4):
        #     (frontier, _) = section
        #     disjoint_frontiers.append(frontier)

        #     i = max(4, i % 12)
        #     self.cheekyHighlight(frontier, i)

        # return disjoint_frontiers
        return disjoint_sections

    def getSampleThatContainsSectionEntirely(self, section, grid, sample_size):
        ''' Returns top-left coords of sample containing all the tile information relevant
            to deciding sure moves of the given section if such a sample exists.
            Otherwise, returns None.
            Note that fringe tiles depend on their adjacent tiles, so a fringe tile that
            initially seems to to be on the perimeter of the section actually extends the
            perimeter width/height by 1 extra tile to include those adjacent tiles.
        '''
        (frontier, fringe) = section

        frontier_coords_sorted_by_x = sorted(frontier, key=lambda coords: coords[0])
        frontier_coords_sorted_by_y = sorted(frontier, key=lambda coords: coords[1])
        fringe_coords_sorted_by_x = sorted(fringe, key=lambda coords: coords[0])
        fringe_coords_sorted_by_y = sorted(fringe, key=lambda coords: coords[1])

        frontier_min_x = frontier_coords_sorted_by_x[0][0]
        frontier_max_x = frontier_coords_sorted_by_x[-1][0]
        frontier_min_y = frontier_coords_sorted_by_y[0][1]
        frontier_max_y = frontier_coords_sorted_by_y[-1][1]

        fringe_min_x = fringe_coords_sorted_by_x[0][0]
        fringe_max_x = fringe_coords_sorted_by_x[-1][0]
        fringe_min_y = fringe_coords_sorted_by_y[0][1]
        fringe_max_y = fringe_coords_sorted_by_y[-1][1]

        # Get section's perimeter min and max coords. Fringe tiles depend on adjacent
        # tiles which need to be included in the sample hence the extra tile included for
        # fringe coords
        section_min_x = min(frontier_min_x, fringe_min_x - 1)
        section_max_x = min(frontier_max_x, fringe_max_x + 1)
        section_min_y = min(frontier_min_y, fringe_min_y - 1)
        section_max_y = min(frontier_max_y, fringe_max_y + 1)

        width = section_max_x - section_min_x + 1
        height = section_max_y - section_min_y + 1
        frontier_tiles_can_fit_inside_sample = (width <= sample_size[0]) \
                                           and (height <= sample_size[1])

        if frontier_tiles_can_fit_inside_sample:
            sample_pos = (section_min_x, section_min_y)
        else:
            sample_pos = None
        
        return sample_pos

    # def getAllSamplesPosFullyContainingFrontier(self, frontier_coords, grid, sample_size):
    #     coords_sorted_by_x = sorted(frontier_coords, key=lambda coords: coords[0])
    #     coords_sorted_by_y = sorted(frontier_coords, key=lambda coords: coords[1])

    #     (min_x, _) = coords_sorted_by_x[0]
    #     (max_x, _)  = coords_sorted_by_x[-1]
    #     (_, min_y) = coords_sorted_by_y[0]
    #     (_,max_y)  = coords_sorted_by_y[-1]

    #     frontier_width = max_x - min_x + 1
    #     frontier_height = max_y - min_y + 1

    #     frontier_tiles_can_fit_inside_sample = (frontier_width <= sample_size[0]) \
    #                                        and (frontier_height <= sample_size[1])
        
    #     sample_positions = []

    #     if frontier_tiles_can_fit_inside_sample:
    #         x_wiggle_room = sample_size[0] - frontier_width
    #         y_wiggle_room = sample_size[1] - frontier_height

    #         for x in range(min_x - x_wiggle_room, min_x + 1):
    #             for y in range(min_y - y_wiggle_room, min_y + 1):
    #                 s = self.getSampleAtPosition((x, y), sample_size, grid)
    #                 self.highlightSample(s)
    #                 self.removeAllSampleHighlights(s)
    #                 sample_positions.append((x, y))
            
        
    #     return sample_positions

    def getAllFrontierTileCoords(self):
        for (y, row) in enumerate(self.grid):
            for (x, tile) in enumerate(row):
                if tile.uncovered or tile.is_flagged:
                    continue

                if self.isCoveredTileAFrontierTile(x, y):
                    yield (x, y)

    def getSamplePosOfSampleCenteredOnTile(self, tile_x, tile_y, sample_size):
        # Tile is either at exact center of sample, or slightly left/top of center
        x_offset = ((sample_size[1] - 1) // 2)
        y_offset = ((sample_size[0] - 1) // 2)
        x = tile_x - x_offset
        y = tile_y - y_offset

        # Bound sample to include at most 1 tile thick outside-grid wall tiles.
        max_sample_x = len(self.grid[0]) - sample_size[1] + 1
        max_sample_y = len(self.grid) - sample_size[0] + 1
        x = min(max(x, -1), max_sample_x)
        y = min(max(y, -1), max_sample_y)

        return (x, y)

    def isCoveredTileAFrontierTile(self, tile_x, tile_y):
        '''Assumes input is a non-flagged covered tile '''

        adjacent_coords = [((tile_x + i), (tile_y + j)) for i in (-1, 0, 1) for j in (-1, 0, 1)]

        for (x, y) in adjacent_coords:
            if x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid):
                continue

            if self.grid[y][x].uncovered:
                return True
        
        return False

    def getAllSamplePosOfSamplesWhichCouldGiveSolutions(self, size):
        ''' Returns all sample top-left pos (x, y) where the sample contains atleast one
            uncovered tile and one covered (non-flagged) tile. '''
        (sample_rows, sample_cols) = size
        min_x = -1
        min_y = -1
        max_x = len(self.grid[0]) - sample_cols + 1
        max_y = len(self.grid) - sample_rows + 1

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                if self.sampleCouldGiveSureMoves((x, y), sample_rows, sample_cols):
                    yield (x, y)

    def sampleCouldGiveSureMoves(self, pos, sample_rows, sample_cols):
        ''' A sample is considered to possible contain sure moves if it contains both
            a covered tile and an uncovered tile. i.e., it is not a sample consisting of entirely uncovered tiles
            or entirely covered tiles.
            
            This excludes tiles in self.samples_selection_tiles_to_ignore which are tiles
            that are part of some disjoint section that has already been wholly checked for
            sure moves and is confirmed to not contain any. Therefore having these tiles in a
            sample cannot lead to any sure moves this turn and so are ignored.'''

        exists_covered = False
        exists_uncovered = False

        for x in range(pos[0] - 1, pos[0] + sample_cols + 1):
            for y in range(pos[1] - 1, pos[1] + sample_rows + 1):
                # Out of grid bounds
                if x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid):
                    continue
                
                # Is part of a section that has already been completely checked this turn.
                # That means it cannot lead to any solutions this turn so ignore it.
                if (x, y) in self.sample_selection_tiles_to_ignore:
                    continue

                if self.grid[y][x].uncovered:
                    exists_uncovered = True
                elif not self.grid[y][x].is_flagged:
                    exists_covered = True

                if exists_covered and exists_uncovered:
                    return True
        
        return False

    def getSampleAtPosition(self, pos, size, grid):
        self.sample_pos = pos
        (x, y) = pos
        (rows, columns) = size

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
    def getSampleHash(sample, sample_pos):
        tiles = chain.from_iterable(sample)

        # Wall tile      --> None
        # Uncovered tile --> Num adjacent mines
        # Covered tile   --> -1
        simpler_sample = tuple(None if tile is None else tile.num_adjacent_mines if tile.uncovered else -1 for tile in tiles)

        return hash((simpler_sample, sample_pos))

    def getAllSureMovesFromSample(self, sample, sample_pos):
        self.sample_count += 1

        sps_sure_moves = self.singlePointStrategy(sample)
        disjoint_sections = self.getDisjointSections(sample)

        if disjoint_sections:
            if self.use_num_mines_constraint:
                brute_sure_moves = self.bruteForceWithAllConstraints(sample, disjoint_sections, self.mines_left, sps_sure_moves)
            else:
                brute_sure_moves = self.bruteForceWithJustAdacentMinesConstraints(sample, disjoint_sections, sps_sure_moves) 
        else:
            brute_sure_moves = set()

        sure_moves = sps_sure_moves | brute_sure_moves

        if sure_moves:
            sure_moves = self.sampleMovesToGridMoves(sure_moves, sample_pos)
            sure_moves = self.pruneIllegalSureMoves(sure_moves)

        return sure_moves

    def singlePointStrategy(self, sample):
        adjacent_info = list(self.getTilesAndAdjacentsOfInterestForSPS(sample))
        all_sure_mines = set()
        all_sure_safe = set()

        moves_found = True

        # Sure moves found after an iteration can lead to the discovery of new sure moves in the same sample.
        # Therefore, SPS should be repeated a bunch until it can no longer find any more sure moves in the sample.
        while moves_found:
            moves_found = False

            for i in range(len(adjacent_info) - 1, -1, -1):
                (num_adjacent_unknown_mines, adjacent_unknown) = adjacent_info[i]
                sure_mines = set()
                sure_safe = set()
                
                adjacent_unknown -= all_sure_safe
                unknown_discovered_to_be_mines = (adjacent_unknown & all_sure_mines)

                if unknown_discovered_to_be_mines:
                    num_adjacent_unknown_mines -= len(unknown_discovered_to_be_mines)
                    adjacent_unknown -= unknown_discovered_to_be_mines                

                if num_adjacent_unknown_mines == 0:
                    sure_safe.update(adjacent_unknown)
                elif num_adjacent_unknown_mines == len(adjacent_unknown):
                    sure_mines.update(adjacent_unknown)

                if sure_mines or sure_safe:
                    moves_found = True
                    all_sure_mines.update(sure_mines)
                    all_sure_safe.update(sure_safe)

                    # Once sure moves are found around a tile, it can't give us any more sure moves.
                    adjacent_info.pop(i)
                else:
                    # Update info
                    adjacent_info[i] = (num_adjacent_unknown_mines, adjacent_unknown)
            
        return {(x, y, True) for (x, y) in all_sure_mines} | {(x, y, False) for (x, y) in all_sure_safe}

    def getTilesAndAdjacentsOfInterestForSPS(self, sample):
        for (y, row) in enumerate(sample):
            for (x, tile) in enumerate(row):
                if not tile or not tile.uncovered or tile.num_adjacent_mines < 1:
                    continue

                num_adjacent_unknown_mines = tile.num_adjacent_mines
                adjacent_coords = [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
                
                adjacent_unknown = set()

                while adjacent_coords:
                    (adj_x, adj_y) = adjacent_coords.pop()

                    # Tiles outside sample are assumed to be covered unknown tiles
                    if adj_x < 0 or adj_y < 0 or adj_x >= len(sample[0]) or adj_y >= len(sample):
                        adjacent_unknown.add((adj_x, adj_y))
                        continue
                    
                    if sample[adj_y][adj_x] is None:
                        # If dim==0, wall is a column at (wall_dim_location, y), for any y.
                        # If dim==1, wall is a row at (x, wall_dim_location), for any x.
                        (wall_dim_location, dim) = self.getWallLocation((x, y), (adj_x, adj_y))

                        # If it's known where the entire wall is, remove all coordinates belonging to that wall.
                        if wall_dim_location is not None:
                            adjacent_unknown = {coords for coords in adjacent_unknown if coords[dim] != wall_dim_location}
                            adjacent_coords = [coords for coords in adjacent_coords if coords[dim] != wall_dim_location]

                        continue

                    if sample[adj_y][adj_x].uncovered:
                        continue

                    if sample[adj_y][adj_x].is_flagged:
                        num_adjacent_unknown_mines -= 1
                    else:
                        adjacent_unknown.add((adj_x, adj_y))

                # Can't discover moves with SPS if there are no unknown tiles around the uncovered tile
                if not adjacent_unknown:
                    continue
                
                yield (num_adjacent_unknown_mines, adjacent_unknown)

    def bruteForceWithAllConstraints(self, sample, disjoint_sections, mines_left, sure_moves):
        (num_tiles_outside_sample_surrounding, num_unknown_tiles_outside, num_unknown_tiles_inside) = self.getNumTilesOutsideSampleSurroundingAndNumUnknownTilesForSample(sample)

        # It's pointless to try bruteforcing if all unknown tiles from sample already have their solutions discovered
        if len(sure_moves) == (num_unknown_tiles_outside + num_unknown_tiles_inside):
            return set()

        # It's faster (and much simpler) to bruteforce one merged section if including total mines left constraint.
        all_sections_as_one = self.mergeSections(disjoint_sections)

        frontier_tile_is_inside_sample = []

        for (x, y) in all_sections_as_one[0]:
            is_inside_sample = x >= 0 and y >= 0 and x < len(sample[0]) and y < len(sample)
            if is_inside_sample:
                frontier_tile_is_inside_sample.append(1)
                num_unknown_tiles_inside -= 1
            else:
                frontier_tile_is_inside_sample.append(0)
                num_unknown_tiles_outside -= 1

        return self.bruteForceSection(sample, all_sections_as_one, sure_moves, num_unknown_tiles_outside, num_unknown_tiles_inside, frontier_tile_is_inside_sample, mines_left, num_tiles_outside_sample_surrounding)

    def getNumTilesOutsideSampleSurroundingAndNumUnknownTilesForSample(self, sample):
        # Calculate how many unknown tiles there are from given sample (including surrounding area) for which a solution could possibly be found
        num_unknown_inside_sample = sum(1 for tile in chain.from_iterable(sample) if tile and not tile.uncovered and not tile.is_flagged)

        (tiles_outside_sample_surrounding, num_unknown_outside_sample) = self.getNumTilesOutsideSampleSurroundingAndNumUnknownSurroundingSample(sample)

        return (tiles_outside_sample_surrounding, num_unknown_outside_sample, num_unknown_inside_sample)

    def getNumTilesOutsideSampleSurroundingAndNumUnknownSurroundingSample(self, sample):
        non_wall_tiles_in_sample = 0
        top_is_wall = False
        right_is_wall = False
        bottom_is_wall = False
        left_is_wall = False
        max_x = len(sample[0]) - 1
        max_y = len(sample) - 1

        # Count number of non wall tiles inside sample, and figure out where the walls are (relative to sample) if any are detected.
        for (y, row) in enumerate(sample):
            for (x, tile) in enumerate(row):
                if tile is None:
                    is_corner = (x, y) in [(0, 0), (0, max_y), (max_x, 0), (max_x, max_y)]

                    # Can't assume whether row or column is walls from a corner.
                    if is_corner:
                        continue

                    if y == 0:
                        top_is_wall = True
                    elif y == max_y:
                        bottom_is_wall = True
                    
                    if x == 0:
                        left_is_wall = True
                    elif x == max_x:
                        right_is_wall = True
                else:
                    non_wall_tiles_in_sample += 1

        t = int(top_is_wall)
        r = int(right_is_wall)
        b = int(bottom_is_wall)
        l = int(left_is_wall)

        # Calculate how many non-wall tiles are in and around sample
        non_wall_width = len(sample[0]) + 2 - (2 * (l + r))
        non_wall_height = len(sample) + 2 - (2 * (t + b))
        tiles_inside_sample_and_surrounding = non_wall_width * non_wall_height

        # Calculate how many tiles are outside sample and its surrounding area
        total_tiles_in_grid = len(self.grid[0]) * len(self.grid)
        tiles_outside_sample_surrounding = max(total_tiles_in_grid - tiles_inside_sample_and_surrounding, 0)

        unknown_tiles_surrounding_sample = tiles_inside_sample_and_surrounding - non_wall_tiles_in_sample

        return (tiles_outside_sample_surrounding, unknown_tiles_surrounding_sample)

    def bruteForceWithJustAdacentMinesConstraints(self, sample, disjoint_sections, sure_moves):
        # When only using adjacent mines constraints, the number unknown tiles that could possibly give
        # solutions from sample is exactly equal to number of frontier tiles. All other unknown tiles in and around sample have
        # no adjacent uncovered tiles and so cannot yield a solution from this bruteforce.
        (all_frontier_tiles, _) = self.mergeSections(disjoint_sections)
        num_unknown_from_sample = len(all_frontier_tiles)

        # It's pointless to try bruteforcing if all unknown tiles from sample already have their solutions discovered
        if len(sure_moves) == num_unknown_from_sample:
            return set()

        all_sure_moves = set()

        # It's faster to bruteforce each disjoint section seperately if only considering adjacent mine constraints
        for section in disjoint_sections:
            section_sure_moves = self.bruteForceSection(sample, section, sure_moves)
            all_sure_moves.update(section_sure_moves)
        
        return all_sure_moves

    def bruteForceSection(self, sample, section, sure_moves, num_unknown_tiles_outside=None, num_unknown_tiles_inside=None, frontier_tile_is_inside_sample=None, total_mines_left=None, num_tiles_outside_sample=None):
        (frontier, fringe) = section

        # Need tiles to be ordered so that a specific matrix column can refer to a specific tile
        # (column i in matrix (left-to-right asc., 0-based) represents tile at index i in list).
        frontier = list(frontier)
        adjacent_mines_constraints = self.createAdjacentMinesConstraintMatrixOfSample(frontier, fringe)

        sure_moves_indexes = {(frontier.index((x, y)), is_mine) for (x, y, is_mine) in sure_moves if ((x, y) in frontier)}

        bruted_sure_moves, unknown_definite_solution = self.bruteForceUsingConstraintsSolver(frontier, adjacent_mines_constraints, sure_moves_indexes, num_unknown_tiles_outside, num_unknown_tiles_inside, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample)
        
        if unknown_definite_solution is not None:
            for (y, row) in enumerate(sample):
                for (x, tile) in enumerate(row):
                    if (x, y) in sure_moves or (x, y) in bruted_sure_moves:
                        continue

                    if not tile.uncovered and not tile.flagged:
                        sure_move_of_unknown_tile = (x, y, unknown_definite_solution)
                        bruted_sure_moves.add(sure_move_of_unknown_tile)

        return bruted_sure_moves

    def bruteForceUsingConstraintsSolver(self, frontier, adjacent_mines_constraints, sure_moves_indexes, num_unknown_tiles_outside=None, num_unknown_tiles_inside=None, frontier_tile_is_inside_sample=None, total_mines_left=None, num_tiles_outside_sample=None):
        (frontier_definite_solutions, unknown_definite_solution) =  self.cp_solver.searchForDefiniteSolutions(adjacent_mines_constraints, sure_moves_indexes, num_unknown_tiles_outside, num_unknown_tiles_inside, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample)

        sure_moves = set()

        # Convert frontier results into sure moves.
        for (index, is_mine) in frontier_definite_solutions:
            # Don't consider the already known solutions
            if (index, is_mine) in sure_moves_indexes:
                continue

            coords = frontier[index]
            sure_moves.add((*coords, is_mine,))

        return sure_moves, unknown_definite_solution
                
    def getDisjointSections(self, sample):
        disjoint_sections = []

        for (y, row) in enumerate(sample):
            for (x, tile) in enumerate(row):
                # Only consider uncovered tiles with a number larger than 0, as those are the only
                # ones from which useful constraints can be made (and whether or not those constraints are disjoint
                # determines whether or not the sections are disjoint).
                if not tile or not tile.uncovered or tile.num_adjacent_mines < 1:
                    continue
                # self.cheekyHighlight((x, y), 4)
                disjoint_sections = self.updateSampleDisjointSectionsBasedOnUncoveredTile(sample, disjoint_sections, (x, y), tile.num_adjacent_mines)
                # h = self.disjointSectionsToHighlights(disjoint_sections)
                # self.cheekyHighlight(h)
                # self.removeHighlight(h)
                # self.removeHighlight((x, y), 4)

        return disjoint_sections

    def updateSampleDisjointSectionsBasedOnUncoveredTile(self, sample, disjoint_sections, tile_coords, num_adjacent_mines):
        adjacent_section = self.getAdjacentSectionForUncoveredSampleTile(sample, tile_coords, num_adjacent_mines)
        frontier = adjacent_section[0]
        # self.cheekyHighlight(frontier, 7)
        # self.removeHighlight(frontier, 7)
           
        # Can only merge or find new sections based on frontier tiles in the adjacent section
        if frontier:
            disjoint_sections = self.updateDisjointSectionBasedOnAdjacentSection(disjoint_sections, adjacent_section)

        return disjoint_sections

    def getAdjacentSectionForUncoveredSampleTile(self, sample, tile_coords, num_adjacent_mines):
        adjacent_coords = [(tile_coords[0] + i, tile_coords[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if not (i == 0 and j == 0)]
        mines_left_around_tile = num_adjacent_mines
        frontier = set()

        while adjacent_coords:
            (x, y) = adjacent_coords.pop()
            
            # Outside tile is assumed to be covered, therefore this adjacent tile
            # is considered a frontier tile.
            if x < 0 or y < 0 or x >= len(sample[0]) or y >= len(sample):
                frontier.add((x, y))
                continue
            
            adjacent = sample[y][x]

            # If we know, or can figure out, that a tile is a wall tile then
            # exclude it from the section (even if that wall tile is outside the sample)
            if adjacent is None:
                # If dim==0, wall is a row at (wall_dim_location, y), for any y.
                # If dim==1, wall is a colum at (x, wall_dim_location), for any x.
                (wall_dim_location, dim) = self.getWallLocation(tile_coords, (x, y))

                # If it's known where the entire wall is, remove all coordinates belonging to that wall.
                if wall_dim_location is not None:
                    frontier = set(coords for coords in frontier if coords[dim] != wall_dim_location)
                    adjacent_coords = [coords for coords in adjacent_coords if coords[dim] != wall_dim_location]

                continue

            if adjacent.uncovered:
                continue

            if adjacent.is_flagged:
                mines_left_around_tile -= 1
            else:
                frontier.add((x, y))

        fringe = {(*tile_coords, mines_left_around_tile)}

        return (frontier, fringe)

    def updateDisjointSectionBasedOnAdjacentSection(self, disjoint_sections, adjacent_section):
        (frontier, fringe) = adjacent_section

        updated_disjoint_sections = []
        sections_to_merge = []

        # 'Disjoint' sections that share any of the frontier tiles are not really
        # disjoint; they should be merged.
        for section in disjoint_sections:
            section_frontier = section[0]
            section_is_disjoint_from_adjacent = (section_frontier.isdisjoint(frontier))

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
        (all_frontier_sets, all_fringe_sets) = zip(*sections_to_merge)
        return (set.union(*all_frontier_sets), set.union(*all_fringe_sets))

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

    def createAdjacentMinesConstraintMatrixOfSample(self, frontier, fringe):
        matrix = []

        # Build up matrix of row equations.
        for (fringe_x, fringe_y, num_unknown_adjacent_mines) in fringe:
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
            matrix_row.append(num_unknown_adjacent_mines)

            matrix.append(matrix_row)

        return matrix

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
        if game_state in [_Game.State.LOSE, _Game.State.ILLEGAL_MOVE] and self.last_move_was_sure_move:
            raise AssertionError("Sure move lost a game!")

        # Removes coordinates from every tile. A sample's coordinates will be inferred
        # from the element's position in the sample.
        converted_grid = [[SampleTile(tile) for tile in row] for row in grid]
        self.grid = converted_grid

        self.mines_left = mines_left
        self.game_state = game_state

        # Last played move could have uncovered multiple tiles, making some sure moves now illegal moves.
        self.sure_moves_not_played_yet = self.pruneIllegalSureMoves(self.sure_moves_not_played_yet)
        self.last_move_was_sure_move = None

        self.sample_selection_tiles_to_ignore = set()

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
        # Use full grid & boundaries if sample size is large enough to contain the full grid (or if set to None).
        # Instead of putting logic of deducing that the sample contains the full grid (thus its surroundings are boundaries, rather than unknown tiles)
        # in the solving algorithms, it's much easier to just default to a sample size that contains entire grid & boundaries in this case.
        # The result is the same.
        use_full_grid = self.SAMPLE_SIZE is None or (self.SAMPLE_SIZE[0] >= len(self.grid) and self.SAMPLE_SIZE[1] >= len(self.grid[0]))
        if use_full_grid :
            self.SAMPLE_SIZE = (len(self.grid) + 2, len(self.grid[0]) + 2)

        self.random = Random(self.seed)
        self.sure_moves_not_played_yet = set()
        self.samples_considered_already = set()
        self.had_to_guess_this_game = False

    @staticmethod
    def disjointSectionsToHighlights(sections):
        ''' 
            Each disjoint section is given a certain highlight code. This highlight
            cycles through a number of consecutive numbered highlights, stepping through
            each highlight in the cycle once per disjoint section.
        '''
        START_HIGHLIGHT_NUM = 7
        END_HIGHLIGHT_NUM = 10
        
        highlights = []

        code_i = START_HIGHLIGHT_NUM

        for (frontier, fringe) in sections:
            all_section_tiles = frontier | fringe
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

        # Remove coordinates that are out of bounds
        for i in range(len(tile_coords) -1, -1, -1):
            (x, y) = tile_coords[i]
            outside_grid_bounds = (x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid))

            if outside_grid_bounds:
                tile_coords.pop(i)
                codes.pop(i)
    
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

        if tile.uncovered:
            self.num_adjacent_mines = tile.num_adjacent_mines
        else:
            self.is_flagged = tile.is_flagged

