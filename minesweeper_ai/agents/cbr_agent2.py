import random
import time
from minesweeper_ai._game import _Game
from .agent import Agent


class Case:
    """
    Case problem is a 5x5 grid, each tile takes on one of 12 string values: 0-8(num adjacent mines), -(covered), F (flag), W(wall)
    Solution is a boolean value; True means middle tile is a mine, False means it's not.
    """

    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution


class CBRAgent2(Agent):
    def __init__(self):
        self.case_base = []
        self.prev_case_with_mine_clicked = None
        self.cases_with_flag_and_target_coords = []
        self.tiles_chosen_from = []

        # Constants
        self.SAMPLE_ROWS = 5
        self.SAMPLE_COLUMNS = 5
        self.SAMPLE_ROWS_MID = (self.SAMPLE_ROWS - 1) // 2
        self.SAMPLE_COLUMNS_MID = (self.SAMPLE_COLUMNS - 1) // 2

    def nextMove(self):
        if self.game_state == _Game.State.START:
            # First click of a game is random
            action = self.getRandomLegalClickAction()
        else:
            action = self.chooseBestMove()

        return action

    def getRandomLegalClickAction(self):
        x = random.randint(0, len(self.grid[0]))
        y = random.randint(0, len(self.grid))

        while self.isIllegalMove(x, y, False):
            x = random.randint(0, len(self.grid[0]))
            y = random.randint(0, len(self.grid))

        return (x, y, False)

    def isIllegalMove(self, x, y, toggle_flag):
        # Out of bounds
        if x < 0 or y < 0 or x >= len(self.grid[0]) or y >= len(self.grid):
            return True

        # Tile already uncovered
        if self.grid[y][x].uncovered:
            return True

        # Can't uncover a flagged tile
        if not toggle_flag and self.grid[y][x].is_flagged:
            return True

        return False

    def chooseBestMove(self):
        case, target_coords = self.scanGridAndGetBestMove()

        flag_tile = case.solution

        # Remember case until its outcome is found out. For flagged cases, that's at the end
        # of a game. For mine clicks, that's immediately after the move has been made.
        if flag_tile:
            self.cases_with_flag_and_target_coords.append((case, target_coords))
        else:
            self.prev_case_with_mine_clicked = case

        return (*target_coords, flag_tile)

    def scanGridAndGetBestMove(self):
        frontier_tiles = self.getFrontierTiles()

        if frontier_tiles:
            tiles_to_choose_from = frontier_tiles
        else:
            tiles_to_choose_from = self.getAllCoveredNonFlaggedTiles()

        # Save for highlighting later
        self.tiles_chosen_from = tiles_to_choose_from

        if self.case_base:
            (
                case,
                target_coords,
            ) = self.pickMostConfidentMoveFromTilesGetItsCaseWithSolution(
                tiles_to_choose_from
            )
        else:
            # No cases learned yet. Can't do any better than random.
            case, target_coords = self.pickRandomMoveFromTilesAndGetItsCaseWithSolution(
                tiles_to_choose_from
            )

        return (case, target_coords)

    def getAllCoveredNonFlaggedTiles(self):
        covered_non_flagged_tiles = []

        for row in self.grid:
            for tile in row:
                if not tile.uncovered and not tile.is_flagged:
                    covered_non_flagged_tiles.append(tile)

        return covered_non_flagged_tiles

    def pickMostConfidentMoveFromTilesGetItsCaseWithSolution(self, tiles):
        most_confident_option = (None, None, None, -1)

        for tile in tiles:
            case = self.convertToCase(tile)
            similar_cases_with_scores = self.retrieveSimilarCases(case)
            (is_mine, confidence) = self.getSolutionAndConfidence(
                case, similar_cases_with_scores
            )
            # Certain about move; quit searching the grid and make the move.
            if confidence == 1.0:
                most_confident_option = (case, tile, is_mine, confidence)
                break

            # Keep track of which case on grid seems to have the most definite outcome
            if confidence > most_confident_option[3]:
                most_confident_option = (case, tile, is_mine, confidence)

        case, target_tile, is_mine, _ = most_confident_option

        case.solution = is_mine
        coords = (target_tile.x, target_tile.y)

        return case, coords

    def convertToCase(self, tile):
        problem = self.getSampleAreaAroundTile(tile)
        solution = None
        return Case(problem, solution)

    def pickRandomMoveFromTilesAndGetItsCaseWithSolution(self, tiles):
        tile = random.choice(tiles)

        case = self.convertToCase(tile)
        case.solution = False  # Guess no mine at tile, so click it.

        coords = (tile.x, tile.y)

        return (case, coords)

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

        if self.prev_case_with_mine_clicked:
            choice_was_correct = self.game_state in [_Game.State.PLAY, _Game.State.WIN]
            self.reviseCase(self.prev_case_with_mine_clicked, choice_was_correct)

            self.prev_case_with_mine_clicked = None

    def reviseCase(self, case, choice_was_correct):
        # Binary decision. Just flip it and you've now got the right solution.
        if not choice_was_correct:
            case.solution = not case.solution

        self.retainCaseIfUseful(case)

    def retainCaseIfUseful(self, case):
        # Retain all unique cases for now (DEFINITELY NOT THE WAY TO DO THIS IN THE REAL THING)
        if case not in self.case_base:
            shouldRetain = True

        if shouldRetain:
            self.case_base.append(case)

    def onGameReset(self):
        # Evaluate flag choices and learn from them if useful.
        for case, (x, y) in self.cases_with_flag_and_target_coords:
            choice_was_correct = case.solution == self.grid[x][y].is_mine
            self.reviseCase(case, choice_was_correct)

        self.cases_with_flag_and_target_coords = []

    """
        Returns a two-dimensional list that represents the 5x5 grid sample centered on the input tile.
        
        Each element of the grid (sample[i][j]) is a single-character string value representing the state of that tile.
        The string value can take on one of the following options:
            > a digit in range [0, 8] (number of adjacent mines around),
            > - (tile is covered), 
            > F (flagged), 
            > W (wall).
    """

    def getSampleAreaAroundTile(self, tile):
        x_offsets = range(
            -self.SAMPLE_COLUMNS_MID, (self.SAMPLE_COLUMNS - self.SAMPLE_COLUMNS_MID)
        )
        y_offsets = range(
            -self.SAMPLE_ROWS_MID, (self.SAMPLE_ROWS - self.SAMPLE_ROWS_MID)
        )
        num_rows = len(self.grid)
        num_columns = len(self.grid[0])

        sample = []

        for y_offset in y_offsets:
            new_y = tile.y + y_offset
            column = []

            # Out of bounds vertically. All tiles in rows are a wall.
            if new_y < 0 or new_y >= num_rows:
                column = ["W"] * self.SAMPLE_ROWS
                sample.append(column)
                continue

            for x_offset in x_offsets:
                new_x = tile.x + x_offset

                # Out of bounds horizontally. Tile is a wall
                if new_x < 0 or new_x >= num_columns:
                    column.append("W")
                    continue

                new_tile = self.grid[new_y][new_x]

                if new_tile.uncovered:
                    tile_representation = str(new_tile.num_adjacent_mines)
                    column.append(tile_representation)
                elif new_tile.is_flagged:
                    column.append("F")
                else:
                    column.append("-")

            sample.append(column)

        return sample

    # Returns a list of all 'frontier tiles' - the covered tiles that are on the border between covered-uncovered tiles.
    def getFrontierTiles(self):
        frontier_tiles = []

        for row in self.grid:
            for tile in row:
                if self.isFrontierTile(tile):
                    frontier_tiles.append(tile)

        return frontier_tiles

    # Frontier tiles are covered non-flagged tiles which also have atleast one adjacent uncovered tile
    def isFrontierTile(self, tile):
        if tile.uncovered or tile.is_flagged:
            return False

        num_rows = len(self.grid)
        num_columns = len(self.grid[0])

        for x_offset in [-1, 0, 1]:
            new_x = tile.x + x_offset

            # Out of bounds.
            if new_x < 0 or new_x >= num_columns:
                continue

            for y_offset in [-1, 0, 1]:
                new_y = tile.y + y_offset

                # Out of bounds.
                if new_y < 0 or new_y >= num_rows:
                    continue

                if self.grid[new_y][new_x].uncovered:
                    return True

        return False

    # Returns a list of all cases that are ranked to be most similar using K-means clustering.
    # ^^ would be the better version. This just naively ranks similarities and returns the 2 most
    # 'similar' cases.
    def retrieveSimilarCases(self, case):
        cases_and_similarity_scores = []

        for known_case in self.case_base:
            similarity_score = self.calculateSimilarity(case, known_case)
            cases_and_similarity_scores.append((known_case, similarity_score))

        # Sort by similarity scores
        cases_and_similarity_scores.sort(key=lambda x: x[1], reverse=True)

        return cases_and_similarity_scores[:2]

    # Returns already used solution if an exact case match is found. Otherwise a solution is adapted from the similar cases. Confidence score 0.0 - 1.0 too.
    # Using cases' similarity score as the measure of confidence in a solution.
    # Input is the problem case, and a sorted list of tuples (similar_case, similarity_score) sorted by similarity score in ascending order.
    def getSolutionAndConfidence(self, case, similar_cases_with_score):
        (most_similar_case, score) = similar_cases_with_score[0]
        return most_similar_case.solution, score

    # # Returns a score between 0.0 to 1.0 rating how similar the two cases are.
    # # Using very naive method for now: using proportion of tiles that are the same in both cases.
    # # Assumes both cases have same case problem structure. Does not take into account symmetries.
    # def calculateSimilarity(self, case_1, case_2):
    #     similar_tiles = 0

    #     for y in range(len(case_1.problem)):
    #         for x in range(len(case_1.problem[0])):
    #             if case_1.problem[y][x] == case_2.problem[y][x]:
    #                 similar_tiles += 1

    #     num_tiles = len(case_1.problem) * len(case_1.problem[0])

    #     return similar_tiles / num_tiles

    def is_case_of_pattern(self, case, pattern):
        """Checks if case matches pattern (i.e., pattern is a subset of the case).
        Note that tiles aren't matched exactly value-for-value as pattern cases can
        have generalised tile values."""
        similar_tiles = 0

        pattern_rotations = get_case_rotations(known_case)

        # Attempt a match of pattern case against every rotated version of it.
        for pattern_case in pattern_rotations:
            pattern_does_not_match = False

            for (y, row) in enumerate(pattern_case):
                for (x, pattern_tile) in enumerate(row):
                    case_tile = case[y][x]

                    if is_tile_different_from_pattern_tile(case_tile, pattern_tile):
                        pattern_does_not_match = True

                if pattern_does_not_match:
                    break

            if pattern_does_not_match:
                continue
            else:
                return True  # Pattern match found!

        return False

        for y in range(len(case_1.problem)):
            for x in range(len(case_1.problem[0])):
                if case_1.problem[y][x] == case_2.problem[y][x]:
                    similar_tiles += 1

        num_tiles = len(case_1.problem) * len(case_1.problem[0])

        return similar_tiles / num_tiles

    def highlightTiles(self):
        tiles_to_highlight = []

        for tile in self.tiles_chosen_from:
            highlight_code = 1
            tiles_to_highlight.append((tile.x, tile.y, highlight_code))

        self.tiles_chosen_from = []

        return tiles_to_highlight

    def onGameBegin(self, game_seed):
        pass

    def feedRenderer(self, renderer):
        self.renderer = renderer
