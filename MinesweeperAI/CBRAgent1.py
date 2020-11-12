from Agent import Agent
from Game import Game
from random import randint

class Case:
    '''
        Case problem is a 5x5 grid, each tile takes on one of 12 string values: 0-8(num adjacent mines), -(covered), F (flag), W(wall)
        Solution is a boolean value; True means middle tile is a mine, False means it's not. 
    '''
    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution


class CBRAgent1(Agent):
    def __init__(self):
        self.case_base = []
        self.prev_case_with_mine_clicked = None
        self.cases_with_flag = []
        
        # Constants
        self.SAMPLE_ROWS = 5
        self.SAMPLE_COLUMNS = 5
        self.SAMPLE_ROWS_MID = (self.SAMPLE_ROWS - 1) // 2
        self.SAMPLE_COLUMNS_MID = (self.SAMPLE_COLUMNS - 1) // 2


    def nextMove(self):
        # First click of a game is random
        if self.game_state == Game.State.START:
            x, y = self.pickRandomCoordinates()
            action = (x, y, False)
        else:
            action = getNextMoveBasedOnPreviousCases()

        return action


    def pickRandomCoordinates(self):
        x = randint(0, len(self.grid[0]))
        y = randint(0, len(self.grid))
        return (x, y)


    def getNextMoveBasedOnPreviousCases(self):
        case, target_coords = self.scanGridForMostConfidentMoveAndGetItsSolution()
        
        flag_tile = case.solution

        # Remember case until its outcome is found out. For flagged cases, that's at the end
        # of a game. For mine clicks, that's immediately after the move has been made.
        if flag_tile:
            self.cases_with_flag_and_target_tiles.append((case, target_coords))
        else:
            self.prev_case_with_mine_clicked = case

        return (*target_coords, flag_tile)


    def scanGridForMostConfidentMoveAndGetItsSolution(self):
        frontier_tiles = self.getFrontierTiles()

        most_confident_option = (None, None, None, -1)

        for tile in frontier_tiles:
            case = convertToCase(tile)
            similar_cases = self.retrieveSimilarCases(case)
            (is_mine, confidence) = getSolutionAndConfidence(case, similar_cases)
            print("mine: {}, confidence: {} ".format(is_mine, confidence))
            # Certain about move; quit searching the grid and make the move.
            if confidence == 1.0:
                most_confident_option = (case, tile, is_mine, confidence)
                break
            
            # Keep track of which case on grid seems to have the most definite outcome
            if confidence > most_confident_option.confidence:
                most_confident_option = (case, tile, is_mine, confidence)

        case, is_mine, target_tile, _ = most_confident_option
        
        case.solution = is_mine

        return case, target_tile


    def convertToCase(self, tile):
        problem = self.getSampleAreaAroundTile(tile)
        solution = None
        return Case(problem, solution)


    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

        if self.prev_case_with_mine_clicked:
            choice_was_correct = self.game_state in [Game.State.PLAYING, Game.State.WIN]
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
            choice_was_correct = (case.solution == self.grid[x][y].is_mine)
            self.reviseCase(case, choice_was_correct)

        self.cases_with_flag_and_target_coords = []


    '''
        Returns a two-dimensional list that represents the 5x5 grid sample centered on the input tile.
        
        Each element of the grid (sample[i][j]) is a single-character string value representing the state of that tile.
        The string value can take on one of the following options:
            > a digit in range [0, 8] (number of adjacent mines around),
            > - (tile is covered), 
            > F (flagged), 
            > W (wall).
    '''
    def getSampleAreaAroundTile(self, tile):
        x_offsets = range(-SAMPLE_COLUMNS_MID, (SAMPLE_COLUMNS - SAMPLE_COLUMNS_MID))
        y_offsets = range(-SAMPLE_ROWS_MID, (SAMPLE_ROWS - SAMPLE_ROWS_MID))
        num_rows = len(self.grid[0])
        num_columns = len(self.grid)

        sample = []

        for y_offset in y_offsets:
            new_y = tile.y + y_offset
            column = []

            # Out of bounds vertically. All tiles in rows are a wall.
            if (new_y < 0 or new_y >= num_rows):
                column = ['W'] * SAMPLE_ROWS
                sample.append(column)
                continue
            
            for x_offset in x_offsets:
                new_x = tile.x + x_offset
                
                # Out of bounds horizontally. Tile is a wall
                if (new_x < 0 or new_x >= num_columns):
                    column.append('W')
                    continue

                new_tile = grid[new_y][new_x]
                
                if new_tile.uncovered:
                    tile_representation = str(new_tile.num_adjacent_mines)
                    column.append(tile_representation)
                elif new_tile.is_flagged:
                    column.append('F')
                else:
                    column.append('-')

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
    
    
    # Frontier tiles are those who have an adjacent uncovered tile
    def isFrontierTile(self, tile):
        num_rows = len(self.grid)
        num_columns = len(self.grid[0])

        uncovered_tile_found = False

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
                
                if self.grid[new_x][new_y].uncovered:
                    return True
                
    
    # Returns a list of all cases that are ranked to be most similar using K-means clustering.
    # ^^ would be the better version. This just naively ranks similarities and returns the 2 most
    # 'similar' cases.
    def retrieveSimilarCases(self, case):
        cases_and_similarity_scores = []

        for known_case in self.case_base:
            similarity_score = self.calculateSimilarity(case, known_case)
            cases_and_similarity_scores.append((known_case, similarity_score))

        # Sort by similarity scores
        cases_and_similarity_scores.sort(key=lambda x: x[1], reversed=True)
        
        return cases_and_similarity_scores[:2]


    # Returns already used solution if an exact case match is found. Otherwise a solution is adapted from the similar cases. Confidence score 0.0 - 1.0 too.
    # Using cases' similarity score as the measure of confidence in a solution.
    def getSolutionAndConfidence(self, case, similar_cases):
        most_similar_case_and_similarity_score = (None, 0)

        for similar_case in similar_cases:
            # Same case, we already know the exact solution.
            if case == similar_case:
                return (similar_case.solution, 1.0)

            similarity_score = self.calculateSimilarity(case, similar_case)

            if similarity_score > most_similar_case_and_similarity_score[1]:
                most_similar_case_and_similarity_score = (similar_case, similarity_score)

        return (most_similar_case_and_similarity_score[0].solution, most_similar_case_and_similarity_score[1])


    # Returns a score between 0.0 to 1.0 rating how similar the two cases are.
    # Using very naive method for now: using proportion of tiles that are the same in both cases.
    # Assumes both cases have same case problem structure. Does not take into account symmetries.
    def calculateSimilarity(self, case_1, case_2):
        similar_tiles = 0

        for y in range(case_1.problem):
            for x in range(case_1.problem[0]):
                if case_1.problem[y][x] == case_2.problem[y][x]:
                    similar_tiles += 1

        num_tiles = len(case_1.problem) * len(case_1.problem[0])

        return similar_tiles / num_tiles
