from Agent import Agent

class Case:
    '''
        Case problem is a 5x5 grid, each cell takes on one of 12 string values: 0-8(num adjacent mines), -(covered), F (flag), W(wall)
        Solution is a boolean value; True means center cell is a mine, False means it's not. 
    '''
    def __init__(self, problem_description, solution):
        self.problem_description = problem_description
        self.solution = solution


class CBRAgent(Agent):
    def nextMove(self):
        pass

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

    def onGameReset(self):
        pass
    

    '''
        Returns a two-dimensional list that represents the 5x5 grid sample centered on the input cell.
        
        Each element of the grid (sample[i][j]) is a single-character string value representing the state of that cell.
        The string value can take on one of the following options:
            > a digit in range [0, 8] (number of adjacent mines around),
            > - (cell is covered), 
            > F (flagged), 
            > W (wall).

    '''
    def getSampleAreaAroundCell(self, cell):
        SAMPLE_ROWS = 5
        SAMPLE_COLUMNS = 5
        SAMPLE_ROWS_MID = (SAMPLE_ROWS - 1) // 2
        SAMPLE_COLUMNS_MID = (SAMPLE_COLUMNS - 1) // 2
        x_offsets = range(-SAMPLE_COLUMNS_MID, (SAMPLE_COLUMNS - SAMPLE_COLUMNS_MID))
        y_offsets = range(-SAMPLE_ROWS_MID, (SAMPLE_ROWS - SAMPLE_ROWS_MID))
        num_rows = len(self.grid[0])
        num_columns = len(self.grid)

        sample = []

        for x_offset in x_offsets:
            new_x = cell.x + x_offset
            column = []

            # Out of bounds horizontally. All cells in column are a wall.
            if (new_x < 0 or new_x >= num_columns):
                column = ['W'] * SAMPLE_ROWS
                sample.append(column)
                continue
            
            for y_offset in y_offsets:
                new_y = cell.y + y_offset

                # Out of bounds vertically. Cell is a wall
                if (new_y < 0 or new_y >= num_rows):
                    column.append('W')
                    sample.append(column)
                    continue

                new_cell = self.grid[new_x][new_y]
                
                if new_cell.uncovered:
                    # print("cell at ({}, {}) is uncovered! num adjacent mines: {}".format(new_cell.x, new_cell.y, new_cell.num_adjacent_mines))
                    cell_representation = str(new_cell.num_adjacent_mines)
                    column.append(cell_representation)
                elif new_cell.is_flagged:
                    # print("cell at ({}, {}) is flagged!".format(new_cell.x, new_cell.y))
                    column.append('F')
                else:
                    # print("cell at ({}, {}) is still covered".format(new_cell.x, new_cell.y))
                    column.append('-')

            # print("x_offset: {}, column: {}".format(x_offset, column))
            sample.append(column)
        # print(sample)
        
        return sample

    
    # Returns a list of all 'frontier cells' - the covered cells that are on the border between covered-uncovered cells.
    def getFrontierCells(self, ): 
        pass
    
    # Returns a list of all cases that are ranked to be most similar using K-means clustering.
    def retrieveSimilarCases(self): 
        pass
    
    # Returns already used solution if an exact case match is found. Otherwise a solution is adapted from the similar cases. Confidence score 0.0 - 1.0 too.
    def getSolutionAndConfidence(self): 
        pass
    
    # Decides whether or not to retain cases. Might adapt them (generalise) before adding to case base.
    def evaluate(self, cases): 
        pass

# prev_case = None
# flagged_cases_with_coords = []

# # Must return (x, y, is_flag)
# def next(self):
#     frontier_cells = getFrontierCells(grid)

#     most_likely = (None, None, 0)

#     for cell in frontier_cells:
#         case = convertToCase(cell)
#         similar_cases = retrieveSimilarCases()
#         (is_mine, confidence) = getSolutionAndConfidence(case, similar_cases)

#         if confidence == 1.0:
#             most_likely = (case, confidence)
#             break
        
#         most_likely = max(most_likely.confidence, confidence)
    
#     most_likely.case.solution = flag_the_cell
#     prev_case = most_likely.case
#     cell = most_likely.case.center_cell
#     return (cell.x, cell.y, flag_the_cell)


# def result(self, grid, game_state):
#     if game_state = GameState.WON:
#         gameWon(grid)
#     elif game_state = GameState.LOST:
#         gameLost(grid)
#     else:
#         moveMade(grid)


# def gameLost(self, grid):
#     cases_to_evaluate = []

#     # Learn from previous move
#     prev_case.is_mine = True

#     # Learn flagged cases
#     for (case, coords) in flagged_cases_with_coords:
#         x, y = coords
#         case.is_mine = grid[x][y].is_mine

#     evaluate(cases_to_evaluate)


# def gameWon(self, grid):
#     evaluate(prev_case)

# def moveMade(self, grid):
    

