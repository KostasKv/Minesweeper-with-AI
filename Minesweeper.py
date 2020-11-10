from Game import Game, Cell
from PygameRenderer import PygameRenderer
from NoScreenRenderer import NoScreenRenderer
from ExampleAgents import RandomAgent, RandomLegalMovesAgent
from CBRAgent import CBRAgent


def getSampleAreaAroundCell(grid, cell):
        SAMPLE_ROWS = 5
        SAMPLE_COLUMNS = 5
        SAMPLE_ROWS_MID = (SAMPLE_ROWS - 1) // 2
        SAMPLE_COLUMNS_MID = (SAMPLE_COLUMNS - 1) // 2
        x_offsets = range(-SAMPLE_COLUMNS_MID, (SAMPLE_COLUMNS - SAMPLE_COLUMNS_MID))
        y_offsets = range(-SAMPLE_ROWS_MID, (SAMPLE_ROWS - SAMPLE_ROWS_MID))
        num_rows = len(grid[0])
        num_columns = len(grid)

        print("x-offsets: {}".format(list(x_offsets)))
        print("y-offsets: {}".format(list(y_offsets)))
        print("getting sample around cell at ({}, {})".format(cell.x, cell.y))
        sample = []

        for y_offset in y_offsets:
            new_y = cell.y + y_offset
            column = []

            # Out of bounds vertically. All cells in rows are a wall.
            if (new_y < 0 or new_y >= num_rows):
                column = ['W'] * SAMPLE_ROWS
                sample.append(column)
                continue
            
            for x_offset in x_offsets:
                new_x = cell.x + x_offset
                
                # Out of bounds horizontally. Cell is a wall
                if (new_x < 0 or new_x >= num_columns):
                    column.append('W')
                    continue

                new_cell = grid[new_y][new_x]
                print("x_offset: {}, y_offset: {}, new_cell: ({}, {})".format(x_offset, y_offset, new_cell.x, new_cell.y))
                
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
        print(sample)
        
        return sample           



class Executor():
    def __init__(self, game, num_games):
        self.game = game
        self.games_left = num_games


    def getGameConfig(self):
        return self.game.config


    def getGridAndMinesLeftAndState(self):
        return (self.game.grid, self.game.mines_left, self.game.state)


    ''' Input: action as a tuple (x, y, toggle_flag), where x, y is the 0-based coordinates
        of the tile to interact with, and toggle_flag is a boolean indiciating whether to toggle flag
        status of the cell or to click the cell.
        An action that is an integer -1 indicates the user chose to force reset the game.

        If game is in a finished state, then a call to this method will trigger a reset of the game, and
        return the appropriate grid and state. In this case the action is ignored.
        
        Returns: (grid, mines_left, game_state) as they are after the move has been made,
                 or returns None if all games have been finished, regardless of input.'''
    def makeMove(self, action):
        if self.games_left <= 0:
            return None
        
        in_end_game_state = self.game.state in [Game.State.WIN, Game.State.LOSE, Game.State.ILLEGAL_MOVE]
        force_reset = (action == -1)

        if in_end_game_state or force_reset:
            self.game.reset()
            self.games_left -= 1

            if self.games_left <= 0:
                # Final game just finished. Return None to indicate this.
                return None
        elif self.isLegalMove(action):
            self.processMove(action)
        else:
            self.game.state = Game.State.ILLEGAL_MOVE

        return (self.game.grid, self.game.mines_left, self.game.state)


    def isLegalMove(self, action):
        (x, y, toggle_flag) = action

        out_of_bounds = (x < 0) or (y < 0) or (x >= self.game.config['rows']) or (y >= self.game.config['columns'])
        
        if out_of_bounds or self.game.grid[y][x].uncovered:
            return False
        
        if not toggle_flag and self.game.grid[y][x].is_flagged:
            return False
        
        return True
    
    
    # Assumes input is a legal move. Changes game grid and game state based on action.
    def processMove(self, action):
        (x, y, toggle_flag) = action

        if toggle_flag:
            self.toggleFlag(x, y)
            is_end_of_game = False  
        else:
            is_end_of_game = self.clickMine(x, y)
        
        # Toggle flag should not change game state from START to PLAY.
        # If it's end of game, state should not be overwritten back to PLAY.
        if not is_end_of_game and not toggle_flag:
            self.game.state = Game.State.PLAY
    

    # Changes game grid (one cell gets flagged/unflagged).
    def toggleFlag(self, x, y):
        if self.game.grid[y][x].is_flagged:
            self.game.grid[y][x].is_flagged = False
            self.game.mines_left += 1
        else:
            self.game.grid[y][x].is_flagged = True
            self.game.mines_left -= 1


    # Returns boolean indicating whether game has been finished or not.
    # Method changes game's grid and state.
    def clickMine(self, x, y):
        end_of_game = False

        if self.game.grid[y][x].is_mine:
            self.game.grid[y][x].uncovered = True
            self.game.state = Game.State.LOSE
            end_of_game = True
        else:
            self.uncover(x, y)
            
            if self.gameWon():
                self.game.state = Game.State.WIN
                end_of_game = True
    
        return end_of_game


    def uncover(self, initial_x, initial_y):
        cells_to_uncover = [(initial_x, initial_y)]
        
        while cells_to_uncover:
            x, y = cells_to_uncover.pop(0)
            cell = self.game.grid[y][x]

            cell.uncovered = True

            if cell.num_adjacent_mines == 0:
                adjacent_cells_coords = self.game.getAdjacentCellsCoords(x, y)

                for x, y in adjacent_cells_coords:
                    if self.game.grid[y][x].uncovered == False and not self.game.grid[y][x].is_flagged:
                        if not (x, y) in cells_to_uncover:
                            cells_to_uncover.append((x, y))
    

    def gameWon(self):
        for column in self.game.grid:
            for cell in column:
                # If any non-mine cell is still covered, then game has not yet been won
                if not cell.is_mine and not cell.uncovered:
                    return False

        return True

    
    def forceResetGame(self):
        self.game.reset()
        self.games_left -= 1

        if self.games_left <= 0:
            return None
        else:
            return (self.game.grid, self.game.mines_left, self.game.state)

            
def playGames(executor, renderer, verbose):
    # Start of all games. Start renderer and get agent's very first move.
    action = renderer.getNextMoveFromAgent()

    # Play until all games are finished
    while result := executor.makeMove(action):
        if verbose:
            print("Made move {}.\tResult: {} mines left, game state {}".format(action, result[1], result[2]))
        
        # DEBUG SAMPLE THING
        x, y, _ = action
        grid = result[0]
        renderer.sample = getSampleAreaAroundCell(grid, grid[y][x])

        renderer.updateFromResult(result)
        action = renderer.getNextMoveFromAgent()

    renderer.onEndOfGames()


def run(agent=None, config={'rows':16, 'columns':16, 'num_mines':8}, num_games=500, visualise=True, verbose=1):
    # If user is going to manually play, then they have to see the board.
    if not agent:
        visualise = True

    game = Game(config)
    executor = Executor(game, num_games)

    if visualise:
        renderer = PygameRenderer(config, game.grid, agent)
    else:
        renderer = NoScreenRenderer(config, game.grid, agent)

    playGames(executor, renderer, verbose)


def gridToSample(grid):
    num_rows = len(grid)
    num_columns = len(grid[0])
    SAMPLE_ROWS = num_rows
    SAMPLE_COLUMNS = num_columns
    SAMPLE_ROWS_MID = (SAMPLE_ROWS - 1) // 2
    SAMPLE_COLUMNS_MID = (SAMPLE_COLUMNS - 1) // 2
    x_offsets = range(-SAMPLE_COLUMNS_MID, (SAMPLE_COLUMNS - SAMPLE_COLUMNS_MID))
    y_offsets = range(-SAMPLE_ROWS_MID, (SAMPLE_ROWS - SAMPLE_ROWS_MID))

    cell = grid[SAMPLE_ROWS_MID][SAMPLE_COLUMNS_MID]

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

            new_cell = grid[new_x][new_y]
            
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

def testSampleStuff():
    # Initialise game n controller
    config={'rows':8, 'columns':8, 'num_mines':10}
    game = Game(config)
    executor = Executor(game, num_games=1)
    renderer = PygameRenderer(config, game.grid, None)
    
    # Initialise agent
    cbr_agent = CBRAgent()
    cbr_agent.update(game.grid, game.mines_left, game.state)
    
    # Make a move
    executor.makeMove((3, 3, False))

    # Show full grid first
    grid_sample = gridToSample(game.grid)
    renderer.visualiseSample(grid_sample)

    # Get some samples and show them on screen for checking
    some_cell = game.grid[3][3]
    sample = cbr_agent.getSampleAreaAroundCell(some_cell)
    renderer.visualiseSample(sample)

    # Make a move
    executor.makeMove((0, 0, False))

    # Show full grid first
    grid_sample = gridToSample(game.grid)
    renderer.visualiseSample(grid_sample)

    # Get some samples and show them on screen for checking
    some_cell = game.grid[0][0]
    sample = cbr_agent.getSampleAreaAroundCell(some_cell)
    renderer.visualiseSample(sample)

    # corner_cell = game.grid[0][0]
    # corner_sample = cbr_agent.getSampleAreaAroundCell(corner_cell)
    # visualiseSample(corner_sample)



if __name__ == '__main__':
    # random_agent = RandomAgent()
    # random_legal_agent = RandomLegalMovesAgent()
    # # run(random_agent)
    # run(random_legal_agent, visualise=False, verbose=2)
    run()
    # testSampleStuff()
