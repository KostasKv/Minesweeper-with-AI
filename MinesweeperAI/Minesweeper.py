from Game import Game, Cell
from PygameRenderer import PygameRenderer
from NoScreenRenderer import NoScreenRenderer
from ExampleAgents import RandomAgent, RandomLegalMovesAgent
from CBRAgent1 import CBRAgent1


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

        out_of_bounds = (x < 0) or (y < 0) or (x >= self.game.config['columns']) or (y >= self.game.config['rows'])
        
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
        # Temp verbose solution. It should give more information, and the responsibility should be put on the renderer
        # to display the information (however the information could be processed outside of it and fed to the renderer.
        # Not yet sure which is preferable)
        if verbose:
            print("Made move {}.\tResult: {} mines left, game state {}".format(action, result[1], result[2]))

        renderer.updateFromResult(result)
        action = renderer.getNextMoveFromAgent()

    renderer.onEndOfGames()


def run(agent=None, config={'rows':50, 'columns':100, 'num_mines':1000}, num_games=500, visualise=True, verbose=1):
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


if __name__ == '__main__':
    random_agent = RandomAgent()
    random_legal_agent = RandomLegalMovesAgent()
    cbr_agent_1 = CBRAgent1()

    # # run(random_agent)
    # run(random_legal_agent, visualise=False, verbose=2)
    run(verbose=1)
    # run(cbr_agent_1, visualise=True, verbose=True)

