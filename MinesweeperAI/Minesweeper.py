from Game import Game, Tile
from PygameRenderer import PygameRenderer
from NoScreenRenderer import NoScreenRenderer
from ExampleAgents import RandomAgent, RandomLegalMovesAgent, PickFirstUncoveredAgent
from CBRAgent1 import CBRAgent1
from NoUnnecessaryGuessSolver import NoUnnecessaryGuessSolver
import time
import itertools
import cProfile


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
        status of the tile or to click the tile.
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
            if self.game.state == Game.State.START and not action[2]:
                safe_tile = (action[0], action[1])
                self.game.populateGrid(safe_tile)
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
    

    # Changes game grid (one tile gets flagged/unflagged).
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
        tiles_to_uncover = [(initial_x, initial_y)]
        
        while tiles_to_uncover:
            x, y = tiles_to_uncover.pop(0)
            tile = self.game.grid[y][x]

            tile.uncovered = True

            if tile.num_adjacent_mines == 0:
                adjacent_tiles_coords = self.game.getAdjacentTilesCoords(x, y)

                for x, y in adjacent_tiles_coords:
                    if self.game.grid[y][x].uncovered == False and not self.game.grid[y][x].is_flagged:
                        if not (x, y) in tiles_to_uncover:
                            tiles_to_uncover.append((x, y))
    

    def gameWon(self):
        for column in self.game.grid:
            for tile in column:
                # If any non-mine tile is still covered, then game has not yet been won
                if not tile.is_mine and not tile.uncovered:
                    return False

        return True


    def forceResetGame(self):
        self.game.reset()
        self.games_left -= 1

        if self.games_left <= 0:
            return None
        else:
            return (self.game.grid, self.game.mines_left, self.game.state)

            
def playGames(executor, renderer, verbose, num_games):
    stats = {'wins': 0,
             'num_games': num_games}
    
    start_time = time.time()

    # First move of all games.
    action = renderer.getNextMove()
    result = executor.makeMove(action)


    # Play until all games are finished
    while result:
        # Temp verbose solution. It should give more information, and the responsibility should be put on the renderer
        # to display the information (however the information could be processed outside of it and fed to the renderer.
        # Not yet sure which is preferable)
        if verbose:
            print("Made move {}.\tResult: {} mines left, game state {}".format(action, result[1], result[2]))

        if result[2] == Game.State.WIN:
            stats['wins'] += 1


        renderer.updateFromResult(result)
        action = renderer.getNextMove()
        result = executor.makeMove(action)

    end_time = time.time()
    stats['time_elapsed'] = end_time - start_time

    # print("wins: {} ({}%)".format(wins, round((wins / num_games) * 100, 2)))

    agent_stats = renderer.onEndOfGames()

    if agent_stats:
        stats = {**stats, **agent_stats}
    
    return stats


def run(agent=None, config={'rows':8, 'columns':8, 'num_mines':10}, num_games=10, visualise=True, verbose=1, seed=None):
    # If user is going to manually play, then they have to see the board.
    if not agent:
        visualise = True

    game = Game(config, seed)
    executor = Executor(game, num_games)

    if visualise:
        renderer = PygameRenderer(config, game.grid, agent)
    else:
        renderer = NoScreenRenderer(config, game.grid, agent)

    return playGames(executor, renderer, verbose, num_games)


if __name__ == '__main__':
    # Constants (configurables)
    profile = False
    benchmark = False
    
    num_games_profile = 10
    num_games_benchmark = 10
    num_games_other = 2
    config = {'rows': 16, 'columns': 30, 'num_mines': 99}
    run_seed = 57
    agent_seed = 14
    
    # Solvers
    random_agent = RandomAgent()
    random_legal_agent = RandomLegalMovesAgent()
    pick_first_uncovered_agent = PickFirstUncoveredAgent()
    solver_agent = NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=(10, 10), use_num_mines_constraint=False)

    # Learners
    cbr_agent_1 = CBRAgent1()


    if profile:
        print("Profiling. Running {} games...".format(num_games_profile))
        # cProfile.run("run(solver_agent, config=config, visualise=False, verbose=False, num_games=num_games_profile, seed=run_seed)", "solver.prof")
        cProfile.run("run(NoUnnecessaryGuessSolver(seed=agent_seed, use_num_mines_constraint=False), config=config, visualise=False, verbose=False, num_games=num_games_profile, seed=run_seed)", "solver1.prof")
        cProfile.run("run(NoUnnecessaryGuessSolver(seed=agent_seed, use_num_mines_constraint=True), config=config, visualise=False, verbose=False, num_games=num_games_profile, seed=run_seed)", "solver2.prof")
    if benchmark:
        # sample_sizes = [(32, 18)]
        # u = [True]
        sample_sizes = [(4, 4), (5, 5), (6, 6), (32, 18)]
        u = [True, False]
        configs = [{'rows': 9, 'columns': 9, 'num_mines': 10}, {'rows': 16, 'columns': 16, 'num_mines': 40}, {'rows': 16, 'columns': 30, 'num_mines': 99}]
        combinations = len(sample_sizes) * len(u) * len(configs)
        print("Benchmarking. Running {} games total, spread over {} different configurations...".format(num_games_benchmark * combinations, combinations))
        
        results = []

        for sample_size in sample_sizes:
            for use_num_mines_constraint in u:
                for config in configs:
                    print("Sample size: {}\tuse_num_mines_constraint: {}, config: {}".format(sample_size, use_num_mines_constraint, config))
                    solver_agent = NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=use_num_mines_constraint)
                    start = time.time()
                    result = run(solver_agent, config=config, visualise=False, verbose=False, num_games=num_games_benchmark, seed=run_seed)
                    end = time.time()
                    results.append(result)
                    print("Time taken: {}s\n".format(end - start))
    if not benchmark and not profile:
        # run(verbose=0, config=config, seed=57, visualise=True)
        # run(random_agent, config=config, verbose=True, visualise=True)
        # run(random_legal_agent, config=config, visualise=True, verbose=False, num_games=50, seed=57)
        results = run(solver_agent, config=config, visualise=False, verbose=False, num_games=num_games_other, seed=run_seed)
        # run(cbr_agent_1, visualise=True, verbose=True, num_games=10)

    print(results)
    print("Program stopped.")
