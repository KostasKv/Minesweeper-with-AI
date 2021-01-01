import time
import cProfile

from ._game import _Game
from._executor import _Executor
from ._pygame_renderer import PygameRenderer
from ._no_screen_renderer import NoScreenRenderer

            
def playGames(executor, renderer, verbose):
    ''' Program's main loop '''

    stats = {'wins': 0, 'wins_without_guess': 0}
    
    start_time = time.time()

    # First move of all games.
    action = renderer.getNextMove()
    result = executor.make_move(action)

    # Play until all games are finished
    while result:
        # Temp verbose solution. It should give more information, and the responsibility should be put on the renderer
        # to display the information (however the information could be processed outside of it and fed to the renderer.
        # Not yet sure which is preferable)
        if verbose:
            print("Made move {}.\tResult: {} mines left, game state {}".format(action, result[1], result[2]))

        if result[2] == _Game.State.WIN:
            stats['wins'] += 1

            if not renderer.agent.had_to_guess_this_game:
                stats['wins_without_guess'] += 1

        renderer.updateFromResult(result)
        action = renderer.getNextMove()
        result = executor.make_move(action)

    end_time = time.time()
    stats['time_elapsed'] = end_time - start_time

    # print("wins: {} ({}%)".format(wins, round((wins / num_games) * 100, 2)))

    agent_stats = renderer.onEndOfGames()

    if agent_stats:
        stats = {**stats, **agent_stats}
    
    return stats


def run(agent=None, config={'rows':8, 'columns':8, 'num_mines':10}, num_games=10, visualise=True, verbose=1, seed=None, game_seeds=None):
    # If user is going to manually play, then they have to see the board.
    if not agent:
        visualise = True

    game = _Game(config)

    if game_seeds:
        executor = _Executor(game, game_seeds=game_seeds)
    else:
        executor = _Executor(game, num_games, seed=seed)

    if visualise:
        renderer = PygameRenderer(config, game.grid, agent)
    else:
        renderer = NoScreenRenderer(config, game.grid, agent)

    return playGames(executor, renderer, verbose)
    