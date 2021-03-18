import time
import cProfile

from ._game import _Game
from._executor import _Executor
from ._pygame_renderer import PygameRenderer
from ._no_screen_renderer import NoScreenRenderer

# For use in method 'update_stats_from_action_result'. Just need this to persist between function calls so we can time game durations.
game_start_time = None


def run(agent=None, config={'rows':8, 'columns':8, 'num_mines':10, 'first_click_is_zero':True}, num_games=10, visualise=True, verbose=1, seed=None, game_seeds=None):
    # If user is going to manually play, then they have to see the board.
    if not agent:
        visualise = True

    game = _Game(config)

    if game_seeds:
        executor = _Executor(game, game_seeds=game_seeds)
    else:
        executor = _Executor(game, num_games, seed=seed)

    if visualise:
        renderer = PygameRenderer(config, game.grid, agent, executor.current_game_seed)
    else:
        renderer = NoScreenRenderer(config, game.grid, agent, executor.current_game_seed)

    return playGames(executor, renderer, verbose)

def playGames(executor, renderer, verbose):
    ''' Program's main loop '''
    global game_start_time

    stats = initialise_stats()
    
    start_time = time.time()
    game_start_time = time.time()

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

        stats = update_stats_from_action_result(stats, result, renderer, executor)

        renderer.updateFromResult(result, executor.current_game_seed)
        action = renderer.getNextMove()
        result = executor.make_move(action)

    end_time = time.time()
    stats['time_elapsed'] = end_time - start_time

    # print("wins: {} ({}%)".format(wins, round((wins / num_games) * 100, 2)))

    agent_stats = renderer.onEndOfGames()

    if agent_stats:
        stats = {**stats, **agent_stats}
    
    return stats

def initialise_stats():
    return {
        'wins': 0,
        'wins_without_guess': 0,
        'games': [],
    }

def update_stats_from_action_result(stats, result, renderer, executor):
    global game_start_time
    (grid, _, state) = result

    if state == _Game.State.START:
        game_start_time = time.time()

    if _Game.is_end_of_game_state(state):
        game_end_time = time.time()

        game_stats = {
            'seed': executor.current_game_seed,
            'grid_mines': grid,
            'win': state == _Game.State.WIN,
            'seconds_elapsed': game_end_time - game_start_time,
            'sample_width': renderer.agent.SAMPLE_SIZE[0],
            'sample_height': renderer.agent.SAMPLE_SIZE[1],
            'use_mine_count': renderer.agent.use_num_mines_constraint,
            'first_click_always_zero': executor.get_game_config()['first_click_is_zero'],
            'num_guesses': renderer.agent.num_guesses_for_game,
            'first_click_pos_x': renderer.agent.first_click_pos_this_game[0],
            'first_click_pos_y': renderer.agent.first_click_pos_this_game[1],
            'turns': renderer.agent.get_game_turns_stats(),
        }

        stats['games'].append(game_stats)
        
        if state == _Game.State.WIN:
            stats['wins'] += 1
            if game_stats['num_guesses'] == 0:
                stats['wins_without_guess'] += 1

    return stats


def create_game_seeds(num_games, run_seed):
    ''' Creates a batch of game seeds from a given game configuration and run seed.
        This method exposes the internal method for creating game seeds. '''
    return _Executor.create_game_seeds(num_games, run_seed)
