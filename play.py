import time
import cProfile
import random

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.random_agent import RandomAgent
from minesweeper_ai.agents.random_legal_moves_agent import RandomLegalMovesAgent
from minesweeper_ai.agents.pick_first_uncovered_agent import PickFirstUncoveredAgent
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver
from minesweeper_ai.agents.cbr_agent1 import CBRAgent1
from bitarray import bitarray

def main():
    main_play()

def main_play():
    game_seeds = None

    human_player = True
    # Constants (configurables)
    profile = False
    benchmark = False
    num_games_profile = 100
    num_games_benchmark = 10
    num_games_other = 10
    # config = {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True}
    config = {'rows': 8, 'columns': 8, 'num_mines': 10, 'first_click_is_zero': False}
    # run_seed = 54
    # agent_seed = 13
    run_seed = 40   # Same run seed as main experiment
    agent_seed = 4040   # Same agent seed as main experiment
    # game_seeds = [7083311470311291716, 6969] * 3
    # game_seeds = [-8635908151614172643, 3031145623968078003, 4833213684712801012, 5882193735102876406, -3445616753041698909]
    # game_seeds = [-8635908151614172643, 3031145623968078003, 4833213684712801012, 5882193735102876406, -3445616753041698909, 8588763334019272104, 8777523799321782054, 6959701420644399351, 5958711173367462203, 4731185150046434087]
    game_seeds = [4731185150046434087]
    random.shuffle(game_seeds)
    print(game_seeds)
    sample_size = (5, 5)
    
    # Solvers
    random_agent = RandomAgent()
    random_legal_agent = RandomLegalMovesAgent()
    pick_first_uncovered_agent = PickFirstUncoveredAgent()
    solver_agent = NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=True)

    # Learners
    cbr_agent_1 = CBRAgent1()


    if profile:
        print("Profiling. Running {} games...".format(num_games_profile))
        cProfile.run("minesweeper.run(NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=False), config=config, visualise=False, verbose=False, num_games=num_games_profile, seed=run_seed, game_seeds=game_seeds)", "solver1.prof")
        cProfile.run("minesweeper.run(NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=True), config=config, visualise=False, verbose=False, num_games=num_games_profile, seed=run_seed, game_seeds=game_seeds)", "solver2.prof")
    if benchmark:
        # sample_sizes = [(32, 18)]
        # u = [True]
        # sample_sizes = [(4, 4), (5, 5), (6, 6), (32, 18)]
        # u = [True, False]
        # configs = [{'rows': 9, 'columns': 9, 'num_mines': 10}, {'rows': 16, 'columns': 16, 'num_mines': 40}, {'rows': 16, 'columns': 30, 'num_mines': 99}]
        sample_sizes = [(18, 18)]
        configs = [{'rows': 16, 'columns': 16, 'num_mines': 40}]
        u = [True]
        combinations = len(sample_sizes) * len(u) * len(configs)
        print("Benchmarking. Running {} games total, spread over {} different configurations...".format(num_games_benchmark * combinations, combinations))
        
        results = []

        for sample_size in sample_sizes:
            for use_num_mines_constraint in u:
                for config in configs:
                    print("Sample size: {}\tuse_num_mines_constraint: {}, config: {}".format(sample_size, use_num_mines_constraint, config))
                    solver_agent = NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=use_num_mines_constraint)
                    start = time.time()
                    result = minesweeper.run(solver_agent, config=config, visualise=False, verbose=False, num_games=num_games_benchmark, seed=run_seed)
                    end = time.time()
                    results.append(result)
                    print("Time taken: {}s\n".format(end - start))
    if not benchmark and not profile:
        # minesweeper.run(verbose=0, config=config, seed=57, visualise=True)
        # run(random_agent, config=config, verbose=True, visualise=True)
        # run(random_legal_agent, config=config, visualise=True, verbose=False, num_games=50, seed=57)
        if human_player:
            solver_agent = None
        
        results = minesweeper.run(solver_agent, config=config, visualise=True, verbose=False, num_games=num_games_other, seed=run_seed, game_seeds=game_seeds)
        # run(cbr_agent_1, visualise=True, verbose=True, num_games=10)
        

        all_binary_grid_mines = [
            '0x0020008650008060100000',
            '0x00C4300008140802000000',
            '0x04801180000008008A0080',
            '0x18000002060001C0042000',
            '0x2000000110144002004500',
            '0x00002000020062D0300000',
            '0x2140000020110201100800',
            '0x0024680044800060000000',
            '0x0020000008600C00143000',
            '0x4088280000200000005300',
        ]

        for (i, game_stats) in enumerate(sorted(results['games'], key=lambda x: x['seed'])):
            binary_grid_mines = all_binary_grid_mines[i]
            binary_grid_mines = binary_grid_mines[2:]   # trim 0x prefix
            x = bytes.fromhex(binary_grid_mines)
            
            y = grid_to_binary(game_stats['grid_mines'])

            print(f"({game_stats['first_click_pos_x']}, {game_stats['first_click_pos_y']})\t{game_stats['seed']}" )
            print(f"{x} == {y} : {x == y}\n")
    print("Program stopped.")

def grid_to_binary(grid):
    binary_grid = bitarray()

    for row in grid:
        for tile in row:
            if tile.is_mine:
                binary_grid.append(True)
            else:
                binary_grid.append(False)

    return binary_grid.tobytes()


if __name__ == '__main__':
    main()
    