import time
import cProfile
import random

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.random_agent import RandomAgent
from minesweeper_ai.agents.random_legal_moves_agent import RandomLegalMovesAgent
from minesweeper_ai.agents.pick_first_uncovered_agent import PickFirstUncoveredAgent
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver
from minesweeper_ai.agents.linear_equations_solver import LinearEquationsSolver
from minesweeper_ai.agents.cbr_agent1 import CBRAgent1
from minesweeper_ai.agents.cbr_agent2 import CBRAgent2
from bitarray import bitarray

def main():
    main_play()

def main_play():
    game_seeds = None
    human_player = False
    profile = False
    benchmark = False
    num_games_profile = 100
    num_games_benchmark = 10
    num_games_other = 10
    config = {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': True}
    run_seed = 40   # Same run seed as main experiment
    agent_seed = 4040   # Same agent seed as main experiment
    sample_size = None

    # Agents selection
    random_agent = RandomAgent()
    random_legal_agent = RandomLegalMovesAgent()
    pick_first_uncovered_agent = PickFirstUncoveredAgent()
    main_solver = NoUnnecessaryGuessSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=True, can_flag=False)
    linear_solver_agent = LinearEquationsSolver(seed=agent_seed, sample_size=sample_size, use_num_mines_constraint=False)
    cbr_agent_1 = CBRAgent1()
    cbr_agent_2 = CBRAgent2()

    # Chosen agent
    agent = main_solver

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
                    result = minesweeper.run(agent, config=config, visualise=False, verbose=False, num_games=num_games_benchmark, seed=run_seed)
                    end = time.time()
                    results.append(result)
                    print("Time taken: {}s\n".format(end - start))
    if not benchmark and not profile:
        if human_player:
            agent = None
    
        results = minesweeper.run(agent, config=config, visualise=True, verbose=False, num_games=num_games_other, seed=run_seed, game_seeds=game_seeds)

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
    