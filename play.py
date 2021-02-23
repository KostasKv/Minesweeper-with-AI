import time
import cProfile

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.random_agent import RandomAgent
from minesweeper_ai.agents.random_legal_moves_agent import RandomLegalMovesAgent
from minesweeper_ai.agents.pick_first_uncovered_agent import PickFirstUncoveredAgent
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver
from minesweeper_ai.agents.cbr_agent1 import CBRAgent1

if __name__ == '__main__':
    # Constants (configurables)
    profile = False
    benchmark = False
    num_games_profile = 100
    num_games_benchmark = 10
    num_games_other = 100
    config = {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True}
    run_seed = 57
    agent_seed = 14
    # game_seeds = [7083311470311291716]
    game_seeds = None
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
        results = minesweeper.run(solver_agent, config=config, visualise=True, verbose=False, num_games=num_games_other, seed=run_seed, game_seeds=game_seeds)
        # run(cbr_agent_1, visualise=True, verbose=True, num_games=10)
    print("Program stopped.")
