from MinesweeperAI import Minesweeper, Agents

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
    random_agent = Agents.RandomAgent()
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