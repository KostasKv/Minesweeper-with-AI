import time
from datetime import timedelta
import os
import csv
from multiprocessing import Pool, current_process
from copy import copy
import itertools

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import psutil
import more_itertools
from sqlalchemy import create_engine, MetaData, insert, select
from sqlalchemy.sql import and_
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.pool import NullPool
from bitarray import bitarray

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver


def main():
    experiment = getExperiment5()
    batch_size = 5

    # Use however many logical cores there are on the machine
    num_processes = psutil.cpu_count(logical=True)

    # Running naive alg. experiment on home PC. Freeing up processor space so PC can
    # still be used while script is running.
    num_processes -= 2

    runExperiment(experiment, batch_size, num_processes, skip_complete_tasks=False)


def runExperiment(experiment, batch_size, num_processes, skip_complete_tasks=True):
    (tasks_info, constants) = experimentPrepAndGetTasksAndConstants(
        experiment, batch_size, num_processes, skip_complete_tasks
    )

    if tasks_info is None:
        return  # All tasks already finished & skipped. End program.

    task_handler = experiment["task_handler"]

    # Run experiment tasks with multiple processes running in parallel
    with Pool(processes=num_processes) as p:
        all_results = list(
            tqdm(p.imap_unordered(task_handler, tasks_info), total=len(tasks_info))
        )

    # # DEBUG: single-process run to allow for easier debug sessions
    # all_results = [task_handler(task_info) for task_info in tasks_info]

    onEndOfExperiment(experiment, all_results, constants)


def experimentPrepAndGetTasksAndConstants(
    experiment, batch_size, num_processes, skip_complete_tasks
):
    print(f"Preparing experiment '{experiment['title']}':")
    print("Creating tasks...", end=" ")
    parameter_grid, constants = getSplitParameterGridAndConstants(experiment)
    constants["batch_size"] = batch_size
    constants["num_processes"] = num_processes
    tasks, num_param_combos = createTasksFromSplitParameterGrid(
        parameter_grid, batch_size
    )
    print("DONE")

    num_all_tasks = len(tasks)

    if skip_complete_tasks:
        print("Checking for finished tasks...", end=" ")
        finished_task_ids = fetch_finished_task_ids()
        print("DONE")

        if len(finished_task_ids) == num_all_tasks:
            print("All experiment tasks have already been finished! Exitting...")
            return (None, None)

        if finished_task_ids:
            print(
                f"{len(finished_task_ids)}/{num_all_tasks} tasks already finished. Filtering out finished tasks...",
                end=" ",
            )
            tasks = filter_finished_tasks(tasks, finished_task_ids)
            print("DONE")
        else:
            print("No tasks finished; This is the first run.")

    else:
        finished_task_ids = []

    print("Experiment ready to run.\n\n")
    num_games = constants["num_games"]
    total_games = num_games * num_param_combos

    print(
        "Running {} games for each of {} different parameter combinations...".format(
            num_games, num_param_combos
        )
    )
    if finished_task_ids:
        games_left = total_games - (batch_size * len(finished_task_ids))
        print(f"\nTotal games: {total_games}\tTotal tasks: {num_all_tasks}")
        print(
            f"Games left : {games_left}\tBatch size : {batch_size}\t\tTasks left: {len(tasks)}\t\tNum processes: {num_processes}"
        )
    else:
        print(
            f"\nTotal games: {total_games}\tBatch size: {batch_size}\t\tTotal tasks: {num_all_tasks}\t\tNum processes: {num_processes}"
        )

    return (tasks, constants)


def getSplitParameterGridAndConstants(experiment):
    agent_variables = experiment["agent_parameters"]["variable"]
    other_variables = experiment["other_parameters"]["variable"]
    agent_constants = experiment["agent_parameters"]["constant"]
    other_constants = experiment["other_parameters"]["constant"]

    if agent_variables is None:
        agent_variables = {}
    if other_variables is None:
        other_variables = {}
    if agent_constants is None:
        agent_constants = {}
    if other_constants is None:
        other_constants = {}

    variables = {}

    # Adding prefixes to distinguish between same-name variables from the two dicts
    for (key, value) in agent_variables.items():
        new_key = "agent_" + key
        variables[new_key] = value

    for (key, value) in other_variables.items():
        new_key = "other_" + key
        variables[new_key] = value

    variables_parameter_grid = ParameterGrid(variables)
    split_parameter_grid_with_constants = []

    # Split each parameter combination dict in grid into tuple of 2 dicts, where the former
    # contains all the agent parameters, and the latter all the others. Including constants.
    for parameters_combo in variables_parameter_grid:
        agent_parameters = {**agent_constants}
        other_parameters = {**other_constants}

        for (key, value) in parameters_combo.items():
            (prefix, real_key) = key.split("_", 1)

            if prefix == "agent":
                agent_parameters[real_key] = value
            else:
                other_parameters[real_key] = value

        split_parameter_grid_with_constants.append((agent_parameters, other_parameters))

    # lazy patch for 'seed' being called same thing in both dicts
    constants = {**agent_constants, **other_constants}
    constants.pop("seed")
    constants["agent_seed"] = agent_constants["seed"]
    constants["run_seed"] = other_constants["seed"]

    return (split_parameter_grid_with_constants, constants)


def createTasksFromSplitParameterGrid(parameter_grid, batch_size):
    batched_tasks_grouped_by_parameter = []
    num_param_combos = 0

    for (parameters_id, (agent_parameters, other_parameters)) in enumerate(
        parameter_grid, 1
    ):
        # SKIP NON-NULL SAMPLE SIZES THAT ARE LARGE ENOUGH TO CONTAIN ENTIRE GRID (assuming here that sample size None is one of the param choices)
        # AS THEY WILL GET THE EXACT SAME RESULTS AS THE FULL-GRID SAMPLE (denoted by sample size None)
        skip_parameters = is_sample_size_redundant(
            agent_parameters["sample_size"], other_parameters["config"]
        )
        if skip_parameters:
            continue

        num_param_combos += 1
        tasks = createTasksFromParameters(
            agent_parameters, other_parameters, batch_size=batch_size
        )

        # Sticking on the parameters_id to each task so the results from all the tasks
        # can more easily be grouped by their parameters
        tasks_with_param_id = [(parameters_id, task) for task in tasks]
        batched_tasks_grouped_by_parameter.append(tasks_with_param_id)

    interleaved_tasks = more_itertools.roundrobin(*batched_tasks_grouped_by_parameter)
    interleaved_tasks_with_task_id = [
        (task_id, param_id, task)
        for (task_id, (param_id, task)) in enumerate(interleaved_tasks, 1)
    ]
    return interleaved_tasks_with_task_id, num_param_combos


def is_sample_size_redundant(sample_size, game_config):
    if sample_size is None:
        return False  # None means full grid, and those are the non-redundant ones that are to be run.

    sample_rows, sample_cols = sample_size
    return sample_rows >= game_config["rows"] and sample_cols >= game_config["columns"]


def createTasksFromParameters(agent_parameters, other_parameters, batch_size):
    """ Batch size is the number of games to play for a task. """
    # Create game seeds for entire run on this combination of parameters
    num_games = other_parameters["num_games"]
    run_seed = other_parameters["seed"]
    game_seeds = minesweeper.create_game_seeds(num_games, run_seed)

    solver_agent = NoUnnecessaryGuessSolver(**agent_parameters)
    method = minesweeper.run
    args = (solver_agent,)
    tasks = []

    # Batch game seeds and put them into tasks
    for seed_batch in more_itertools.chunked(game_seeds, batch_size):
        kwargs = copy(
            other_parameters
        )  # Ensure each task uses a different kwargs after modification
        kwargs["game_seeds"] = seed_batch

        task = (method, args, kwargs)
        tasks.append(task)

    return tasks


def filter_finished_tasks(tasks_info, finished_task_ids):
    # Allow for O(1) lookups on average to speed up the process
    finished_task_ids = set(finished_task_ids)

    return list(itertools.filterfalse(lambda x: x[0] in finished_task_ids, tasks_info))


def fetch_finished_task_ids():
    (engine, meta_data) = get_database_engine_and_reflected_meta_data()

    table_finished_task = meta_data.tables["finished_task"]

    # Get all finished task ids from database
    with engine.begin() as connection:
        select_query = select([table_finished_task.c.id]).order_by(
            table_finished_task.c.id
        )
        finished_task_ids = connection.execute(select_query).fetchall()

    engine.dispose()

    return [task_id for (task_id,) in finished_task_ids]


def onEndOfExperiment(experiment, all_results, constants):
    # Save experiment constants info as seperate file
    constants_output_file_name = appendToFileName(experiment["title"], "_other-data")
    saveDictRowsAsCsv([constants], constants_output_file_name)

    callback = experiment["on_finish"]

    if callback:
        callback(experiment, all_results)


def saveResultsToCsv(experiment, results):
    output_file_name = experiment["title"]
    saveDictRowsAsCsv(results, output_file_name)


def saveDictRowsAsCsv(dict_rows, file_name):
    """Each dict should represent a row where the keys are the field names for the CSV file.
    File is saved in OUTPUT_DIR."""
    path = createOutputPathCSVFile(file_name)

    with open(path, "w", newline="") as csvfile:
        fieldnames = list(dict_rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def createOutputPathCSVFile(file_name):
    validateFileName(file_name)

    # Strip whitespace and .csv extension
    file_name = file_name.strip()

    # Remove .csv extension
    file_name_without_ext, ext = os.path.splitext(file_name)
    if ext == ".csv":
        file_name = file_name_without_ext

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")

    path = os.path.join(OUTPUT_DIR, file_name)

    # Don't overwrite existing data. Instead append a number to the filename.
    if os.path.exists(path + ".csv"):
        i = 1
        while os.path.exists(f"{path}({i}).csv"):
            i += 1

        path += f"({i})"

    path += ".csv"

    return path


def validateFileName(output_file_name):
    try:
        assert isinstance(output_file_name, str)
        assert output_file_name.strip() != ""
    except:
        raise ValueError("Output file name must be a non-empty string.")


def appendToFileName(name, suffix):
    (name, ext) = os.path.splitext(name)
    return name + suffix + ext


def task_handler_store_results_in_database_on_task_finish(task_info):
    """Warning: Only for use with main experiment! Not configured to work with
    any other experiment."""

    # unpack
    (task_id, _, task) = task_info
    (method, args, kwargs) = task

    # run task
    results = method(*args, **kwargs)

    # Extra info for results to be stored
    results["pid"] = current_process().pid
    results["task_id"] = task_id

    store_task_results_in_database(results, task)


def store_task_results_in_database(results, task):
    (engine, meta_data) = get_database_engine_and_reflected_meta_data()
    game_config = extract_game_config_from_task(task)
    transaction_attempts_left = 10
    transaction_not_complete = True

    while transaction_not_complete and transaction_attempts_left > 0:
        try:
            store_results_in_single_transaction(results, game_config, engine, meta_data)
            transaction_not_complete = False
        except OperationalError:
            # Deadlock detected. Retry transaction
            transaction_attempts_left -= 1

    if transaction_attempts_left < 0:
        raise OperationalError  # All transaction attempts failed. Proceed to panic.

    engine.dispose()


def store_results_in_single_transaction(results, game_config, engine, meta_data):
    with Session(engine, future=True) as session:
        # Finished task entity has to be stored first so that each inserted game entity can reference this task (with task id)
        store_finished_task(session, meta_data, results)

        for game_stats in results["games"]:
            grid_id = store_grid_if_not_exists(
                session, meta_data, game_config, game_stats
            )

            game_stats["task_id"] = results["task_id"]
            game_stats["grid_id"] = grid_id
            store_game_and_turns_and_samples(session, meta_data, game_stats)

        session.commit()


def get_database_engine_and_reflected_meta_data():
    # bank account details
    user = "cokk"
    # topsecretword = "8iCyrvxoK4RMitkZ"
    # host = "lnx-cokk-1.lunet.lboro.ac.uk"
    db_name = "run2"

    # topsecretword = "password"
    # host = "localhost"

    engine = create_engine(
        f"mysql://{user}:{topsecretword}@{host}/{db_name}?charset=utf8mb4",
        isolation_level="SERIALIZABLE",
        poolclass=NullPool,
    )

    meta_data = MetaData()
    meta_data.reflect(engine)

    return (engine, meta_data)


def extract_game_config_from_task(task):
    (_, _, x) = task
    return x["config"]


def store_finished_task(session, meta_data, results):
    table_finished_task = meta_data.tables["finished_task"]
    insert_task_id = insert(table_finished_task).values(
        id=results["task_id"], pid=results["pid"]
    )
    session.execute(insert_task_id)


def store_grid_if_not_exists(session, meta_data, game_config, game_stats):
    # Get tables required
    table_difficulty = meta_data.tables["difficulty"]
    table_grid = meta_data.tables["grid"]

    # Reference to difficulty needed for grid, so fetch difficulty id
    difficulty_id = fetch_difficulty_id(session, table_difficulty, game_config)

    # Extract game info needed for just the grid entity
    game_seed = game_stats["seed"]
    grid_mines = game_stats["grid_mines"]
    grid_mines = grid_to_binary(grid_mines)

    try:
        # Try blindly inserting grid first
        insert_grid = insert(table_grid).values(
            difficulty_id=difficulty_id, seed=game_seed, grid_mines=grid_mines
        )
        result = session.execute(insert_grid)
        grid_id = result.lastrowid
    except IntegrityError:
        # This specific grid already exists in the database. Let's fetch its id.
        query = select([table_grid.c.id]).where(
            and_(
                table_grid.c.difficulty_id == difficulty_id,
                table_grid.c.seed == game_seed,
                table_grid.c.grid_mines == grid_mines,
            )
        )
        result = session.execute(query).fetchone()

        grid_id = result[0]

    return grid_id


def fetch_difficulty_id(session, table_difficulty, game_config):
    query = select([table_difficulty.c.id]).where(
        and_(
            table_difficulty.c.rows == game_config["rows"],
            table_difficulty.c.columns == game_config["columns"],
            table_difficulty.c.mines == game_config["num_mines"],
        )
    )
    result = session.execute(query).fetchone()
    return result[0]


def store_game_and_turns_and_samples(session, meta_data, game_stats):
    # Get table references
    table_game = meta_data.tables["game"]
    table_turn = meta_data.tables["turn"]
    table_sample = meta_data.tables["sample"]

    # GAME - store and get id (to put into turn entities)
    game_attributes = {
        k: v
        for (k, v) in game_stats.items()
        if k not in ["seed", "grid_mines", "turns"]
    }
    game_id = store_entity_and_return_id(session, table_game, game_attributes)

    for turn_stats in game_stats["turns"]:
        # TURN - store and get turn id (to put into sample entities)
        turn_attributes = {
            k: v for (k, v) in turn_stats.items() if k != "samples_stats"
        }
        turn_attributes["game_id"] = game_id
        turn_id = store_entity_and_return_id(session, table_turn, turn_attributes)

        for sample_stats in turn_stats["samples_stats"]:
            sample_stats["turn_id"] = turn_id

            # SAMPLE - store
            convert_fields_and_store_sample(session, table_sample, sample_stats)


def convert_fields_and_store_sample(session, table_sample, sample_stats):
    sample_attributes = {
        k: v
        for (k, v) in sample_stats.items()
        if k not in ["disjoint_sections_sizes", "has_wall"]
    }
    sample_attributes["disjoint_sections_sizes"] = encode_disjoint_sections_sizes(
        sample_stats["disjoint_sections_sizes"]
    )
    sample_attributes["has_wall"] = encode_has_wall(sample_stats["has_wall"])

    return store_entity_and_return_id(session, table_sample, sample_attributes)


def store_entity_and_return_id(session, table, entity_dict):
    """Input game stats show be a dict representing a single game entity
    where each (key, value) pair in the dict is a column name and that field's value."""
    insert_query = insert(table).values(entity_dict)
    result = session.execute(insert_query)
    return result.lastrowid


def grid_to_binary(grid):
    binary_grid = bitarray()

    for row in grid:
        for tile in row:
            if tile.is_mine:
                binary_grid.append(True)
            else:
                binary_grid.append(False)

    return binary_grid.tobytes()


def encode_disjoint_sections_sizes(disjoint_sections_sizes):
    """encoding is a string in format "x1,y1#x2,y2#...#xn,yn" where xi and yi are the
    number of tiles in the fringe and frontier, respectively, of the i'th section
    (n sections overall)."""
    return bytes(
        "#".join(
            f"{frontier_len},{fringe_len}"
            for (frontier_len, fringe_len) in disjoint_sections_sizes
        ),
        "utf8",
    )


def encode_has_wall(has_wall):
    """ Encoding format is binary string: 0000urdl, where each u,r,d,l is 1 if wall in that direction, 0 otherwise (u-up, r-right, d-down, l-left) """
    binary_has_wall = bitarray()

    # Pad with 4 zeros
    for _ in range(4):
        binary_has_wall.append(0)

    binary_has_wall.append(has_wall["top"])
    binary_has_wall.append(has_wall["right"])
    binary_has_wall.append(has_wall["bottom"])
    binary_has_wall.append(has_wall["left"])

    return binary_has_wall


def complete_task_and_return_results_including_game_info(task_info):
    start = time.time()

    # unpack
    (*_, task) = task_info
    (method, args, kwargs) = task

    # run task
    results = method(*args, **kwargs)

    end = time.time()
    results["task_time_elapsed"] = end - start
    return add_extra_info_to_task_results(results, task_info)


def add_extra_info_to_task_results(results_initial, task_info):
    """ Extends task's results with extra information. """
    (task_id, parameters_id, task) = task_info
    _, args, kargs = task
    agent = args[0]

    results = {
        "wins": results_initial["wins"],
        "wins_without_guess": results_initial["wins_without_guess"],
        "time_elapsed": results_initial["time_elapsed"],
        "samples_considered": results_initial["samples_considered"],
        "samples_with_solutions": results_initial["samples_with_solutions"],
    }

    results["difficulty"] = configToDifficultyString(kargs["config"])
    results["sample_size"] = "x".join(
        str(num) for num in agent.SAMPLE_SIZE
    )  # represent sample size (A, B) as string 'AxB'. Bit easier to understand.
    results["use_num_mines_constraint"] = agent.use_num_mines_constraint
    results["first_click_pos"] = agent.first_click_pos
    results["first_click_is_zero"] = kargs["config"]["first_click_is_zero"]
    results["naive_alg_steps"] = agent.naive_alg_steps
    results["parameters_id"] = parameters_id
    results["task_id"] = task_id

    return results


def configToDifficultyString(config):
    if config["columns"] == 8 and config["rows"] == 8:
        difficulty = "Beginner (8x8)"
    elif config["columns"] == 9 and config["rows"] == 9:
        difficulty = "Beginner (9x9)"
    elif config["columns"] == 16 and config["rows"] == 16:
        difficulty = "Intermediate (16x16)"
    elif config["columns"] == 30 and config["rows"] == 16:
        difficulty = "Expert (16x30)"
    else:
        raise ValueError("Difficulty not recognised from config {}".format(config))

    return difficulty


def getExperimentTEST():
    """ Purpose: like experiment1 but just for testing this script & has few games"""

    title = "T to the E to the ST"

    agent_parameters = {
        "variable": {"first_click_pos": [None, (3, 3)]},
        "constant": {
            "seed": 14,
            "sample_size": None,  # Sample size None means use full grid
            "use_num_mines_constraint": True,
        },
    }

    other_parameters = {
        "variable": {
            "config": [
                {"rows": 9, "columns": 9, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 9,
                    "columns": 9,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": False,
                },
            ],
        },
        "constant": {
            "num_games": 50,
            "seed": 57,
            "verbose": False,
            "visualise": False,
        },
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        "title": title,
        "agent_parameters": agent_parameters,
        "other_parameters": other_parameters,
        "task_handler": task_handler,
        "on_finish": on_finish,
    }

    return experiment


def getExperiment1():
    """Expected Duration: ~2 hours

    Purpose: The rule of whether first-click is a zero tile, and first-click's position (random or fixed) seems
    to affect the win rate. Each combination is tried to see how large of an effect they have and how these
    compare to the existing solvers.
    """

    title = "First-click variation experiment"

    agent_parameters = {
        "variable": {"first_click_pos": [None, (3, 3)]},
        "constant": {
            "seed": 14,
            "sample_size": None,  # Sample size None means use full grid
            "use_num_mines_constraint": True,
        },
    }

    other_parameters = {
        "variable": {
            "config": [
                {"rows": 9, "columns": 9, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 9,
                    "columns": 9,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": False,
                },
            ],
        },
        "constant": {
            "num_games": 1000,
            "seed": 57,
            "verbose": False,
            "visualise": False,
        },
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        "title": title,
        "agent_parameters": agent_parameters,
        "other_parameters": other_parameters,
        "task_handler": task_handler,
        "on_finish": on_finish,
    }

    return experiment


def getExperiment2():
    """Purpose: Twofold:
        1. Measure how the win rate converges as more games are played (to give an estimate of how many games
           need to be played by the main experiment per parameter combo for precise results)
        2. Verify that the solver's win rate is the rate expected for each difficulty
           using the full grid as its sample size (expected win rate comes from existing solvers that also initially
           exhaust all definitely-safe moves that can be deduced).

    Expected win rates (no guesses) are as follows:
    - ~92% Beginner
    - ~72% Intermediate
    - ~16% Expert
    """

    title = "Solver verification experiment"

    agent_parameters = {
        "variable": {},
        "constant": {
            "seed": 20,
            "sample_size": None,  # Sample size None means use full grid
            "use_num_mines_constraint": True,
            "first_click_pos": (3, 3),
        },
    }

    other_parameters = {
        "variable": {
            "config": [
                {"rows": 9, "columns": 9, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": True,
                },
            ],
        },
        "constant": {
            "num_games": 100000,
            "seed": 2020,
            "verbose": False,
            "visualise": False,
        },
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        "title": title,
        "agent_parameters": agent_parameters,
        "other_parameters": other_parameters,
        "task_handler": task_handler,
        "on_finish": on_finish,
    }

    return experiment


def getExperiment3():
    """Purpose: A short experiment (~1 day or less) to do the following main things, among others:
    1. Try out numerous sample sizes with/without mine count constraint so as to give an indication of what the win rates look like for them (to help choose
       which sample sizes to test in a bigger solver experiment with many more games).
    2. Provide measurements of how long it takes to play a certain number of games for specific parameter combos, allowing for a reasonable estimate
       of the total run time to be made for any subsequent bigger solver experiments."""

    title = "Short broad experiment"

    agent_parameters = {
        "variable": {
            "sample_size": [
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                None,
            ],  # Sample size None means use full grid
            "use_num_mines_constraint": [True, False],
        },
        "constant": {"seed": 2020, "first_click_pos": None},
    }

    other_parameters = {
        "variable": {
            "config": [
                {"rows": 8, "columns": 8, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 8,
                    "columns": 8,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {"rows": 9, "columns": 9, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 9,
                    "columns": 9,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": False,
                },
            ],
        },
        "constant": {
            "num_games": 1000,
            "seed": 20,
            "verbose": False,
            "visualise": False,
        },
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        "title": title,
        "agent_parameters": agent_parameters,
        "other_parameters": other_parameters,
        "task_handler": task_handler,
        "on_finish": on_finish,
    }

    return experiment


def getExperiment4():
    title = "Main experiment"

    agent_parameters = {
        "variable": {
            "sample_size": [
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10),
                None,
            ],  # Sample size None means use full grid
            "use_num_mines_constraint": [True, False],
            "can_flag": [True, False],
        },
        "constant": {
            "seed": 4040,
            "first_click_pos": None,
        },
    }

    other_parameters = {
        "variable": {
            "config": [
                {"rows": 8, "columns": 8, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 8,
                    "columns": 8,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {"rows": 9, "columns": 9, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 9,
                    "columns": 9,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": False,
                },
            ],
        },
        "constant": {
            "num_games": 25000,
            "seed": 40,
            "verbose": False,
            "visualise": False,
        },
    }

    task_handler = task_handler_store_results_in_database_on_task_finish
    on_finish = None

    experiment = {
        "title": title,
        "agent_parameters": agent_parameters,
        "other_parameters": other_parameters,
        "task_handler": task_handler,
        "on_finish": on_finish,
    }

    return experiment


def getExperiment5():
    """Purpose: Measure performance of naive algorithm."""

    title = "Naive algorithm experiment"

    agent_parameters = {
        "variable": {"naive_alg_steps": [0, 1, 2, 3, 4, 5, 6, 7, None]},
        "constant": {
            "seed": 20,
            "sample_size": None,  # Sample size None means use full grid
            "first_click_pos": None,
        },
    }

    other_parameters = {
        "variable": {
            "config": [
                {"rows": 8, "columns": 8, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 8,
                    "columns": 8,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {"rows": 9, "columns": 9, "num_mines": 10, "first_click_is_zero": True},
                {
                    "rows": 9,
                    "columns": 9,
                    "num_mines": 10,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 16,
                    "num_mines": 40,
                    "first_click_is_zero": False,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": True,
                },
                {
                    "rows": 16,
                    "columns": 30,
                    "num_mines": 99,
                    "first_click_is_zero": False,
                },
            ],
        },
        "constant": {
            "num_games": 25000,
            "seed": 2020,
            "verbose": False,
            "visualise": False,
        },
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        "title": title,
        "agent_parameters": agent_parameters,
        "other_parameters": other_parameters,
        "task_handler": task_handler,
        "on_finish": on_finish,
    }

    return experiment


if __name__ == "__main__":
    main()
