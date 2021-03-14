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
from sqlalchemy.dialects.mysql import VARBINARY
from bitarray import bitarray

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver


def main():
    experiment = getExperiment4()
    batch_size = 1
    num_processes = 6

    runExperiment(experiment, batch_size, num_processes)


def runExperiment(experiment, batch_size, num_processes):
    (tasks_info, constants) = experimentPrepAndGetTasksAndConstants(experiment, batch_size, num_processes)
    
    task_handler = experiment['task_handler']

    # Run experiment tasks with multiple processes running in parallel
    # with Pool(processes=num_processes) as p:
    #     all_results = list(tqdm(p.imap_unordered(task_handler, tasks_info), total=len(tasks_info)))

    # single-t
    all_results = [task_handler(task_info) for task_info in tasks_info]

    onEndOfExperiment(experiment, all_results, constants)

def experimentPrepAndGetTasksAndConstants(experiment, batch_size, num_processes):
    print("Preparing experiment '{}'...".format(experiment['title']), end="")

    parameter_grid, constants = getSplitParameterGridAndConstants(experiment)
    constants['batch_size'] = batch_size
    constants['num_processes'] = num_processes

    tasks = createTasksFromSplitParameterGrid(parameter_grid, batch_size)
    num_combinations = len(parameter_grid)
    num_games = constants['num_games']

    # Display start-of-experiment info
    print(" DONE")
    print("Running {} games for each of {} different parameter combinations...".format(num_games, num_combinations))
    print("\nTotal games: {}   Batch size: {}   Total tasks: {}    Num processes: {}".format((num_games * num_combinations), batch_size, len(tasks), num_processes))
    return (tasks, constants)

def getSplitParameterGridAndConstants(experiment):
    agent_variables = experiment['agent_parameters']['variable']
    other_variables = experiment['other_parameters']['variable']
    agent_constants = experiment['agent_parameters']['constant']
    other_constants = experiment['other_parameters']['constant']

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
        new_key = 'agent_' + key
        variables[new_key] = value

    for (key, value) in other_variables.items():
        new_key = 'other_' + key
        variables[new_key] = value

    variables_parameter_grid = ParameterGrid(variables)
    split_parameter_grid_with_constants = []
    
    # Split each parameter combination dict in grid into tuple of 2 dicts, where the former
    # contains all the agent parameters, and the latter all the others. Including constants.
    for parameters_combo in variables_parameter_grid:
        agent_parameters = {**agent_constants}
        other_parameters = {**other_constants}

        for (key, value) in parameters_combo.items():
            (prefix, real_key) = key.split('_', 1)

            if prefix == 'agent':
                agent_parameters[real_key] = value
            else:
                other_parameters[real_key] = value
        
        split_parameter_grid_with_constants.append((agent_parameters, other_parameters))

    # lazy patch for 'seed' being called same thing in both dicts
    constants = {**agent_constants, **other_constants}
    constants.pop('seed')
    constants['agent_seed'] = agent_constants['seed']
    constants['run_seed'] = other_constants['seed']

    return (split_parameter_grid_with_constants, constants)

def createTasksFromSplitParameterGrid(parameter_grid, batch_size):
    batched_tasks_grouped_by_parameter = []

    for (parameters_id, (agent_parameters, other_parameters)) in enumerate(parameter_grid, 1):
        tasks = createTasksFromParameters(agent_parameters, other_parameters, batch_size=batch_size)
        
        # Sticking on the parameters_id to each task so the results from all the tasks 
        # can more easily be grouped by their parameters
        tasks_with_param_id = [(parameters_id, task) for task in tasks]
        batched_tasks_grouped_by_parameter.append(tasks_with_param_id)

    return list(more_itertools.roundrobin(*batched_tasks_grouped_by_parameter))

def createTasksFromParameters(agent_parameters, other_parameters, batch_size):
    ''' Batch size is the number of games to play for a task. '''
    # Create game seeds for entire run on this combination of parameters
    num_games = other_parameters['num_games']
    run_seed = other_parameters['seed']
    game_seeds = minesweeper.create_game_seeds(num_games, run_seed)
    
    solver_agent = NoUnnecessaryGuessSolver(**agent_parameters)
    method = minesweeper.run
    args = (solver_agent, )
    tasks = []

    # Batch game seeds and put them into tasks
    for seed_batch in more_itertools.chunked(game_seeds, batch_size):
        kwargs = copy(other_parameters)     # Ensure each task uses a different kwargs after modification
        kwargs['game_seeds'] = seed_batch

        task = (method, args, kwargs)
        tasks.append(task)

    return tasks

def onEndOfExperiment(experiment, all_results, constants):
    # Save experiment constants info as seperate file
    constants_output_file_name = appendToFileName(experiment['title'], "_other-data")
    saveDictRowsAsCsv([constants], constants_output_file_name)

    callback = experiment['on_finish']
    callback(experiment, all_results)

def saveResultsToCsv(experiment, results):
    output_file_name = experiment['title']
    saveDictRowsAsCsv(results, output_file_name)

def saveDictRowsAsCsv(dict_rows, file_name):
    ''' Each dict should represent a row where the keys are the field names for the CSV file.
        File is saved in OUTPUT_DIR. '''
    path = createOutputPathCSVFile(file_name)

    with open(path, 'w', newline='') as csvfile:
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
    if ext == '.csv':
        file_name = file_name_without_ext

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data')

    path = os.path.join(OUTPUT_DIR, file_name)

    # Don't overwrite existing data. Instead append a number to the filename.
    if os.path.exists(path + '.csv'):
        i = 1
        while os.path.exists(f"{path}({i}).csv"):
            i += 1
        
        path += f'({i})'
    
    path += '.csv'

    return path

def validateFileName(output_file_name):
    try:
        assert(isinstance(output_file_name, str))
        assert(output_file_name.strip() != '')
    except:
        raise ValueError("Output file name must be a non-empty string.")

def appendToFileName(name, suffix):
    (name, ext) = os.path.splitext(name)
    return name + suffix + ext

def task_handler_store_results_in_database_on_task_finish(task_info):
    ''' Warning: Only for use with main experiment! Not configured to work for
        any other experiment.'''
    
    # unpack
    (task_id, task) = task_info
    (method, args, kwargs) = task   

    # run task
    results = method(*args, **kwargs)            

    # Extra info for results to be stored  
    results['pid'] = current_process().pid
    results['task_id'] = task_id
    
    return store_task_results_in_database(results, task_info)

def store_task_results_in_database(results, task_info):
    # bank account details
    user = "cokk"
    # topsecretword = "8iCyrvxoK4RMitkZ" 
    # host = "lnx-cokk-1.lunet.lboro.ac.uk"
    db_name = "cokk"

    topsecretword = "password"
    host = "localhost"

    # connect
    engine = create_engine(f'mysql://{user}:{topsecretword}@{host}/{db_name}?charset=utf8mb4')
    # engine.connect()
    
    # Get needed tables from DB
    meta_data = MetaData()
    meta_data.reflect(engine)
    table_difficulty = meta_data.tables['difficulty']
    table_grid = meta_data.tables['grid']
    table_game = meta_data.tables['game']
    table_turn = meta_data.tables['turn']
    table_sample = meta_data.tables['sample']
    table_finished_task = meta_data.tables['finished_task']

    (_, (_, _, x)) = task_info
    game_config = x['config']


    # Inserts results into DB, all within a single transaction
    with engine.begin() as connection:
        for game_stats in results['games']:
            difficuly_id = fetch_difficulty_id(connection, table_difficulty, game_config)
            grid_id = store_grid_if_not_exists(connection, table_grid, game_stats, difficuly_id)
            store_game_and_related_entities(connection, meta_data, game_stats, grid_id)
            
        store_finished_task(connection, table_finished_task, results)
        

def store_grid_if_not_exists(connection, table_grid, game_stats, difficulty_id):
    grid_mines = grid_to_binary(game_stats['grid_mines'])

    # Get grid id
    query = select([table_grid.c.id]).where(
        and_(
            table_grid.c.difficulty_id == difficulty_id,
            table_grid.c.seed == game_stats['seed'],
            table_grid.c.grid_mines == grid_mines,
            )
        )
    result = connection.execute(query).fetchone()

    if result is None:
        # Insert grid into DB as it hasn't been inserted already
        insert_grid = insert(table_grid).values(difficulty_id=difficulty_id, seed=game_stats['seed'], grid_mines=grid_mines)
        insert_grid.returning(table_grid.c.id)
        insert_grid.prefix_with('ON DUPLICATE IGNORE')
        result = connection.execute(insert_grid)
        grid_id = result.lastrowid
    else:
        grid_id = result[0]

    return grid_id

def fetch_difficulty_id(connection, table_difficulty, game_config):
    query = select([table_difficulty.c.id]).where(
        and_(
            table_difficulty.c.rows == game_config['rows'],
            table_difficulty.c.columns == game_config['columns'],
            table_difficulty.c.mines == game_config['num_mines'],
            )
        )

    result = connection.execute(query).fetchone()
    return result[0]

def store_game_and_related_entities(connection, meta_data, game_stats, grid_id):
    # Get table references
    table_game = meta_data.tables['game']
    table_turn = meta_data.tables['turn']
    table_sample = meta_data.tables['sample']

    # Store game
    game_id = store_game_entity(connection, table_game, game_stats, grid_id)

    # Store turns of game and each turn's samples
    for turn_stats in results['turns']:
        turn_id = store_turn_entity(connection, table_turn, turn_stats, game_id)

        samples_stats = turn_stats['samples_stats']
        store_samples(connection, table_sample, samples_stats, samples_stats, turn_id)

def store_game_entity(connection, table_game, game_stats, grid_id):
    return
    # insert_game_query = insert(table_Game).values(
    #     grid_id=grid_id,
    #     win=
    #     )

def store_turn_entity(connection, table_turn, turn_stats, game_id):
    return

def store_samples(connection, table_sample, samples_stats, turn_id):
    for sample_stats in samples_stats:
        pass
        # insert_sample_query = insert(table_sample).values()

def store_finished_task(connection, table_finished_task, results):
    insert_task_id = insert(table_finished_task).values(id=results['task_id'], pid=results['pid'])
    connection.execute(insert_task_id)


def grid_to_binary(grid):
    binary_grid = bitarray()

    for row in grid:
        for tile in row:
            if tile.is_mine:
                binary_grid.append(True)
            else:
                binary_grid.append(False)

    return binary_grid

def grid_pos_to_binary(pos):
    (x, y) = pos
    binary_string = "{0:08b}".format(x) + "{0:08b}".format(y)    # Two 8-digit binary strings, each representing integers x and y, concatenated.
    return bitarray(binary_string)


def complete_task_and_return_results_including_game_info(task_info):
    start = time.time()

    (method, args, kwargs) = task_info[1]                  # unpack
    results = method(*args, **kwargs)                 # run task

    end = time.time()
    results['task_time_elapsed'] = end - start
    return add_extra_info_to_task_results(results, task_info)

def add_extra_info_to_task_results(results_initial, task_info):
    ''' Extends task's results with extra information. '''
    (parameters_id, task) = task_info
    _, args, kargs = task
    agent = args[0]

    results = {
        'wins': results_initial['wins'],
        'wins_without_guess': results_initial['wins_without_guess'],
        'time_elapsed': results_initial['time_elapsed'],
        'samples_considered': results_initial['samples_considered'],
        'samples_with_solutions': results_initial['samples_with_solutions'],
        }

    results['difficulty'] = configToDifficultyString(kargs['config'])
    results['sample_size'] = 'x'.join(str(num) for num in agent.SAMPLE_SIZE) # represent sample size (A, B) as string 'AxB'. Bit easier to understand.
    results['use_num_mines_constraint'] = agent.use_num_mines_constraint
    results['first_click_pos'] = agent.first_click_pos
    results['first_click_is_zero'] = kargs['config']['first_click_is_zero']
    results['parameters_id'] = parameters_id

    return results

def configToDifficultyString(config):
    if config['columns'] == 8 and config['rows'] == 8:
        difficulty = 'Beginner (8x8)'
    if config['columns'] == 9 and config['rows'] == 9:
        difficulty = 'Beginner (9x9)'
    elif config['columns'] == 16 and config['rows'] == 16:
        difficulty = 'Intermediate (16x16)'
    elif config['columns'] == 30 and config['rows'] == 16:
        difficulty = 'Expert (16x30)'
    else:
        raise ValueError("Difficulty not recognised from config {}".format(config))

    return difficulty

def getExperimentTEST():
    ''' Purpose: like experiment1 but just for testing this script & has few games'''
        
    title = "T to the E to the ST"

    agent_parameters = {
        'variable': {
            'first_click_pos': [None, (3, 3)]
        },
        'constant': {
            'seed': 14,
            'sample_size': None,  # Sample size None means use full grid
            'use_num_mines_constraint': True,
        }
    }

    other_parameters = {
        'variable': {
            'config': [
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': False}
            ],
        },
        'constant': {
            'num_games': 50,
            'seed': 57,
            'verbose': False,
            'visualise': False,  
        }
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters,
        'task_handler': task_handler,
        'on_finish': on_finish
    }

    return experiment

def getExperiment1():
    ''' Expected Duration: ~2 hours
        
        Purpose: The rule of whether first-click is a zero tile, and first-click's position (random or fixed) seems
        to affect the win rate. Each combination is tried to see how large of an effect they have and how these
        compare to the existing solvers.
    '''
        
    title = "First-click variation experiment"

    agent_parameters = {
        'variable': {
            'first_click_pos': [None, (3, 3)]
        },
        'constant': {
            'seed': 14,
            'sample_size': None,  # Sample size None means use full grid
            'use_num_mines_constraint': True,
        }
    }

    other_parameters = {
        'variable': {
            'config': [
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': False}
            ],
        },
        'constant': {
            'num_games': 1000,
            'seed': 57,
            'verbose': False,
            'visualise': False,  
        }
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters,
        'task_handler': task_handler,
        'on_finish': on_finish
    }

    return experiment

def getExperiment2():
    ''' Purpose: Twofold:
            1. Measure how the win rate converges as more games are played (to give an estimate of how many games
               need to be played by the main experiment per parameter combo for precise results)
            2. Verify that the solver's win rate is the rate expected for each difficulty
               using the full grid as its sample size (expected win rate comes from existing solvers that also initially
               exhaust all definitely-safe moves that can be deduced).

        Expected win rates (no guesses) are as follows:
        - ~92% Beginner
        - ~72% Intermediate
        - ~16% Expert 
    '''

    # title = "Solver verification experiment"
    title = "Batch size and processes count experiment" # Temp renaming for other experiment (hijacking this old experiment)

    agent_parameters = {
        'variable': {
            
        },
        'constant': {
            'seed': 20,
            'sample_size': None,  # Sample size None means use full grid
            'use_num_mines_constraint': True,
            'first_click_pos': (3, 3),
        }
    }

    other_parameters = {
        'variable': {
            'config': [
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True},
            ],
        },
        'constant': {
            # 'num_games': 50000,
            'num_games': 7500,   # temp for batch size & process count experiment
            'seed': 2020,
            'verbose': False,
            'visualise': False,  
        }
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters,
        'task_handler': task_handler,
        'on_finish': on_finish
    }

    return experiment


def getExperiment3():
    ''' Purpose: A short experiment (~1 day or less) to do the following main things, among others: 
                 1. Try out numerous sample sizes with/without mine count constraint so as to give an indication of what the win rates look like for them (to help choose
                    which sample sizes to test in a bigger solver experiment with many more games).
                 2. Provide measurements of how long it takes to play a certain number of games for specific parameter combos, allowing for a reasonable estimate
                    of the total run time to be made for any subsequent bigger solver experiments. '''
        
    title = "Short broad experiment"

    agent_parameters = {
        'variable': {
            'sample_size': [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), None],  # Sample size None means use full grid
            'use_num_mines_constraint': [True, False],
        },
        'constant': {
            'seed': 2020,
            'first_click_pos': None
        }
    }

    other_parameters = {
        'variable': {
            'config': [
                {'rows': 8, 'columns': 8, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 8, 'columns': 8, 'num_mines': 10, 'first_click_is_zero': False},
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': False},
            ],
        },
        'constant': {
            'num_games': 1000,
            'seed': 20,
            'verbose': False,
            'visualise': False,  
        }
    }

    task_handler = complete_task_and_return_results_including_game_info
    on_finish = saveResultsToCsv

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters,
        'task_handler': task_handler,
        'on_finish': on_finish
    }

    return experiment

def getExperiment4():
    title = "Main experiment"

    agent_parameters = {
        'variable': {
            'sample_size': [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), None],  # Sample size None means use full grid
            'use_num_mines_constraint': [True, False],
        },
        'constant': {
            'seed': 4040,
            'first_click_pos': None,
        }
    }

    other_parameters = {
        'variable': {
            'config': [
                {'rows': 8, 'columns': 8, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 8, 'columns': 8, 'num_mines': 10, 'first_click_is_zero': False},
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': True},
                {'rows': 9, 'columns': 9, 'num_mines': 10, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 16, 'num_mines': 40, 'first_click_is_zero': False},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': True},
                {'rows': 16, 'columns': 30, 'num_mines': 99, 'first_click_is_zero': False},
            ],
        },
        'constant': {
            # 'num_games': 100000,
            'num_games': 10,
            'seed': 40,
            'verbose': False,
            'visualise': False,  
        }
    }

    task_handler = task_handler_store_results_in_database_on_task_finish
    on_finish = None

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters,
        'task_handler': task_handler,
        'on_finish': on_finish
    }

    return experiment


if __name__ == '__main__':
    main()