import time
from datetime import timedelta
import os
import csv
from multiprocessing import Pool
from copy import copy

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import more_itertools

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver


def main():
    # experiment = getExperiment2()
    experiment = getExperimentTEST()
    games_batch_size = 10
    runExperiment(experiment, games_batch_size)

def runExperiment(experiment, batch_size):
    (tasks_info, constants) = experimentPrepAndGetTasksAndConstants(experiment, batch_size)

    task_handler = experiment['task_handler']
    CPUs_available = os.cpu_count()

    # Run experiment using all CPU cores available
    with Pool(processes=CPUs_available) as p:
        all_results = list(tqdm(p.imap_unordered(task_handler, tasks_info), total=len(tasks_info)))

    onEndOfExperiment(experiment, all_results, constants)

def experimentPrepAndGetTasksAndConstants(experiment, batch_size):
    print("Preparing experiment '{}'...".format(experiment['title']), end="")

    parameter_grid, constants = getSplitParameterGridAndConstants(experiment)
    tasks = createTasksFromSplitParameterGrid(parameter_grid, batch_size)
    num_combinations = len(parameter_grid)
    num_games = constants['num_games']

    # Display start-of-experiment info
    print(" DONE")
    print("Running {} games for each of {} different parameter combinations...".format(num_games, num_combinations))
    print("\nTotal games: {}   Batch size: {}   Total tasks: {}".format((num_games * num_combinations), batch_size, len(tasks)))
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
    for (i, parameters_combo) in enumerate(variables_parameter_grid):
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
    tasks_info = []

    for (parameters_id, (agent_parameters, other_parameters)) in enumerate(parameter_grid, 1):
        tasks = createTasksFromParameters(agent_parameters, other_parameters, batch_size=batch_size)
        
        # Sticking on the parameters_id to each task so the results from all the tasks 
        # can more easily be grouped by their parameters
        for task in tasks:
            task_info = (parameters_id, task)
            tasks_info.append(task_info)

    return tasks_info

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

def completeTaskAndReturnExtendedResults(task_info):
    (method, args, kwargs) = task_info[1]                  # unpack
    results = method(*args, **kwargs)                 # run task
    return packageTaskResults(results, task_info)

def packageTaskResults(results, task_info):
    ''' Extends task's results with extra information. '''
    (parameters_id, task) = task_info
    _, args, kargs = task
    agent = args[0]

    results['difficulty'] = configToDifficultyString(kargs['config'])
    results['sample_size'] = 'x'.join(str(num) for num in agent.SAMPLE_SIZE) # represent sample size (A, B) as string 'AxB'. Bit easier to understand.
    results['use_num_mines_constraint'] = agent.use_num_mines_constraint
    results['first_click_pos'] = agent.first_click_pos
    results['first_click_is_zero'] = kargs['config']['first_click_is_zero']
    results['parameters_id'] = parameters_id

    return results

def configToDifficultyString(config):
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

    task_handler = completeTaskAndReturnExtendedResults
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
    ''' Purpose: A very short experiment (<1 hour) to verify that the solver's win rate is as expected for each difficulty
        using the full grid as its sample size (this is compared to other existing solvers that also play all safe moves
        that can be deduced first).

        Expected win rates (no guesses) are as follows:
        - ~92% Beginner
        - ~72% Intermediate
        - ~16% Expert 
        
        A select few factors are varied as they may have an impact on the win rate. Each combination is tried in the hopes
        of finding the set of results that are comparable to other existing solvers, and also to see how large of an impact they
        have on the win rate, if any.
    '''
        
    title = "Solver Verification Experiment"

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

    task_handler = completeTaskAndReturnExtendedResults
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
    ''' Purpose: A short experiment (~1 day or less) to do the following main things, among others: 
                 1. Verify no-unecessary-guess solver works as intended by checking the solver gets the expected win rates for each difficulties (full grid, no guess).
                 2. Try out numerous sample sizes with/without mine count constraint so as to give an indication of what the win rates look like for them (to help choose
                    which sample sizes to test in a bigger solver experiment with many more games).
                 3. Provide measurements of how long it takes to play a certain number of games for specific parameter combos, allowing for a reasonable estimate
                    of the total run time to be made for any subsequent bigger solver experiments. '''
        
    title = "Short Experiment"

    agent_parameters = {
        'variable': {
            'sample_size': [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), None],  # Sample size None means use full grid
            'use_num_mines_constraint': [True, False]
        },
        'constant': {
            'seed': 14,
            'first_click_pos': None
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
            'num_games': 2,
            'seed': 57,
            'verbose': False,
            'visualise': False,  
        }
    }

    task_handler = completeTaskAndReturnExtendedResults
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
    title = "Main Experiment"

    agent_parameters = {
        'variable': {
            'first_click_pos': [None, (2, 2)],
            'first_click_is_zero': [True, False],
            
        },
        'constant': {
            'sample_size': None,  # Sample size None means use full grid
            'seed': 20,
            'first_click_pos': None,
            'use_num_mines_constraint': True
        }
    }

    other_parameters = {
        'variable': None,
        'constant': {
            'num_games': int(1e5),
            'config': {'rows': 9, 'columns': 9, 'num_mines': 10},
            'seed': 50,
            'verbose': False,
            'visualise': False,  
        }
    }

    raise NotImplementedError("Need to link this experiment to database first")
    task_handler = None
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