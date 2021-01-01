import time
from datetime import timedelta
import os
import csv

import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data')
output_path_main = None
output_path_constants = None
all_results = []

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

def createTaskFromParameters(agent_parameters, other_parameters):
    solver_agent = NoUnnecessaryGuessSolver(**agent_parameters)
    
    method = minesweeper.run
    args = (solver_agent, )
    kwargs = other_parameters

    return (method, args, kwargs)

def createTasksFromSplitParameterGrid(parameter_grid):
    tasks = []

    for (i, (agent_parameters, other_parameters)) in enumerate(parameter_grid):
        task = createTaskFromParameters(agent_parameters, other_parameters)
        tasks.append(task)

    return tasks

def completeTask(task, callback):
    (method, args, kwargs) = task
    results = method(*args, **kwargs)
    callback(results, task)

def appendToFileName(name, suffix):
    (name, ext) = os.path.splitext(name)
    return name + suffix + ext

def onTaskCompleteStoreResultsInGlobal(results, task):
    global all_results

    _, args, kargs = task
    agent = args[0]

    # Append extra info to the task results before storing
    results['difficulty'] = configToDifficultyString(kargs['config'])
    results['sample_size'] = 'x'.join(str(num) for num in agent.SAMPLE_SIZE)    # represent sample size tuple (A, B) as string 'AxB'. Easier to parse from csv imo.
    results['use_num_mines_constraint'] = agent.use_num_mines_constraint

    all_results.append(results)

def saveExperimentResultsFromGlobalToCsv(experiment):
    global all_results
    output_file_name = experiment['title']
    saveDictRowsAsCsv(all_results, output_file_name)

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

def runExperiment(experiment, task_finish_callback, experiment_finish_callback):
    print("Preparing experiment '{}'...".format(experiment['title']), end="")

    # Prep
    parameter_grid, constants = getSplitParameterGridAndConstants(experiment)
    tasks = createTasksFromSplitParameterGrid(parameter_grid)
    
    # Save experiment info
    constants_output_file_name = appendToFileName(experiment['title'], "_other-data")
    saveDictRowsAsCsv([constants], constants_output_file_name)

    # Print start-of-experiment info
    num_combinations = len(parameter_grid)
    num_games = constants['num_games']
    print(" DONE")
    print("Now running {} games ({} overall) for {} different parameter combinations...".format(num_games, (num_games * num_combinations), num_combinations), end='')

    # Run experiment
    for task in tqdm(tasks, desc="Tasks complete", unit="task"):
        completeTask(task, callback=task_finish_callback)

    # End of experiment
    print(" DONE")
    experiment_finish_callback(experiment)

def getExperimentOne():
    ''' Purpose: A short experiment (~1 day or less) to do the following main things, among others: 
                 1. Verify no-unecessary-guess solver works as intended by checking the solver gets the expected win rates for each difficulties (full grid, no guess).
                 2. Try out numerous sample sizes with/without mine count constraint so as to give an indication of what the win rates look like for them (to help choose
                    which sample sizes to test in a bigger solver experiment with many more games).
                 3. Provide measurements of how long it takes to play a certain number of games for specific parameter combos, allowing for a reasonable estimate
                    of the total run time to be made for any subsequent bigger solver experiments. '''
        
    title = "experiment 1 (the short one)"

    agent_parameters = {
        'variable': {
            'sample_size': [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), None],  # Sample size None means use full grid
            'use_num_mines_constraint': [True, False],
        },
        'constant': {
            'seed': 14,
            'first_click_pos': None,
        }
    }

    other_parameters = {
        'variable': {
            'config': [
                {'rows': 9, 'columns': 9, 'num_mines': 10},
                {'rows': 16, 'columns': 16, 'num_mines': 40},
                {'rows': 16, 'columns': 30, 'num_mines': 99}
            ],
        },
        'constant': {
            'num_games': 1000,
            'seed': 57,
            'verbose': False,
            'visualise': False,  
        }
    }

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters
    }

    return experiment

def getExperimentTwo():
    title = "experiment 2"

    agent_parameters = {
        'variable': {
            'first_click_pos': [None, (2, 2)]
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

    experiment = {
        'title': title,
        'agent_parameters': agent_parameters,
        'other_parameters': other_parameters
    }

    return experiment

if __name__ == '__main__':
    experiment = getExperimentOne()
    runExperiment(experiment, task_finish_callback=onTaskCompleteStoreResultsInGlobal, experiment_finish_callback=saveExperimentResultsFromGlobalToCsv)