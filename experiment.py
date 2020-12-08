import time
from datetime import timedelta
import os
import csv

import pandas as pd
from sklearn.model_selection import ParameterGrid

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data')
output_path_main = None
output_path_constants = None
all_results = []

def validateFileName(output_file_name):
    try:
        assert(isinstance(output_file_name, str))
        assert(output_file_name.strip() != '')
    except:
        raise ValueError("Output file name must be a non-empty string.")

def saveResultsToCsv(results, file_name):
    ''' Results are saved in 'data' folder. '''

    with open(path, 'w', newline='') as csvfile:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

def convertRunResultConfigToDifficultyString(result):
    config = result['config']

    if config['columns'] == 9 and config['rows'] == 9:
        difficulty = 'Beginner (9x9)'
    elif config['columns'] == 16 and config['rows'] == 16:
        difficulty = 'Intermediate (16x16)'
    elif config['columns'] == 30 and config['rows'] == 16:
        difficulty = 'Expert (16x30)'
    else:
        raise ValueError("Difficulty not recognised from config {}".format(config))

    # Replace config with difficulty string
    result.pop('config')
    result['difficulty'] = difficulty

    return result

def createTaskFromParameters(agent_parameters, other_parameters):
    solver_agent = NoUnnecessaryGuessSolver(**agent_parameters)
    
    method = minesweeper.run
    args = (solver_agent)
    kwargs = **other_parameters

    return (method, args, kwargs)

def createTasksFromSplitParameterGrid(parameter_grid):
    tasks = []

    for (i, (agent_parameters, other_parameters)) in enumerate(parameter_grid):
        if agent_parameters['sample_size'] is None:
             # Use sample that includes whole grid + walls
            agent_parameters['sample_size'] = (other_parameters['config']['rows'] + 2, other_parameters['config']['columns'] + 2)


    return tasks

def storeResultsOnTaskComplete(results, task):
    raise NotImplementedError

    global all_results


    

    run_result = {**parameters, **results}
    run_result = convertRunResultConfigToDifficultyString(run_result)

    # Print out progress and parameters
    parameters_string = '\t'.join(str(key) + ': ' + str(value) for (key, value) in agent_parameters.items() + other_parameters.items())
    print("{}/{}:\t{}".format(i, len(parameter_grid), parameters_string))

def onAllTasksComplete():
    # Save experiment results
    saveResultsToCsv(all_results, output_file_name)

def completeTask(task, callback=storeResultsOnTaskComplete):
    (method, args, kwargs) = task
    results = method(*args, **kwargs)
    callback(results, task)

def createOutputPathCSVFile(output_file_name):
    validateFileName(output_file_name)

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

def appendToFileName(name, suffix):
    (name, ext) = os.path.splitext(name)
    return name + suffix + ext

def runExperiment(parameter_grid, output_file_name):
    # Prep
    constants_output_file_name = appendToFileName(output_file_name, "_other-data")

    output_path_main = createOutputPathCSVFile(output_file_name)
    output_path_constants = createOutputPathCSVFile()
    
    tasks = createTasksFromSplitParameterGrid(parameter_grid)

    # Play
    for task in tasks:
        completeTask(task, callback=storeResultsOnTaskComplete)

    # End of experiment
    saveAllResultsToCsv()
    saveResultsToCsv(, constants_output)
    saveResultsToCsv([constant_parameters], constants_output)

def getExperimentOneParameters():
    title = "test"

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

def getExperimentTwoParameters():
    title = "experiment2"

    agent_parameters = {
        'variable': {
            'first_click_pos': [None, (2, 2)]
        },
        'constant': {
            'sample_size': None,  # Sample size None means use full grid
            'seed': 20,
            'first_click_pos': None,
            'use_num_mines_constraint': [True]
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

    constants = {**agent_constants, **other_constants}

    return (split_parameter_grid_with_constants, constants)


if __name__ == '__main__':
    # Get experiment
    experiment = getExperimentOneParameters()
    parameter_grid, constants = getSplitParameterGridAndConstants(experiment)

    # Print out start-of-experiment info
    num_combinations = len(parameter_grid)
    num_games = constants['num_games']
    print("Running {} games ({} total) for {} different parameter combinations...".format(num_games, (num_games * num_combinations), num_combinations))
    
    # Start experiment
    start = time.time()
    runExperiment(parameter_grid, experiment['title'])
    end = time.time()

    duration = timedelta(seconds=round(end - start))
    print(f"\nFinished in {duration} sec")
    