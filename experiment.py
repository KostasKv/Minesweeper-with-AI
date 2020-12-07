import time
import os
import csv

import pandas as pd
from sklearn.model_selection import ParameterGrid

from minesweeper_ai import minesweeper
from minesweeper_ai.agents.no_unnecessary_guess_solver import NoUnnecessaryGuessSolver

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data')

def validateFileName(output_file_name):
    try:
        assert(isinstance(output_file_name, str))
        assert(output_file_name.strip() != '')
    except:
        raise ValueError("Output file name must be a non-empty string.")

def saveResultsToCsv(results, file_name):
    ''' Results are saved in 'data' folder. '''
    
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

def runGames(parameter_grid, CONSTANTS, output_file_name):
    validateFileName(output_file_name)
    run_results = []

    for (i, parameters) in enumerate(parameter_grid, 1):
        # Setup
        if parameters['sample_size'] is None:
             # Use sample that includes whole grid + walls
            parameters['sample_size'] = (parameters['config']['rows'] + 2, parameters['config']['columns'] + 2)

        parameters_string = '\t'.join(str(key) + ': ' + str(value) for (key, value) in parameters.items())
        print("{}/{}:\t{}".format(i, len(parameter_grid), parameters_string))

        solver_agent = NoUnnecessaryGuessSolver(seed=CONSTANTS['AGENT_SEED'],
                                                sample_size=parameters['sample_size'],
                                                use_num_mines_constraint=parameters['use_num_mines_constraint'])

        # Play
        results = minesweeper.run(solver_agent,
                                    config=parameters['config'],
                                    visualise=CONSTANTS['VISUALISE'],
                                    verbose=CONSTANTS['VERBOSE'],
                                    num_games=CONSTANTS['NUM_GAMES'],
                                    seed=CONSTANTS['RUN_SEED'])

        run_result = {**parameters, **results}
        run_result = convertRunResultConfigToDifficultyString(run_result)
        run_results.append(run_result)

    # Save experiment results
    saveResultsToCsv(run_results, output_file_name)

    # Save constant parameters too
    constants_output = output_file_name.rstrip('.csv') + "_constant-parameters"
    saveResultsToCsv([constant_parameters], constants_output)
        

if __name__ == '__main__':
    OUTPUT_FILE_NAME = 'experiment_1000_games'

    constant_parameters = {
        'NUM_GAMES': 1000,
        'RUN_SEED': 57,
        'AGENT_SEED': 14,
        'VERBOSE': False,
        'VISUALISE': False,
    }

    variable_parameters = {
        'sample_size': [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), None],  # Sample size None means use full grid
        'use_num_mines_constraint': [True, False],
        'config': [{'rows': 9, 'columns': 9, 'num_mines': 10}, {'rows': 16, 'columns': 16, 'num_mines': 40}, {'rows': 16, 'columns': 30, 'num_mines': 99}],
    }
    parameter_grid = list(ParameterGrid(variable_parameters))

    num_combinations = len(parameter_grid)
    num_games = constant_parameters['NUM_GAMES']
    print("Running {} games ({} total) for {} different parameter combinations...".format(num_games, (num_games * num_combinations), num_combinations))
    runGames(parameter_grid, constant_parameters, OUTPUT_FILE_NAME)
    print("\nFINISHED")
    