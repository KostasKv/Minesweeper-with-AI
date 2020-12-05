from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model
from copy import deepcopy

class CpSolver():
    def searchForDefiniteSolutions(self, adjacent_mines_constraints, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample):
        num_frontier = len(frontier_tile_is_inside_sample)
        (proper_model, variables) = self.getModelWithBoardConstraints(num_frontier, adjacent_mines_constraints, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample)
        
        # Get first solution
        solver = cp_model.CpSolver()
        status = solver.Solve(proper_model)
        potential_definites = [solver.BooleanValue(v) for v in variables]
        
        # Test each variable using an opposite assignment from that in the first solution.
        # If there doesn't exist a feasible solution using the opposite assignment, we know
        # that variable must have the assignment found in the first solution
        # (since it has that same value for all possible enumerations).
        for definite_index in range(len(potential_definites)):
            if potential_definites[definite_index] is None:
                continue

            # Put in opposite-val constraint on variable we're testing.
            var = variables[definite_index]
            opp_val = not potential_definites[definite_index]
            test_model = deepcopy(proper_model)
            test_model.Add(var == opp_val)

            status = solver.Solve(test_model)

            if status != cp_model.INFEASIBLE:
                solution = [solver.BooleanValue(v) for v in variables]

                for i in range(definite_index, len(solution)):
                    # There exists two valid enumerations where the variable at index
                    # i is True in one and False in the other; it's not a definite solution.
                    if potential_definites[i] != solution[i]:
                        potential_definites[i] = None

        return [(i, x) for (i, x) in enumerate(potential_definites) if x is not None]

    def getModelWithBoardConstraints(self, num_frontier, adjacent_mines_constraints, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample):
        ''' Create a model built with all useful constraints possible for minesweeper using the information given.
            Assumes adjacent mine constraint coefficients are either 1 or 0. '''
        model = cp_model.CpModel()
        
        # Create the variables
        variables = [model.NewBoolVar(str(i)) for i in range(num_frontier)]

        # Create adjacent mines constraints
        for row in adjacent_mines_constraints:
            x = [variables[i] for i in range(len(row) - 1) if row[i] != 0]
            sum_value = row[-1]
            model.Add(sum(x) == sum_value)
        
        # Split variables into those that are inside the sample, and those that are outside it.
        v_inside = []
        v_outside = []
        for (i, is_inside_sample) in enumerate(frontier_tile_is_inside_sample):
            if is_inside_sample:
                v_inside.append(variables[i])
            else:
                v_outside.append(variables[i])

        # Create total mines constraint
        min_mines = total_mines_left - num_tiles_outside_sample + len(v_outside)
        model.Add(min_mines - sum(v_outside) <= sum(v_inside) <= total_mines_left)

        return (model, variables)
