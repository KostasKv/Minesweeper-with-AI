from ortools.sat.python import cp_model
from copy import deepcopy

class CpSolver():
    def searchForDefiniteSolutions(self, adjacent_mines_constraints, sure_moves, num_unknown_tiles=None, frontier_tile_is_inside_sample=None, total_mines_left=None, num_tiles_outside_sample=None, outside_flagged=0):
        if total_mines_left is not None and num_tiles_outside_sample is not None and frontier_tile_is_inside_sample is not None and num_unknown_tiles is not None:
            num_frontier = len(frontier_tile_is_inside_sample)
            (model, variables) = self.getModelWithAllBoardConstraints(num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample, outside_flagged)
        elif adjacent_mines_constraints:
            num_frontier = len(adjacent_mines_constraints[0]) - 1
            (model, variables) = self.getModelWithAdjacentConstraints(num_frontier, adjacent_mines_constraints, sure_moves)
        else:
            return []   # No adjacent mine constraints given, and not enough information for total-mines-left constraint.)

        # num_non_frontier = num_unknown_tiles - num_frontier
        return self.searchForDefiniteSolutionsUsingModelAndItsVars(model, variables)

    def searchForDefiniteSolutionsUsingModelAndItsVars(self, model, variables):
        solver = cp_model.CpSolver()
        solver.Solve(model)
        potential_definites = [solver.BooleanValue(v) for v in variables]
        
        # Test each frontier variable using an opposite assignment from that in the first solution.
        # If there doesn't exist a feasible solution using the opposite assignment, we know
        # that variable must have the assignment found in the first solution
        # (since it has that same value for all possible enumerations).
        for definite_index in range(len(potential_definites)):
            if potential_definites[definite_index] is None:
                continue
            
            # We know setting the constant to it's opposite bool value will be infeasible.
            # No need to exhaust that entire search space just to prove it.
            is_constant = variables[definite_index].Name() == ''
            # is_outside_frontier = variables[definite_index].Name()[0] == 'y'
            if is_constant:
                continue

            # Put in opposite-val constraint on variable we're testing.
            var = variables[definite_index]
            opp_val = not potential_definites[definite_index]
            test_model = deepcopy(model)
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


    def getModelWithAllBoardConstraints(self, num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample, outside_flagged=0):
        ''' Create a model built with all useful constraints possible for minesweeper using the information given.
            Assumes adjacent mine constraint coefficients are either 1 or 0. '''
        (model, variables) = self.getModelWithAdjacentConstraints(num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles)

        # Split variables into those that are inside the sample, and those that are outside it.
        v_inside = []
        v_outside = []
        for (i, is_inside_sample) in enumerate(frontier_tile_is_inside_sample):
            if is_inside_sample:
                v_inside.append(variables[i])
            else:
                v_outside.append(variables[i])

        # Create total mines constraint
        # min_mines = total_mines_left - num_tiles_outside_sample + len(v_outside)
        min_mines = total_mines_left - num_tiles_outside_sample
        min_mines = max(0, min_mines)   # Num mines that can be assigned obviously won't be below 0

        # model.Add(min_mines - sum(v_outside) - outside_flagged <= sum(v_inside) <= total_mines_left)
        model.Add(min_mines - sum(v_outside) <= sum(v_inside) <= total_mines_left)

        return (model, variables)
    
    def getModelWithAdjacentConstraints(self, num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles=None):
        ''' Returns a model built with adjacent mine constraints.
            Assumes adjacent mine constraint coefficients are either 1 or 0. '''
        model = cp_model.CpModel()
        
        variables = []

        for i in range(num_frontier):
            if (i, True) in sure_moves:
                var = model.NewConstant(1)  # Is mine
            elif (i, False) in sure_moves:
                var = model.NewConstant(0)  # Is clear
            else:
                var = model.NewBoolVar(f'x{i}') # Could be either. Variables are named so as to differentiate them from constants.

            variables.append(var)

        if num_unknown_tiles is not None:
            # Add extra variables for any extra unknown tiles which aren't on the frontier (they can't have adjacent constraints but need to be included
            # in total mines constraint, and so the variables must be included in model if using total mines constraint). 
            variables.extend(model.NewBoolVar(f'y{i}') for i in range(num_frontier, num_unknown_tiles))

        # # Create the variables
        # variables = [model.NewBoolVar(str(i)) for i in range(num_frontier)]

        # Create adjacent mines constraints
        for row in adjacent_mines_constraints:
            x = [variables[i] for i in range(len(row) - 1) if row[i] != 0]
            sum_value = row[-1]
            model.Add(sum(x) == sum_value)

        return (model, variables)
    
