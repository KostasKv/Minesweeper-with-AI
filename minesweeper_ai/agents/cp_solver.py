from ortools.sat.python import cp_model
from copy import deepcopy

class CpSolver():
    def searchForDefiniteSolutions(self, adjacent_mines_constraints, frontier_tile_is_inside_sample=None, total_mines_left=None, num_tiles_outside_sample=None, outside_flagged=None):
        if total_mines_left is not None and num_tiles_outside_sample is not None and frontier_tile_is_inside_sample is not None:
            num_frontier = len(frontier_tile_is_inside_sample)
            (model, variables) = self.getModelWithAllBoardConstraints(num_frontier, adjacent_mines_constraints, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample, outside_flagged)
        elif adjacent_mines_constraints:
            num_frontier = len(adjacent_mines_constraints[0]) - 1
            (model, variables) = self.getModelWithAdjacentConstraints(num_frontier, adjacent_mines_constraints)
        else:
            return []   # No adjacent mine constraints given, and not enough information for total-mines-left constraint.)

        return self.searchForDefiniteSolutionsUsingModelAndItsVars(model, variables)

    def searchForDefiniteSolutionsUsingModelAndItsVars(self, model, variables):
        # Get first solution (there should always be atleast one as constraints given are based off of
        # a valid board. No need to check if status is infeasible)
        solver = cp_model.CpSolver()
        solver.Solve(model)
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


    def getModelWithAllBoardConstraints(self, num_frontier, adjacent_mines_constraints, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample, outside_flagged=None):
        ''' Create a model built with all useful constraints possible for minesweeper using the information given.
            Assumes adjacent mine constraint coefficients are either 1 or 0. '''
        (model, variables) = self.getModelWithAdjacentConstraints(num_frontier, adjacent_mines_constraints)

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
        min_mines = max(0, min_mines)   # Min mines can't be below 0

        if outside_flagged is None:
            model.Add(min_mines - sum(v_outside) <= sum(v_inside) <= total_mines_left)
        else:
            model.Add(min_mines - sum(v_outside) - outside_flagged <= sum(v_inside) <= total_mines_left)

        return (model, variables)
    
    def getModelWithAdjacentConstraints(self, num_frontier, adjacent_mines_constraints):
        ''' Returns a model built with adjacent mine constraints.
            Assumes adjacent mine constraint coefficients are either 1 or 0. '''
        model = cp_model.CpModel()
        
        # Create the variables
        variables = [model.NewBoolVar(str(i)) for i in range(num_frontier)]

        # Create adjacent mines constraints
        for row in adjacent_mines_constraints:
            x = [variables[i] for i in range(len(row) - 1) if row[i] != 0]
            sum_value = row[-1]
            model.Add(sum(x) == sum_value)

        return (model, variables)
    
