from ortools.sat.python import cp_model
from copy import deepcopy

class CpSolver():
    def searchForDefiniteSolutions(self, adjacent_mines_constraints, sure_moves, num_unknown_tiles_outside=None, num_unknown_tiles_inside=None, frontier_tile_is_inside_sample=None, total_mines_left=None, num_tiles_outside_sample=None, outside_flagged=0):
        if total_mines_left is not None and num_tiles_outside_sample is not None and frontier_tile_is_inside_sample is not None and num_unknown_tiles_outside is not None and num_unknown_tiles_inside is not None:
            num_frontier = len(frontier_tile_is_inside_sample)
            (model, variables) = self.getModelWithAllBoardConstraints(num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles_outside, num_unknown_tiles_inside, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample, outside_flagged)
        elif adjacent_mines_constraints:
            num_frontier = len(adjacent_mines_constraints[0]) - 1
            (model, variables) = self.getModelWithAdjacentConstraints(num_frontier, adjacent_mines_constraints, sure_moves)
        else:
            return []   # No adjacent mine constraints given, and not enough information for total-mines-left constraint.)

        return self.searchForDefiniteSolutionsUsingModelAndItsVars(model, variables, num_frontier)

    def searchForDefiniteSolutionsUsingModelAndItsVars(self, model, variables, num_frontier):
        '''Expects variables to be sorted such that all frontier variables are at the beginning of the variables list'''
        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.INFEASIBLE:
            raise ValueError("Model provided does not represent the board correctly; not a single solution was found for it.")
        
        frontier_vars = variables[:num_frontier]

        test_vars = frontier_vars
        # Only test frontier tiles and a single unknown (non-frontier) tile inside sample.
        
        for v in variables[num_frontier:]:
            if v.Name().startswith("unknown_inside"):
                unknown_inside_var = v
                test_vars.append(unknown_inside_var)
                break
        else:
            unknown_inside_var = None

        
        potential_definites = [solver.BooleanValue(v) for v in test_vars]

        # unknown_inside_potential_definite = solver.BooleanValue(unknown_inside_var) if unknown_inside_var else None

        # Test each frontier variable using an opposite assignment from that in the first solution.
        # If there doesn't exist a feasible solution using the opposite assignment, we know
        # that variable must have the assignment found in the first solution
        # (since it has that same value for all possible enumerations).
        for definite_index in range(len(potential_definites)):
            if potential_definites[definite_index] is None:
                continue
            
            var = test_vars[definite_index]

            # Skip if it's a constant. We already know its definite value, no need to flip its value and exhaust
            # the search space just to prove the opposite value is infeasible, which is already known.
            if var.Name() == '':
                continue

            # Put in opposite-val constraint on variable we're testing. If that's infeasible then we know the variable can only
            # ever be assigned its already discovered value (it's a definite value), otherwise it's indefinite.
            opp_val = not potential_definites[definite_index]
            test_model = deepcopy(model)
            test_model.Add(var == opp_val)

            status = solver.Solve(test_model)

            if status != cp_model.INFEASIBLE:
                solution = [solver.BooleanValue(v) for v in frontier_vars]

                for i in range(definite_index, len(solution)):
                    # There exists two valid enumerations where the variable at index
                    # i is True in one and False in the other; it's not a definite solution.
                    if potential_definites[i] != solution[i]:
                        potential_definites[i] = None
                
                # # Update unknown inside var potential definite from solution too, if there is such a variable.
                # if unknown_inside_var is not None and unknown_inside_potential_definite != solver.BooleanValue(unknown_inside_var):
                #     unknown_inside_potential_definite = None

        
        if unknown_inside_var is not None:
            unknown_inside_var_definite_solution = potential_definites.pop()
        else:
            unknown_inside_var_definite_solution = None

        frontier_definite_solutions = [(i, x) for (i, x) in enumerate(potential_definites) if x is not None]

        return (frontier_definite_solutions, unknown_inside_var_definite_solution)


    def getModelWithAllBoardConstraints(self, num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles_outside, num_unknown_tiles_inside, frontier_tile_is_inside_sample, total_mines_left, num_tiles_outside_sample, outside_flagged=0):
        ''' Create a model built with all useful constraints possible for minesweeper using the information given.
            Assumes adjacent mine constraint coefficients are either 1 or 0. '''
        (model, variables) = self.getModelWithAdjacentConstraints(num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles_outside, num_unknown_tiles_inside)

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
    
    def getModelWithAdjacentConstraints(self, num_frontier, adjacent_mines_constraints, sure_moves, num_unknown_tiles_outside=None, num_unknown_tiles_inside=None):
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
                var = model.NewBoolVar(f'frontier{i}') # Could be either. Variables are named so as to differentiate them from constants.

            variables.append(var)

        if num_unknown_tiles_outside is not None and num_unknown_tiles_inside is not None:
            # Add extra variables for any extra unknown tiles which aren't on the frontier (they can't have adjacent constraints but need to be included
            # in total mines constraint, and so the variables must be included in model if using total mines constraint). 
            variables.extend(model.NewBoolVar(f'unknown_outside{i}') for i in range(num_unknown_tiles_outside))
            variables.extend(model.NewBoolVar(f'unknown_inside{i}') for i in range(num_unknown_tiles_inside))

        # # Create the variables
        # variables = [model.NewBoolVar(str(i)) for i in range(num_frontier)]

        # Create adjacent mines constraints
        for row in adjacent_mines_constraints:
            x = [variables[i] for i in range(len(row) - 1) if row[i] != 0]
            sum_value = row[-1]
            model.Add(sum(x) == sum_value)

        return (model, variables)
    
