from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ortools.sat.python import cp_model
from copy import deepcopy

class CpSolver():
    class SolutionTracker(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.__variables = variables
            self.__solution_count = 0
            self.count = [0] * len(variables)

        def on_solution_callback(self):
            self.__solution_count += 1

            for (i, v) in enumerate(self.__variables):
                self.count[i] += self.Value(v)
            


        def result(self):
            # print(self.__solution_count, self.count)
            definite_solutions = []

            for (i, x) in enumerate(self.count):
                if x == 0:
                    definite_solutions.append((i, False))
                elif x == self.__solution_count:
                    definite_solutions.append((i, True))

            return definite_solutions

    # class StopOnFirstSolution(cp_model.CpSolverSolutionCallback):
    #     def __init__(self):
    #         cp_model.CpSolverSolutionCallback.__init__(self)

    #     def on_solution_callback(self):
    #         self.StopSearch()


    def searchForDefiniteSolutions(self, adjacent_mines_constraints, total_mines_constraint):
        methods_to_use = [3, 4]

        results = []

        for i in methods_to_use:
            method = getattr(self, "searchForDefiniteSolutions{}".format(i))
            result = method(adjacent_mines_constraints, total_mines_constraint)
            results.append(result)

        # All results are expected to be the same!
        assert(all(x == results[0] for x in results))
        
        return results[0]
        

    def searchForDefiniteSolutions1(self, matrix_row_constraints, total_mines_constraint):
        if not matrix_row_constraints:
            return []

        """Showcases calling the solver to search for all solutions."""
        # Creates the model.
        model = cp_model.CpModel()

        # Create the variables
        variables = [model.NewBoolVar(str(i)) for i in range(len(matrix_row_constraints[0]) - 1)]

        # c = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]

        # Add the adjacent-mines constraints.
        for constraint in matrix_row_constraints:
            x = [int(constraint[i]) * variables[i] for i in range(len(constraint) - 1) if constraint[i] != 0]
            sum_value = constraint[-1]
            model.Add(sum(x) == sum_value)
        
        constraint = total_mines_constraint[:-1]
        total_mines_left = total_mines_constraint[-1]

        # Sample has unknown proportion of total covered unflagged tiles. Therefore total mines left can only
        # give an absolute upper limit on the amount of mines that there can within the sample's covered tiles.
        x = [int(constraint[i]) * variables[i] for i in range(len(constraint) - 1) if constraint[i] != 0]
        model.Add(sum(x) <= total_mines_left)

        # Create a solver and solve.
        solver = cp_model.CpSolver()
        solution_tracker = self.SolutionTracker(variables)
        status = solver.SearchForAllSolutions(model, solution_tracker)

        return solution_tracker.result()

    def searchForDefiniteSolutions2(self, matrix_row_constraints, mines_left_in_entire_board):
        if not matrix_row_constraints:
            return []

        constraints, variables = self.getConstraintsAndVarsFromMatrixRows(matrix_row_constraints, mines_left_in_entire_board)

        model = self.getModelFromConstraintsAndVars(constraints, variables)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        potential_definites = []

        for v in variables:
            potential_definites.append(solver.BooleanValue(v))
        # potential_definites = [solver.BooleanValue(v) for v in variables]
        
        for definite_index in range(len(potential_definites)):
            if potential_definites[definite_index] is None:
                continue

            # Put in opposite-val constraint on variable we're testing.
            var = variables[definite_index]
            opp_val = not potential_definites[definite_index]
            model = self.getModelFromConstraintsAndVars(constraints, variables)
            model.Add(var == opp_val)

            # Solve using that constraint
            status = solver.Solve(model)

            if status != cp_model.INFEASIBLE:
                solution = [solver.BooleanValue(v) for v in variables]

                for (i, x) in enumerate(solution):
                    # If two valid enumerations found where a variable has
                    # different assignments, then it's not a definite solution.
                    if potential_definites[i] != x:
                        potential_definites[i] = None

        return [(i, x) for (i, x) in enumerate(potential_definites) if x is not None]

    def searchForDefiniteSolutions3(self, matrix_row_constraints, total_mines_constraint):
        if not matrix_row_constraints:
            return []

        (proper_model, variables) = self.getModelFromMatrixRows(matrix_row_constraints, total_mines_constraint)

        solver = cp_model.CpSolver()
        status = solver.Solve(proper_model)
        
        potential_definites = []

        for v in variables:
            potential_definites.append(solver.BooleanValue(v))
        # potential_definites = [solver.BooleanValue(v) for v in variables]
        
        for definite_index in range(len(potential_definites)):
            if potential_definites[definite_index] is None:
                continue

            # Put in opposite-val constraint on variable we're testing.
            var = variables[definite_index]
            opp_val = not potential_definites[definite_index]
            test_model = deepcopy(proper_model)
            test_model.Add(var == opp_val)

            # Solve using that constraint
            status = solver.Solve(test_model)

            if status != cp_model.INFEASIBLE:
                solution = [solver.BooleanValue(v) for v in variables]

                for (i, x) in enumerate(solution):
                    # If two valid enumerations found where a variable has
                    # different assignments, then it's not a definite solution.
                    if potential_definites[i] != x:
                        potential_definites[i] = None

        return [(i, x) for (i, x) in enumerate(potential_definites) if x is not None]

    def searchForDefiniteSolutions4(self, matrix_row_constraints, mines_left_in_entire_board):
        if not matrix_row_constraints:
            return []

        (proper_model, variables) = self.getModelFromMatrixRows(matrix_row_constraints, mines_left_in_entire_board)

        solver = cp_model.CpSolver()
        status = solver.Solve(proper_model)
        
        potential_definites = []

        for v in variables:
            potential_definites.append(solver.BooleanValue(v))
        # potential_definites = [solver.BooleanValue(v) for v in variables]
        
        for definite_index in range(len(potential_definites)):
            if potential_definites[definite_index] is None:
                continue

            # Put in opposite-val constraint on variable we're testing.
            var = variables[definite_index]
            opp_val = not potential_definites[definite_index]
            test_model = deepcopy(proper_model)
            test_model.Add(var == opp_val)

            # Solve using that constraint
            status = solver.Solve(test_model)

            if status == cp_model.INFEASIBLE:
                # Don't try adding constraint if no more tests will be done (Add() is an expensive call)
                if any(potential_definites[i] is not None for i in range(definite_index + 1, len(potential_definites))):
                    # Proved definite solution is correct. Keep that constraint in all models from now
                    proper_model.Add(var == potential_definites[definite_index])
                else:
                    break
            else:
                solution = [solver.BooleanValue(v) for v in variables]

                for (i, x) in enumerate(solution):
                    # If two valid enumerations found where a variable has
                    # different assignments, then it's not a definite solution.
                    if potential_definites[i] != x:
                        potential_definites[i] = None

        return [(i, x) for (i, x) in enumerate(potential_definites) if x is not None]

    def getConstraintsAndVarsFromMatrixRows(self, matrix_rows, total_mines_constraint):
        model = cp_model.CpModel()
        
        # Create the variables
        variables = [model.NewBoolVar(str(i)) for i in range(len(matrix_rows[0]) - 1)]
        constraints = []

        # Create the constraints.
        for row in matrix_rows:
            x = [int(row[i]) * variables[i] for i in range(len(row) - 1) if row[i] != 0]
            sum_value = row[-1]
            constraint = cp_model.BoundedLinearExpression(sum(x), (sum_value, sum_value))
            constraints.append(constraint)
        
        constraint = total_mines_constraint[:-1]
        total_mines_left = total_mines_constraint[-1]

        # Sample has unknown proportion of total covered unflagged tiles. Therefore total mines left can only
        # give an absolute upper limit on the amount of mines that there can within the sample's covered tiles.
        x = [int(constraint[i]) * variables[i] for i in range(len(constraint) - 1) if constraint[i] != 0]
        model.Add(sum(x) <= total_mines_left)

        return (constraints, variables)

    def getModelFromConstraintsAndVars(self, constraints, variables):
        model = cp_model.CpModel()

        for i in range(len(variables)):
            model.NewBoolVar(str(i))

        for constraint in constraints:
            model.Add(constraint)

        return model, variables

    def getModelFromMatrixRows(self, matrix_rows, total_mines_constraint):
        model = cp_model.CpModel()
        
        # Create the variables
        variables = [model.NewBoolVar(str(i)) for i in range(len(matrix_rows[0]) - 1)]
        constraints = []

        # Create the constraints.
        for row in matrix_rows:
            x = [int(row[i]) * variables[i] for i in range(len(row) - 1) if row[i] != 0]
            sum_value = row[-1]
            model.Add(sum(x) == sum_value)

        constraint = total_mines_constraint[:-1]
        total_mines_left = total_mines_constraint[-1]

        # Sample has unknown proportion of total covered unflagged tiles. Therefore total mines left can only
        # give an absolute upper limit on the amount of mines that there can within the sample's covered tiles.
        x = [int(constraint[i]) * variables[i] for i in range(len(constraint) - 1) if constraint[i] != 0]
        model.Add(sum(x) <= total_mines_left)

        return (model, variables)

if __name__ == '__main__':
    c = [[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 2],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, -1, -1, -1, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, -1, 1, 0, 0]]

    solver = CpSolver()
    answers = solver.searchForDefiniteSolutionsUsingCpSolver(c)
    print(answers)