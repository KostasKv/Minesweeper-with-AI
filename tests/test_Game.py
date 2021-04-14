import unittest
import json
import os
from collections import namedtuple
from MinesweeperAI.Game import Game


class TestGameConfigurationValidation(unittest.TestCase):
    def test_raiseErrorOnInvalidConfig_valid_configurations(self):
        valid_configurations = [
            # Beginner, Intermediate, Expert difficulties
            {"rows": 8, "columns": 8, "num_mines": 10},
            {"rows": 16, "columns": 16, "num_mines": 40},
            {"rows": 16, "columns": 32, "num_mines": 99},
            # Large and small
            {"rows": 50, "columns": 100, "num_mines": 1000},
            {"rows": 2, "columns": 2, "num_mines": 1},
            # Using max mines possible for grid size. Note max mines is (rows-1)*(columns-1).
            {"rows": 3, "columns": 2, "num_mines": 2},
            {"rows": 2, "columns": 3, "num_mines": 2},
            {"rows": 16, "columns": 16, "num_mines": 225},
            {"rows": 8, "columns": 8, "num_mines": 49},
            {"rows": 16, "columns": 32, "num_mines": 465},
            {"rows": 50, "columns": 100, "num_mines": 4851},
        ]

        for config in valid_configurations:
            with self.subTest():
                try:
                    Game.raiseErrorOnInvalidConfig(config)
                except (TypeError, KeyError, ValueError):
                    self.fail(
                        "Incorrectly raised exception on valid configuration {}".format(
                            config
                        )
                    )

    def test_raiseErrorOnInvalidConfig_invalid_configurations(self):
        invalid_configurations = [
            # Non-dict
            5,
            [5],
            # Wrong dict structure
            {},
            {"ro": 16, "colum": 16, "num_mi": 10},
            {"columns": 16, "num_mines": 10},
            {"rows": 16, "num_mines": 10},
            {"num_mines": 10},
            # Zero value present
            {"rows": 0, "columns": 16, "num_mines": 10},
            {"rows": 16, "columns": 0, "num_mines": 10},
            {"rows": 16, "columns": 16, "num_mines": 0},
            # Negative value present
            {"rows": -16, "columns": 16, "num_mines": 10},
            {"rows": 16, "columns": -16, "num_mines": 10},
            {"rows": 16, "columns": 16, "num_mines": -10},
            # Too many mines for grid size. Note max mines is (rows-1)*(columns-1).
            {"rows": 16, "columns": 16, "num_mines": 226},
            {"rows": 8, "columns": 8, "num_mines": 50},
            {"rows": 16, "columns": 32, "num_mines": 466},
            {"rows": 1, "columns": 1, "num_mines": 1},
            {"rows": 2, "columns": 1, "num_mines": 1},
            {"rows": 1, "columns": 2, "num_mines": 1},
            {"rows": 2, "columns": 3, "num_mines": 3},
            {"rows": 50, "columns": 100, "num_mines": 4852},
        ]

        for config in invalid_configurations:
            with self.subTest():
                with self.assertRaises(
                    (TypeError, ValueError, KeyError),
                    msg="Failed on invalid config: {}".format(config),
                ):
                    Game.raiseErrorOnInvalidConfig(config)


class TestGameMethods(unittest.TestCase):
    def setUp(self):
        config_beginner = {"rows": 8, "columns": 8, "num_mines": 10}
        self.game = Game(config_beginner, seed=57)

    """
        Many of the tests in this class make an assumption as to what
        the game fixture's grid looks like. This makes sure that the fixture
        is still in that state that the other tests assume its in.
        If this fails then it's likely many other tests fail too.
    """

    def testSeededGameFixtureIsWhatIsExpected(self):
        # Check all fields have their expected values
        expected_config = {"rows": 8, "columns": 8, "num_mines": 10}
        self.assertEqual(self.game.config, expected_config)
        self.assertEqual(self.game.state, Game.State.START)
        self.assertEqual(self.game.mines_left, expected_config["num_mines"])

        # These are the values of each grid tile when using the
        # seed 57 for Game. An asterisk '*' represents a mine.
        expected_grid_values = """000001*1
                                  01121211
                                  12*2*100
                                  *2121100
                                  23110000
                                  *2*21211
                                  2322*3*1
                                  1*112*21""".split()

        grid_is_as_expected = self.isGridEqualToExpectedGridValuesAndAllTilesAreCovered(
            self.game.grid, expected_grid_values
        )
        self.assertTrue(grid_is_as_expected)

    def test_generateNewGrid(self):
        self.game.generateNewGrid()

        # Expected first new grid when feeding Game object seed 57.
        # Grid has been manually verified to be a valid minesweeper game.
        expected_new_grid = """0001*11*
                               00011222
                               001111*1
                               001*1111
                               11111111
                               *21222*1
                               *21**222
                               1112211*""".split()

        new_grid_is_as_expected = (
            self.isGridEqualToExpectedGridValuesAndAllTilesAreCovered(
                self.game.grid, expected_new_grid
            )
        )
        self.assertTrue(new_grid_is_as_expected)

    @staticmethod
    def isGridEqualToExpectedGridValuesAndAllTilesAreCovered(
        grid, expected_grid_values
    ):
        rows = len(grid)
        columns = len(grid[0])

        for i in range(rows):
            for j in range(columns):
                tile = grid[i][j]

                if tile.is_mine:
                    tile_value = "*"
                else:
                    tile_value = str(grid[i][j].num_adjacent_mines)

                expected_tile_value = expected_grid_values[i][j]

                if tile_value != expected_tile_value or tile.uncovered:
                    return False

        return True

    def test_createMinelessGrid(self):
        self.game.createMinelessGrid()

        # Grid size check
        rows = len(self.game.grid)
        columns = len(self.game.grid[0])
        self.assertEqual(rows, self.game.config["rows"])
        self.assertEqual(columns, self.game.config["columns"])

        # Ensure cells have no mines
        for i in range(self.game.config["rows"]):
            for j in range(self.game.config["columns"]):
                self.assertFalse(self.game.grid[i][j].is_mine)

    def test_populateGridWithMines(self):
        self.game.createMinelessGrid()
        self.game.populateGridWithMines()

        mine_count = 0

        for i in range(self.game.config["rows"]):
            for j in range(self.game.config["columns"]):
                if self.game.grid[i][j].is_mine:
                    mine_count += 1

        self.assertEqual(self.game.config["num_mines"], mine_count)
