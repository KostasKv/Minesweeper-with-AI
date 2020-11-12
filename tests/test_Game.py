import os
print("current working directory: {}".format(os.getcwd()), end='\n\n')
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print('\n'.join(sys.path))
import unittest
from MinesweeperAI.Game import Game



class TestConfigurationValidation(unittest.TestCase):
    def test_valid_configuration(self):
        valid_configurations = [
            # Beginner, Intermediate, Expert difficulties
            {'rows': 16, 'columns': 16, 'num_mines': 40},
            {'rows': 8, 'columns': 8, 'num_mines': 10},
            {'rows': 16, 'columns': 32, 'num_mines': 99},

            # Large and small
            {'rows': 50, 'columns': 100, 'num_mines': 1000},
            {'rows': 2, 'columns': 2, 'num_mines': 1},

            # Using max mines possible for grid size. Note max mines is (rows-1)*(columns-1).
            {'rows': 3, 'columns': 2, 'num_mines': 2},
            {'rows': 2, 'columns': 3, 'num_mines': 2},
            {'rows': 16, 'columns': 16, 'num_mines': 225},
            {'rows': 8, 'columns': 8, 'num_mines': 49},
            {'rows': 16, 'columns': 32, 'num_mines': 465},
            {'rows': 50, 'columns': 100, 'num_mines': 4851}
        ]

        raised = False

        try:
            for config in valid_configurations:
                Game.throwExceptionOnInvalidConfig(config)
        except:
            raised = True

        self.assertFalse(raised, msg="Incorrectly raised exception on valid configuration {}".format(config))


    def test_invalid_configurations(self):
        invalid_configurations = [
            # Non-dict
            5,
            [5],

            # Wrong dict structure
            {},
            {'ro': 16, 'colum': 16, 'num_mi': 10},
            {'columns': 16, 'num_mines': 10},
            {'rows': 16, 'num_mines': 10},
            {'num_mines': 10},

            # Zero value present
            {'rows': 0, 'columns': 16, 'num_mines': 10},
            {'rows': 16, 'columns': 0, 'num_mines': 10},
            {'rows': 16, 'columns': 16, 'num_mines': 0},

            # Negative value present
            {'rows': -16, 'columns': 16, 'num_mines': 10},
            {'rows': 16, 'columns': -16, 'num_mines': 10},
            {'rows': 16, 'columns': 16, 'num_mines': -10},

            # Too many mines for grid size. Note max mines is (rows-1)*(columns-1).
            {'rows': 16, 'columns': 16, 'num_mines': 226},
            {'rows': 8, 'columns': 8, 'num_mines': 50},
            {'rows': 16, 'columns': 32, 'num_mines': 466},
            {'rows': 1, 'columns': 1, 'num_mines': 1},
            {'rows': 2, 'columns': 1, 'num_mines': 1},
            {'rows': 1, 'columns': 2, 'num_mines': 1},
            {'rows': 2, 'columns': 3, 'num_mines': 3},
            {'rows': 50, 'columns': 100, 'num_mines': 4852},
        ]
        
        for config in invalid_configurations:
            with self.assertRaises(ValueError, msg="Failed on invalid config: {}".format(config)):
                Game.throwExceptionOnInvalidConfig(config)


if __name__ == "__main__":
    
    unittest.main()