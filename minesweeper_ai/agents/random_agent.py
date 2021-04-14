from random import randint

from .agent import Agent


class RandomAgent(Agent):
    def nextMove(self):
        x = randint(0, len(self.grid[0]) - 1)
        y = randint(0, len(self.grid) - 1)
        return (x, y, False)

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

    def onGameBegin(self):
        pass

    def highlightTiles(self):
        pass
