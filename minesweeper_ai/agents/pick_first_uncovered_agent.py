from itertools import chain
from random import randint

from .agent import Agent


class PickFirstUncoveredAgent(Agent):
    def nextMove(self):
        for tile in self.tiles_not_checked:
            if not tile.uncovered and not tile.is_mine:
                return (tile.x, tile.y, False)

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

    def onGameBegin(self):
        # Flatten grid to a 1D list
        self.tiles_not_checked = list(chain.from_iterable(self.grid))

    def highlightTiles(self):
        pass
