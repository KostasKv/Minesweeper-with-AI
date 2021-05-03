from abc import ABC


class Agent(ABC):
    def nextMove(self):
        pass

    def update(self, grid, mines_left, game_state):
        pass

    def onGameBegin(self):
        pass

    def highlightTiles(self):
        pass
