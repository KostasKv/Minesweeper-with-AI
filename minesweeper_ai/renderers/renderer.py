from abc import ABC


class Renderer(ABC):
    def __init__(self, config, grid, agent):
        pass

    def getNextMove(self):
        pass

    def updateFromResult(self, result):
        pass

    def onEndOfGames(self):
        pass
