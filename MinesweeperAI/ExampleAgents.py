from Agent import Agent
from random import randint


class RandomAgent(Agent):
    def nextMove(self):
        x = randint(0, len(self.grid) - 1)
        y = randint(0, len(self.grid[0]) - 1)
        return (x, y, False)

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

    def onGameReset(self):
        print("onGameReset")
        pass


class RandomLegalMovesAgent(Agent):
    def nextMove(self):
        x = randint(0, len(self.grid) - 1)
        y = randint(0, len(self.grid[0]) - 1)

        while not self.isLegal(x, y):
            x = randint(0, len(self.grid) - 1)
            y = randint(0, len(self.grid[0]) - 1)

        return (x, y, False)

    def isLegal(self, x, y):
        toggle_flag = False

        # Out of bounds
        if x < 0 or y < 0 or x > len(self.grid) or y > len(self.grid[0]):
            return False
        
        # Tile already uncovered
        if self.grid[x][y].uncovered:
            return False
        
        # Can't uncover a flagged tile
        if not toggle_flag and self.grid[x][y].is_flagged:
            return False

        return True

    def update(self, grid, mines_left, game_state):
        self.grid = grid
        self.mines_left = mines_left
        self.game_state = game_state

    def onGameReset(self):
        print("onGameReset")
        pass