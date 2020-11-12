from Renderer import Renderer
from Game import Game

class NoScreenRenderer(Renderer):
    def __init__(self, config, grid, agent):
        self.config = config
        self.grid = grid
        self.agent = agent
        self.game_state = Game.State.START
        
        self.initialiseAgent()


    def initialiseAgent(self):
        mines_left = self.config['num_mines']
        self.agent.update(self.grid, mines_left, self.game_state)


    def getNextMoveFromAgent(self):
        if self.game_state in [Game.State.WIN, Game.State.LOSE, Game.State.ILLEGAL_MOVE]:
            action = -1     # End of game so signal for game reset
        else:
            action = self.agent.nextMove()
        
        return action

    
    def updateFromResult(self, result):
        self.game_state = result[2]
        self.agent.update(*result)


    def onEndOfGames(self):
        print("All games completed!")
