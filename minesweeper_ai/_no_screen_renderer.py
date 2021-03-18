from .renderer import Renderer
from ._game import _Game

class NoScreenRenderer(Renderer):
    def __init__(self, config, grid, agent, game_seed):
        self.config = config
        self.grid = grid
        self.agent = agent
        self.game_state = _Game.State.START
        
        self.initialiseAgent(game_seed)


    def initialiseAgent(self, game_seed):
        self.agent.update(self.grid, self.config['num_mines'], self.game_state)
        self.agent.onGameBegin(game_seed)


    def getNextMove(self):
        if self.game_state in [_Game.State.PLAY, _Game.State.START]:
            return self.agent.nextMove()
        else:
            return -1     # End of game so signal for game reset

    
    def updateFromResult(self, result, game_seed):
        self.game_state = result[2]
        self.agent.update(*result)

        if self.game_state == _Game.State.START:
            self.agent.onGameBegin(game_seed)


    def onEndOfGames(self):
        ''' Return game statistics '''
        return self.agent.get_stats()
