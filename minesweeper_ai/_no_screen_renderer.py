from .renderer import Renderer
from .game import Game

class NoScreenRenderer(Renderer):
    def __init__(self, config, grid, agent):
        self.config = config
        self.grid = grid
        self.agent = agent
        self.game_state = Game.State.START
        
        self.initialiseAgent()


    def initialiseAgent(self):
        self.agent.update(self.grid, self.config['num_mines'], self.game_state)
        self.agent.onGameBegin()


    def getNextMove(self):
        if self.game_state in [Game.State.PLAY, Game.State.START]:
            return self.agent.nextMove()
        else:
            return -1     # End of game so signal for game reset

    
    def updateFromResult(self, result):
        self.game_state = result[2]
        self.agent.update(*result)

        if self.game_state == Game.State.START:
            self.agent.onGameBegin()


    def onEndOfGames(self):
        ''' Return game statistics '''
        return {'samples_considered': self.agent.sample_count,
                'samples_with_solutions': self.agent.samples_with_solution_count}
