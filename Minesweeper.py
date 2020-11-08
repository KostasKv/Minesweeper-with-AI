import random
import pygame
from Renderer import PygameRenderer
from enum import Enum

class Cell:
    def __init__(self, is_mine=False):
        self.uncovered = False
        self.is_mine = is_mine
        self.is_flagged = False
        self.num_adjacent_mines = 0


class Game:
    def __init__(self, config):
        self.config = config
        try:
            throwExceptionOnInvalidConfig(config)
        except:
            print("Configuration is invalid!")
        
        self.reset()
        self.state = GameState.START


    def throwExceptionOnInvalidConfig(self):
        return 5


    def reset(self):
        self.createMinelessGrid()
        self.populateGridWithMines()
        self.markGridSquaresWithMineProximityCount()


    def createMinelessGrid(self):
        self.grid = []

        for _ in range(self.config['columns']):
            column = []

            for _ in range(self.config['rows']):
                square = Cell(is_mine=False)
                column.append(square)
            
            self.grid.append(column)


    def populateGridWithMines(self):
        total_slots = self.config['rows'] * self.config['columns']

        if self.config['num_mines'] > total_slots:
            raise Exception("Too many mines! Cannot fit {} mines into a {}x{} grid (which has only {} cells)"
        .format(self.config['num_mines'], self.config['rows'], self.config['columns'], total_slots))

        for _ in range(self.config['num_mines']):
            self.placeMineInRandomEmptySquare()   


    def placeMineInRandomEmptySquare(self):
        mine_placed = False

        # Keep trying random squares until an empty (non-mine) square is found
        while not mine_placed:
            x = random.randrange(0, self.config['columns'])
            y = random.randrange(0, self.config['rows'])

            if not self.grid[x][y].is_mine:
                self.grid[x][y].is_mine = True
                mine_placed = True


    def markGridSquaresWithMineProximityCount(self):
        for x in range(self.config['columns']):
            for y in range(self.config['rows']):
                self.grid[x][y].num_adjacent_mines = self.countAdjacentMines(x, y)


    def countAdjacentMines(self, x, y):
        adjacent_mines_count = 0
        adjacent_cells_coords = self.getAdjacentCellsCoords(x, y)

        for coords in adjacent_cells_coords:
            x, y = coords
            cell = self.grid[x][y]
            if cell.is_mine:
                adjacent_mines_count += 1

        return adjacent_mines_count


    def getAdjacentCellsCoords(self, x, y, exclude_diagonals=False):
        max_x = self.config['columns'] - 1
        max_y = self.config['rows'] - 1

        adjacent_cells = []

        for i in [-1, 0, 1]:
            new_x = x + i

            # Out of bounds, no cell exists there.
            if new_x < 0 or new_x > max_x:
                continue

            for j in [-1, 0, 1]:
                new_y = y + j

                # Out of bounds, no cell exists there.
                if new_y < 0 or new_y > max_y:
                    continue
                
                # We want adjacent cells, not the cell itself
                if new_x == x and new_y == y:
                    continue
                
                if exclude_diagonals and abs(i) == abs(j):
                    continue

                adjacent_cell = (new_x, new_y)
                adjacent_cells.append(adjacent_cell)

        return adjacent_cells
    

class Executor():
    def __init__(self, game, num_games):
        self.game = game
        self.num_games = num_games


    def getGameConfig(self):
        return self.game.config


    def getGrid(self):
        return self.game.grid


    def makeMove(self, x, y, toggle_flag):
        out_of_bounds = (x < 0) or (y < 0) or (x >= self.game.config['rows']) or (y >= self.game.config['columns'])

        if out_of_bounds or self.game.grid[x][y].uncovered:
            return (self.game.grid, GameState.ILLEGAL_MOVE)

        game_state = self.processMove(x, y, toggle_flag)

        if game_state = 
        return (self.game.grid, game_state)

    
    # Assumes input is a legal move
    def processMove(self, x, y, toggle_flag):
        if toggle_flag:
            self.game.grid[x][y].is_flagged = not self.game.grid[x][y].is_flagged
        else:
            if self.game.grid[x][y].is_mine:
                self.game.grid[x][y].uncovered = True
                return GameState.LOST
            
            self.uncover(x, y)

            if self.gameWon():
                return GameState.WON
        
        return GameState.PLAYING
    

    def uncover(self, initial_x, initial_y):
        cells_to_uncover = [(initial_x, initial_y)]
        
        while cells_to_uncover:
            x, y = cells_to_uncover.pop(0)
            cell = self.game.grid[x][y]
            cell.uncovered = True

            if cell.num_adjacent_mines == 0:
                adjacent_cells_coords = self.game.getAdjacentCellsCoords(x, y)

                for x, y in adjacent_cells_coords:
                    if self.game.grid[x][y].uncovered == False:
                        if not (x, y) in cells_to_uncover:
                            cells_to_uncover.append((x, y))
    

    def gameWon(self):
        for column in self.game.grid:
            for cell in column:
                # If any non-mine cell is still covered, then game has not yet been won
                if not cell.is_mine and not cell.uncovered:
                    return False

        return True

    
    def forceResetGame(self):
        self.game.reset()
        return self.game.grid, GameState.PLAYING
            
class GameState(Enum):
    WON = 1
    LOST = 2
    PLAYING = 3
    STARTING = 4
    ILLEGAL_MOVE = 5

    
def run(agent=None, config={'rows':16, 'columns':16, 'num_mines':6}, num_games=1, visualise=True):
    # If user is going to manually play, then they have to see the board.
    if not agent:
        visualise = True

    game = Game(config)
    executor = Executor(game, num_games)

    if visualise:
        renderer = PygameRenderer(executor)
        renderer.takeControl()
    else:
        print("No-visualise mode not implemented")
            

if __name__ == '__main__':
    run()