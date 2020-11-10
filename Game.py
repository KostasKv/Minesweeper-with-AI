import random
from enum import Enum

class Cell:
    def __init__(self, pos, is_mine=False):
        self.x = pos[0]
        self.y = pos[1]
        self.is_mine = is_mine
        self.uncovered = False
        self.is_flagged = False
        self.num_adjacent_mines = 0


class Game:
    def __init__(self, config):
        # Dict consisting of keys: 'rows', 'columns', and 'num_mines'
        self.config = config    

        try:
            self.throwExceptionOnInvalidConfig(config)
        except:
            print("Configuration is invalid!")
        
        self.state = None
        self.mines_left = None
        self.reset()


    def throwExceptionOnInvalidConfig(self, config):
        pass


    def reset(self):
        self.createMinelessGrid()
        self.populateGridWithMines()
        self.markGridSquaresWithMineProximityCount()

        self.state = self.State.START
        self.mines_left = self.config['num_mines']
        

    def createMinelessGrid(self):
        self.grid = []

        for y in range(self.config['rows']):
            row = []

            for x in range(self.config['columns']):
                pos = (x, y)
                square = Cell(pos, is_mine=False)
                row.append(square)
            
            self.grid.append(row)


    '''
        Using convention that max number of mines on a QxP grid is (Q-1)(P-1),
        in accordance to custom ranking rules on http://www.minesweeper.info/custom.php
        and Window's minesweeper implementation http://www.minesweeper.info/wiki/Windows_Minesweeper#Trivia
    '''
    def populateGridWithMines(self):
        max_mines = (self.config['rows'] - 1) * (self.config['columns'] - 1)

        if self.config['num_mines'] > max_mines:
            raise Exception("Too many mines! {0} mines exceeds limit of {1} for a {2}x{3} grid ({2}x{3}={1})"
        .format(self.config['num_mines'], max_mines, self.config['rows'], self.config['columns'], max_mines))

        for _ in range(self.config['num_mines']):
            self.placeMineInRandomEmptySquare()   


    def placeMineInRandomEmptySquare(self):
        mine_placed = False

        # Keep trying random squares until an empty (non-mine) square is found
        while not mine_placed:
            x = random.randrange(0, self.config['columns'])
            y = random.randrange(0, self.config['rows'])

            if not self.grid[y][x].is_mine:
                self.grid[y][x].is_mine = True
                mine_placed = True


    def markGridSquaresWithMineProximityCount(self):
        for x in range(self.config['columns']):
            for y in range(self.config['rows']):
                self.grid[y][x].num_adjacent_mines = self.countAdjacentMines(x, y)


    def countAdjacentMines(self, x, y):
        adjacent_mines_count = 0
        adjacent_cells_coords = self.getAdjacentCellsCoords(x, y)

        for coords in adjacent_cells_coords:
            x, y = coords
            cell = self.grid[y][x]

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
        

    class State(Enum):
        START = 1
        PLAY = 2
        WIN = 3
        LOSE = 4
        ILLEGAL_MOVE = 5