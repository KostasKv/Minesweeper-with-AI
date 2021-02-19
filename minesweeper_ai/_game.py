from random import Random
from enum import Enum


class _Tile:
    def __init__(self, pos, is_mine=False):
        self.x = pos[0]
        self.y = pos[1]
        self.is_mine = is_mine
        self.uncovered = False
        self.is_flagged = False
        self.num_adjacent_mines = 0


class _Game:
    def __init__(self, config):
        self.config = config
        self.raiseErrorOnInvalidConfig(config) # Abruptly stops if invalid.
        self.state = None
        self.mines_left = None
        self.seed = None

    @staticmethod
    def raiseErrorOnInvalidConfig(config):
        if not isinstance(config, dict):
            raise TypeError("Game configuration must be a dict object.")
        
        # Check keys
        required_keys = ['rows', 'columns', 'num_mines']

        if any(key not in config for key in required_keys):
            raise KeyError("Game configuration needs the following keys: {}".format(', '.join(required_keys)))
        
        # # Check all values are positive
        # if any(value <= 0 for value in config.values()):
        #     raise ValueError("All game configuration values must be positive")

        # Check grid isn't too small, i.e. dimensions are Nx1 or 1xM
        if config['rows'] <= 1 or config ['columns'] <= 1:
            raise ValueError("Grid too small! It must have atleast 2 rows and 2 columns.")

        # Check num_mines doesn't exceed limit (X-1)(Y-1), where grid is of dimensions (X, Y).
        max_mines = (config['rows'] - 1) * (config['columns'] - 1)
        
        if config['num_mines'] > max_mines:
            template = "Too many mines! {0} mines exceeds limit of {1} for a {2}x{3} grid: ({2}-1)*({3}-1) = {1} mines max."
            message = template.format(config['num_mines'], max_mines, config['rows'], config['columns'], max_mines)
            raise ValueError(message)
        
    def newGame(self, seed):
        self.seed = seed
        self.state = self.State.START
        self.mines_left = self.config['num_mines']
        self._createNewEmptyHiddenGrid()

    def _createNewEmptyHiddenGrid(self):
        self.grid = []

        for y in range(self.config['rows']):
            row = []

            for x in range(self.config['columns']):
                pos = (x, y)
                tile = _Tile(pos, is_mine=False)
                row.append(tile)
            
            self.grid.append(row)

    def onFirstClick(self, clicked_tile_coords):
        no_mine_tiles = [clicked_tile_coords]

        if self.config['first_click_is_zero'] == True:
            adj_tiles = self.get_adjacent_tile_coords(*clicked_tile_coords)
            no_mine_tiles.extend(adj_tiles)
        
        self._populateGrid(no_mine_tiles, self.seed)

    def _populateGrid(self, excluded_tiles_coords, seed):
        self._populateGridWithMines(excluded_tiles_coords, seed)
        self._markGridSquaresWithMineProximityCount()

    def _populateGridWithMines(self, excluded_tiles_coords, seed):
        ''' 
            Excludes tile that was clicked on in first click by player. First click is always safe in minesweeper.
            Game grid generated is based off of seed provided. Using the same seed will create the same grid.

            Using convention that max number of mines on a QxP grid is (Q-1)(P-1),
            in accordance to custom ranking rules on http://www.minesweeper.info/custom.php
            and Window's minesweeper implementation http://www.minesweeper.info/wiki/Windows_Minesweeper#Trivia
            
            Method assumes Game's configuration is valid.'''

        seeded_generator = Random(seed)
        for _ in range(self.config['num_mines']):
            self._placeMineInRandomEmptySquare(excluded_tiles_coords, seeded_generator)   

    def _placeMineInRandomEmptySquare(self, excluded_tiles_coords, seeded_generator):
        mine_placed = False

        # Keep trying random squares until an empty (non-mine) square is found
        while not mine_placed:
            x = seeded_generator.randrange(0, self.config['columns'])
            y = seeded_generator.randrange(0, self.config['rows'])

            if (x, y) not in excluded_tiles_coords and not self.grid[y][x].is_mine:
                self.grid[y][x].is_mine = True
                mine_placed = True

    def _markGridSquaresWithMineProximityCount(self):
        for x in range(self.config['columns']):
            for y in range(self.config['rows']):
                self.grid[y][x].num_adjacent_mines = self._countAdjacentMines(x, y)

    def _countAdjacentMines(self, x, y):
        adjacent_mines_count = 0
        adjacent_tiles_coords = self.get_adjacent_tile_coords(x, y)

        for coords in adjacent_tiles_coords:
            x, y = coords
            tile = self.grid[y][x]

            if tile.is_mine:
                adjacent_mines_count += 1

        return adjacent_mines_count

    def get_adjacent_tile_coords(self, x, y, exclude_diagonals=False):
        max_x = self.config['columns'] - 1
        max_y = self.config['rows'] - 1

        adjacent_tiles = []

        for i in [-1, 0, 1]:
            new_x = x + i

            # Out of bounds, no tile exists there.
            if new_x < 0 or new_x > max_x:
                continue

            for j in [-1, 0, 1]:
                new_y = y + j

                # Out of bounds, no tile exists there.
                if new_y < 0 or new_y > max_y:
                    continue
                
                # We want adjacent tiles, not the tile itself
                if new_x == x and new_y == y:
                    continue
                
                if exclude_diagonals and abs(i) == abs(j):
                    continue

                adjacent_tile = (new_x, new_y)
                adjacent_tiles.append(adjacent_tile)

        return adjacent_tiles


    class State(Enum):
        START = 1
        PLAY = 2
        WIN = 3
        LOSE = 4
        ILLEGAL_MOVE = 5