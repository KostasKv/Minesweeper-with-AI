from random import Random
from .game import _Game

class _Executor():
    def __init__(self, game, num_games=None, game_seeds=None, seed=None):
        if not num_games and not game_seeds:
            raise ValueError("No game number or game seeds provided")
        if num_games and game_seeds:
            raise ValueError("Both num games and game seeds provided. Only one expected.")

        if num_games:
            self.game_seeds = self._get_game_seeds(num_games, seed)
        else:
            self.game_seeds = iter(game_seeds)

        self.num_games = num_games
        self._game = game
        self._new_games_left = num_games
        self._start_new_game()
        self._all_games_finished = False

    def _get_game_seeds(self, num_games, seed):
        GAME_SEED_RANGE_INCLUSIVE = (-(2**63), 2**63 - 1)   # 64-bit signed integer range
        random_generator = Random(seed)

        for _ in range(num_games):
            yield random_generator.randint(*GAME_SEED_RANGE_INCLUSIVE)

    def _start_new_game(self):
        next_seed = next(self.game_seeds)
        self._game.newGame(next_seed)
        self._new_games_left -= 1

    def get_game_config(self):
        return self._game.config

    def get_grid_and_mines_left_and_state(self):
        return (self._game.grid, self._game.mines_left, self._game.state)

    def make_move(self, action):
        ''' Input: action as a tuple (x, y, toggle_flag), where x, y is the 0-based coordinates
        of the tile to interact with, and toggle_flag is a boolean indiciating whether to toggle flag
        status of the tile or to click the tile.
        An action that is an integer -1 indicates the user chose to force reset the game.

        If game is in a finished state, then a call to this method will trigger a reset of the game, and
        return the appropriate grid and state. In this case the action is ignored.
        
        Returns: (grid, mines_left, game_state) as they are after the move has been made,
                 or returns None if all games have been finished, regardless of input.'''
                
        if self._all_games_finished:
            return None
        
        is_in_end_game_state = self._game.state in [_Game.State.WIN, _Game.State.LOSE, _Game.State.ILLEGAL_MOVE]
        is_force_reset = (action == -1)

        if is_in_end_game_state or is_force_reset:
            if self._new_games_left <= 0:
                self._all_games_finished = True
                return None
            else:
                self._start_new_game()
        elif self._is_legal_move(action):
            if self._game.state == _Game.State.START and not action[2]:
                safe_tile = (action[0], action[1])
                self._game.onFirstClick(safe_tile)
            
            self._process_move(action)
        else:
            self._game.state = _Game.State.ILLEGAL_MOVE

        return (self._game.grid, self._game.mines_left, self._game.state)

    def _is_legal_move(self, action):
        (x, y, toggle_flag) = action

        out_of_bounds = (x < 0) or (y < 0) or (x >= self._game.config['columns']) or (y >= self._game.config['rows'])
        
        if out_of_bounds or self._game.grid[y][x].uncovered:
            return False
        
        if not toggle_flag and self._game.grid[y][x].is_flagged:
            return False
        
        return True
    
    # Assumes input is a legal move. Changes game grid and game state based on action.
    def _process_move(self, action):
        (x, y, toggle_flag) = action

        if toggle_flag:
            self._toggle_flag(x, y)
            is_end_of_game = False  
        else:
            is_end_of_game = self._click_mine(x, y)
        
        # Toggle flag should not change game state from START to PLAY.
        # If it's end of game, state should not be overwritten back to PLAY.
        if not is_end_of_game and not toggle_flag:
            self._game.state = _Game.State.PLAY 

    # Changes game grid (one tile gets flagged/unflagged).
    def _toggle_flag(self, x, y):
        if self._game.grid[y][x].is_flagged:
            self._game.grid[y][x].is_flagged = False
            self._game.mines_left += 1
        else:
            self._game.grid[y][x].is_flagged = True
            self._game.mines_left -= 1

    # Returns boolean indicating whether game has been finished or not.
    # Method changes game's grid and state.
    def _click_mine(self, x, y):
        end_of_game = False

        if self._game.grid[y][x].is_mine:
            self._game.grid[y][x].uncovered = True
            self._game.state = _Game.State.LOSE
            end_of_game = True
        else:
            self._uncover(x, y)
            
            if self._is_game_won():
                self._game.state = _Game.State.WIN
                end_of_game = True
    
        return end_of_game

    def _uncover(self, initial_x, initial_y):
        tiles_to_uncover = [(initial_x, initial_y)]
        
        while tiles_to_uncover:
            x, y = tiles_to_uncover.pop(0)
            tile = self._game.grid[y][x]

            tile.uncovered = True

            if tile.num_adjacent_mines == 0:
                adjacent_tiles_coords = self._game.get_adjacent_tile_coords(x, y)

                for x, y in adjacent_tiles_coords:
                    if self._game.grid[y][x].uncovered == False and not self._game.grid[y][x].is_flagged:
                        if not (x, y) in tiles_to_uncover:
                            tiles_to_uncover.append((x, y))
    
    def _is_game_won(self):
        for column in self._game.grid:
            for tile in column:
                # If any non-mine tile is still covered, then game has not yet been won
                if not tile.is_mine and not tile.uncovered:
                    return False

        return True

    # def _force_reset_game(self):
    #     self._start_new_game()
    #     self._game.reset()
    #     self.games_left -= 1

    #     if self.games_left <= 0:
    #         return None
    #     else:
    #         return (self._game.grid, self._game.mines_left, self._game.state)