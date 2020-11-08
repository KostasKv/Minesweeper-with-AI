import pygame
import os
from spritesheet import Spritesheet
from enum import Enum
from Game import Game
from Renderer import Renderer


class PygameRenderer(Renderer):
    def __init__(self, config, grid, agent):
        self.agent = agent
        self.game_config = config

        # First game's starting conditions
        self.grid = grid
        self.mines_left = self.game_config['num_mines']
        self.game_state = Game.State.START

        self.rendered_objects = []
        self.sprites = {}
        self.object_being_held_info = None
        self.mouse_being_held = False
        self.last_tile_action_coords = None

        # Constants
        self.SPRITES_FOLDER_PATH = "sprites/"
        self.TILE_SIZE = 16
        self.CLOCK_TICK_EVENT = pygame.USEREVENT + 1
        self.AGENT_TRIGGER_EVENT = pygame.USEREVENT + 2
        self.AGENT_TIME_BETWEEN_MOVES = 1000
        self.ONE_SECOND = 1000
        self.NO_TIMER = 0
        self.ACTION_RESET_GAME = -1

        self.initialise()


    def initialise(self):
        self.loadSprites()
        self.initialiseScreenAndBackground()
        self.initialiseObjects()
        self.initialiseAgent()


    def initialiseScreenAndBackground(self):
        pygame.init()
        pygame.display.set_caption("Minesweeper")

        icon = self.sprites['icon_16']
        pygame.display.set_icon(icon)

        self.sprites['background'] = self.generateScaledBackground()

        screen_width = self.sprites['background'].get_width()
        screen_height = self.sprites['background'].get_height()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        

    def loadSprites(self):
        for file_name in os.listdir(self.SPRITES_FOLDER_PATH):
            # Load
            path = self.SPRITES_FOLDER_PATH + file_name
            sprite = pygame.image.load(path)

            # Store
            name = file_name.replace('.png', '')
            self.sprites[name] = sprite
            

    # Smallest background is that with an 8x1 grid (even if user sets it to something smaller).
    # Any extra space in the background needed for a larger grid is measured by how many 
    # extra rows/columns of tiles are needed.
    def generateScaledBackground(self):
        rows = max(self.game_config['rows'], 1)
        columns = max(self.game_config['columns'], 8)

        return self.makeBackground(rows, columns)


    # Background template is split into a grid of 9 images. This function pieces them together, scaling the appropriate pieces
    # to match the space required (dependendant on grid size) to create the scaled background image.
    def makeBackground(self, rows, columns):
        width = self.sprites['background_top_left'].get_width() + (self.TILE_SIZE * columns) + self.sprites['background_top_right'].get_width()
        height = self.sprites['background_top_left'].get_height() + (self.TILE_SIZE * rows) + self.sprites['background_bottom_left'].get_height()

        background_size = (width, height)
        background = pygame.Surface(background_size)

        background = self.putOnCornersOfBackground(background)
        background = self.putOnBordersOfBackground(background, rows, columns)
        background = self.putOnPlayAreaOfBackground(background, rows, columns)
        
        return background


    # Note that each corner of the background is constant in size, i.e. regardless of grid size, they'll stay the same.
    def putOnCornersOfBackground(self, background):
        # Calculate coordinates that are dependant on background and image sizes
        top_right_x = background.get_width() - self.sprites['background_top_right'].get_width()
        bottom_left_y = background.get_height() - self.sprites['background_bottom_left'].get_height()
        bottom_right_x = background.get_width() - self.sprites['background_bottom_right'].get_width()
        bottom_right_y = background.get_height() - self.sprites['background_bottom_right'].get_height()

        top_left_pos = (0, 0)
        top_right_pos = (top_right_x, 0)
        bottom_left_pos = (0, bottom_left_y)
        bottom_right_pos = (bottom_right_x, bottom_right_y)

        background.blit(self.sprites['background_top_left'], top_left_pos)
        background.blit(self.sprites['background_top_right'], top_right_pos)
        background.blit(self.sprites['background_bottom_left'], bottom_left_pos)
        background.blit(self.sprites['background_bottom_right'], bottom_right_pos)

        return background


    def putOnBordersOfBackground(self, background, rows, columns):
        left_border = self.scaleImageOnAxis(self.sprites['background_middle_left'], rows, scale_x_axis=False)
        right_border = self.scaleImageOnAxis(self.sprites['background_middle_right'], rows, scale_x_axis=False)
        top_border = self.scaleImageOnAxis(self.sprites['background_top_middle'], columns, scale_x_axis=True)
        bottom_border = self.scaleImageOnAxis(self.sprites['background_bottom_middle'], columns, scale_x_axis=True)
        
        # Calcualte coordinates that depend on size of other images
        left_border_y = self.sprites['background_top_left'].get_height()
        right_border_x = background.get_width() - self.sprites['background_top_right'].get_width()
        right_border_y = left_border_y
        top_border_x = self.sprites['background_top_left'].get_width()
        bottom_border_x = top_border_x
        bottom_border_y = background.get_height() - self.sprites['background_bottom_left'].get_height()

        left_border_pos = (0, left_border_y)
        right_border_pos = (right_border_x, right_border_y)
        top_border_pos = (top_border_x, 0)
        bottom_border_pos = (bottom_border_x, bottom_border_y)

        background.blit(left_border, left_border_pos)
        background.blit(right_border, right_border_pos)
        background.blit(top_border, top_border_pos)
        background.blit(bottom_border, bottom_border_pos)

        return background


    def putOnPlayAreaOfBackground(self, background, rows, columns):
        play_area_scaled_y = self.scaleImageOnAxis(self.sprites['background_middle'], rows, scale_x_axis=False)
        play_area = self.scaleImageOnAxis(play_area_scaled_y, columns, scale_x_axis=True)

        play_area_x = self.sprites['background_top_left'].get_width()
        play_area_y = self.sprites['background_top_left'].get_height()

        play_area_pos = (play_area_x, play_area_y)

        background.blit(play_area, play_area_pos)

        return background


    def scaleImageOnAxis(self, image, scale_factor, scale_x_axis):
        new_x = image.get_width()
        new_y = image.get_height()

        if scale_x_axis:
            new_x *= scale_factor
        else:
            new_y *= scale_factor

        scaled_image_dimensions = (new_x, new_y)

        return pygame.transform.scale(image, scaled_image_dimensions)


    def initialiseObjects(self):
        self.initialiseGrid()
        self.initialiseHUD()


    def initialiseGrid(self):
        grid_x = self.sprites['background_middle_left'].get_width()
        grid_y = self.sprites['background_top_left'].get_height()

        pos = (grid_x, grid_y)

        grid_obj = Grid(self.screen, self.sprites, pos, self.TILE_SIZE, self.grid, self.game_state, self.agent != None)
        self.rendered_objects.append(grid_obj)


    def initialiseHUD(self):
        COUNTER_X_MARGIN = 5
        COUNTER_Y = 15  # Got by counting pixels. There's probably a better way to figure this out.

        mine_count_x = self.sprites['background_top_left'].get_width() + COUNTER_X_MARGIN
        mine_count_y = COUNTER_Y
        mine_count_pos = (mine_count_x, mine_count_y)
        mine_count = Counter(self.screen, self.sprites, mine_count_pos, "mine_count")
        mine_count.value = self.game_config['num_mines']

        clock_x = self.sprites['background'].get_width() - self.sprites['background_top_right'].get_width() - COUNTER_X_MARGIN - self.sprites['counter_backdrop'].get_width()
        clock_y = COUNTER_Y
        clock_pos = (clock_x, clock_y)
        clock = Counter(self.screen, self.sprites, clock_pos, "clock")

        emote_button_x = ((self.sprites['background'].get_width() // 2) - 3) - (self.TILE_SIZE // 2)
        emote_button_y = COUNTER_Y
        emote_button_pos = (emote_button_x, emote_button_y)
        emote_button = EmoteButton(self.screen, self.sprites, emote_button_pos, self.game_state, self.agent != None)

        self.rendered_objects.append(mine_count)
        self.rendered_objects.append(clock)
        self.rendered_objects.append(emote_button)


    def initialiseAgent(self):
        if self.agent:
            self.agent.update(self.grid, self.mines_left, self.game_state)

            pygame.time.set_timer(self.AGENT_TRIGGER_EVENT, self.AGENT_TIME_BETWEEN_MOVES)


    def getNextMoveFromAgent(self):
        agentAction = None

        while not agentAction:
            agentAction = self.handleEvents()
            self.update()
            self.draw()
        
        if agentAction == self.ACTION_RESET_GAME:
            self.onGameRestart()
        else:
            x, y, _ = agentAction
            self.last_tile_action_coords = (x, y)

        return agentAction


    # Reset counters and remove illegal-move red tint
    def onGameRestart(self):
        for obj in self.rendered_objects:
            if isinstance(obj, Counter):
                obj.value = 0
            if isinstance(obj, IllegalMoveTint):
                self.rendered_objects.remove(obj)


    def update(self):
        for obj in self.rendered_objects:
            if isinstance(obj, Grid):
                obj.updateGrid(self.grid, self.object_being_held_info, self.game_state)
            elif isinstance(obj, Counter):
                if obj.name == 'mine_count':
                    obj.value = self.mines_left
            elif isinstance(obj, EmoteButton):
                if self.object_being_held_info:
                    object_being_held = self.object_being_held_info['object']
                else:
                    object_being_held = None

                obj.updateState(object_being_held, self.game_state)


    def handleEvents(self):
        agent_action = None

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.onMouseButtonDown()
            if event.type == pygame.MOUSEBUTTONUP:
                agent_action = self.onMouseButtonUp(event)
            if event.type == pygame.MOUSEMOTION:
                self.onMouseMotion()
            if event.type == self.CLOCK_TICK_EVENT:
                self.onTimerTick()
            if event.type == self.AGENT_TRIGGER_EVENT:
                agent_action = self.getMoveFromAgent()
            if event.type == pygame.QUIT:
                pygame.quit()

        return agent_action


    def onMouseButtonDown(self):
        self.mouse_being_held = True

        pos = pygame.mouse.get_pos()
        object_at_cursor = self.getObjectAtPosition(pos)

        if object_at_cursor:
            self.object_being_held_info = {'object': object_at_cursor, 'mouse_pos': pos}


    def onMouseButtonUp(self, event):
        self.mouse_being_held = False
        self.object_being_held_info = None

        pos = pygame.mouse.get_pos()
        object_at_cursor = self.getObjectAtPosition(pos)

        action = None

        if isinstance(object_at_cursor, Grid):    
            action = self.onGridClick(event, object_at_cursor, pos)      
        elif isinstance(object_at_cursor, EmoteButton):
            action = self.ACTION_RESET_GAME

        return action


    def onGridClick(self, event, grid_obj, pos):
        # Ignore click if AI is playing, or if in an end-game state 
        if self.agent or (self.game_state in [Game.State.WIN, Game.State.LOSE, Game.State.ILLEGAL_MOVE]):
            return None

        (x, y) = grid_obj.getCellCoordinatesAtPositionOnScreen(pos)

        # Clicking an uncovered cell is an illegal move. Ignore it.
        if self.grid[x][y].uncovered:
            return None
        
        toggle_flag = (event.button == MouseButton.RIGHT.value)
        
        # Trying to uncover a flagged cell is an illegal move. Ignore it.
        if not toggle_flag and self.grid[x][y].is_flagged:
            return None

        # Start clock event on first cell uncovering
        if self.game_state == Game.State.START and not toggle_flag:
            pygame.time.set_timer(self.CLOCK_TICK_EVENT, self.ONE_SECOND) 

        self.last_tile_action_coords = (x, y)

        return (x, y, toggle_flag)


    def onMouseMotion(self):
        if self.mouse_being_held:
            pos = pygame.mouse.get_pos()
            object_at_cursor = self.getObjectAtPosition(pos)

            if object_at_cursor:
                self.object_being_held_info = {'object': object_at_cursor, 'mouse_pos': pos}
            else:
                self.object_being_held_info = None


    def onTimerTick(self):
         for obj in self.rendered_objects:
            if isinstance(obj, Counter) and obj.name == 'clock':
                obj.value += 1


    def getObjectAtPosition(self, pos):
        for obj in self.rendered_objects:
            if obj.rect.collidepoint(pos):
                return obj

        return None


    def getMoveFromAgent(self):
        if self.game_state in [Game.State.WIN, Game.State.LOSE, Game.State.ILLEGAL_MOVE]:
            # Game ended so update screen and then send a game reset action.
            self.onGameRestart()
            action = self.ACTION_RESET_GAME
        else:
            action = self.agent.nextMove()
        
        return action


    def draw(self):
        self.screen.blit(self.sprites['background'], (0, 0))

        for obj in self.rendered_objects:
            obj.draw()
        
        pygame.display.flip()


    def updateFromResult(self, result):
        self.grid, self.mines_left, self.game_state = result

        if self.agent:
            self.agent.update(self.grid, self.mines_left, self.game_state)
        
        if self.game_state in [Game.State.LOSE, Game.State.WIN, Game.State.ILLEGAL_MOVE]:
            pygame.time.set_timer(self.CLOCK_TICK_EVENT, self.NO_TIMER)    # Stop the clock by disabling timer event

            if self.game_state == Game.State.LOSE:
                self.onGameLoss()
            elif self.game_state == Game.State.ILLEGAL_MOVE:
                self.onIllegalMove()


    # Notifies grid object which mine tile was clicked that lost the game so it can show the appropriate sprite
    def onGameLoss(self):
        for obj in self.rendered_objects:
                    if isinstance(obj, Grid):
                        obj.mine_clicked_coords = self.last_tile_action_coords

 
    # Put a red tint over the grid to notify user an illegal move was made
    def onIllegalMove(self):
        grid_rect = None

        # Get grid's area rect
        for obj in self.rendered_objects:
            if isinstance(obj, Grid):
                grid_rect = obj.rect

        tint = IllegalMoveTint(self.screen, grid_rect)
        self.rendered_objects.append(tint)
        
    
    def resetGame(self):
        if self.game_state in [Game.State.START, Game.State.PLAY]:
            result = self.executor.forceResetGame()
        else:
            # Trigger start of new game
            result = self.executor.makeMove(None)
            
        # Out of games.
        if not result:
            return True

        self.updateFromResult(result, 0, 0)

        
        
        return False

    
    def onEndOfGames(self):
        print("onEndOfGames NOT YET IMPLEMENTED")
        pygame.quit()
    

class Counter():
    def __init__(self, screen, sprites, pos, name):
        self.screen = screen
        self.sprites = sprites
        self.value = 0
        self.NUM_DIGITS = 3
        self.rect = self.initialiseRect(pos)
        self.name = name


    def initialiseRect(self, pos):
        counter_size = self.sprites['counter_backdrop'].get_size()
        return pygame.Rect(pos, counter_size)


    def draw(self):
        # Bound value to be drawn to -99 or 999 if it exceeds either. 
        value_to_draw = min(self.value, 999)
        value_to_draw = max(value_to_draw, -99)

        # Pad with leading 0's, if needed to match the required length
        padded_digits_string = str(value_to_draw).zfill(self.NUM_DIGITS)

        digits = [digit for digit in padded_digits_string]
        digit_width = self.sprites['num_0'].get_width()
        digit_height = self.sprites['num_0'].get_height()

        counter_surface = pygame.Surface(self.rect.size)

        # Draw backdrop
        counter_surface.blit(self.sprites['counter_backdrop'], (0, 0))

        # Draw each digit
        for i in range(self.NUM_DIGITS):
            if digits[i] == '-':
                sprite_name = "num_dash"
            else:
                sprite_name = 'num_' + digits[i]
                
            digit_image = self.sprites[sprite_name]

            digit_pos = (1 + (i * digit_width), 1)

            counter_surface.blit(digit_image, digit_pos)
        
        self.screen.blit(counter_surface, self.rect)


class EmoteButton():
    def __init__(self, screen, sprites, pos, game_state, is_agent):
        self.screen = screen
        self.sprites = sprites
        self.rect = self.initialiseRect(pos)
        self.object_being_held = None
        self.game_state = game_state
        self.is_agent = is_agent

    
    def initialiseRect(self, pos):
        size = self.sprites['emote_backdrop'].get_size()
        return pygame.Rect(pos, size)


    def updateState(self, object_being_held, game_state):
        self.object_being_held = object_being_held
        self.game_state = game_state
    

    def draw(self):
        surface = pygame.Surface(self.rect.size)
        surface.blit(self.sprites['emote_backdrop'], (0, 0))

        emote_sprite = self.getEmote()
        surface.blit(emote_sprite, (1, 1))
        
        self.screen.blit(surface, self.rect)

    def getEmote(self):
        if isinstance(self.object_being_held, EmoteButton):
            sprite = self.sprites['emote_smile_pressed']
        elif self.game_state == Game.State.WIN:
            sprite = self.sprites['emote_cool']
        elif self.game_state in [Game.State.LOSE, Game.State.ILLEGAL_MOVE]:
            sprite = self.sprites['emote_dead']
        elif self.object_being_held == None: 
            sprite = self.sprites['emote_smile']
        elif isinstance(self.object_being_held, Grid) and not self.is_agent:
            sprite = self.sprites['emote_shock']
        else:
            sprite = self.sprites['emote_smile']

        return sprite


class Grid():
    def __init__(self, screen, sprites, pos, tile_size, grid, game_state, is_agent):
        self.screen = screen
        self.sprites = sprites
        self.tile_size = tile_size
        self.grid = grid
        self.rows = len(grid[0])
        self.columns = len(grid)
        self.rect = self.initialiseRect(pos)
        self.TILE_NUM_TO_SPRITE_NAME = ["tile_uncovered", "tile_1", "tile_2", "tile_3", "tile_4", "tile_5", "tile_6", "tile_7", "tile_8"]
        self.cell_being_held_coordinates = None
        self.game_state = game_state
        self.mine_clicked_coords = None
        self.is_agent = is_agent


    def initialiseRect(self, pos):
        width = self.columns * self.tile_size
        height = self.rows * self.tile_size
        size = (width, height)
        return pygame.Rect(pos, size)


    def updateGrid(self, grid, object_being_held_info, game_state):
        self.grid = grid
        self.game_state = game_state

        if object_being_held_info and isinstance(object_being_held_info['object'], Grid):
            pos = object_being_held_info['mouse_pos']
            self.cell_being_held_coordinates = self.getCellCoordinatesAtPositionOnScreen(pos)
        else:
            self.cell_being_held_coordinates = None


    def getCellCoordinatesAtPositionOnScreen(self, pos):
        grid_x = pos[0] - self.rect.x
        grid_y = pos[1] - self.rect.y

        x = grid_x // self.tile_size
        y = grid_y // self.tile_size

        return (x, y)


    def draw(self):
        grid_surface = pygame.Surface(self.rect.size)

        for x in range(self.columns):
            for y in range(self.rows):
                cell = self.grid[x][y]
                sprite = self.getCellSprite(cell, x, y)
                
                # Cell position is relative to the grid surface, not overall screen
                cell_x = self.tile_size * x
                cell_y = self.tile_size * y
                cell_top_left_pos = (cell_x, cell_y)

                grid_surface.blit(sprite, cell_top_left_pos)
        
        self.screen.blit(grid_surface, self.rect)


    def getCellSprite(self, cell, x, y):
        if cell.uncovered:
            if cell.is_mine:
                sprite = self.sprites['tile_mine']
            else:
                sprite_name = self.TILE_NUM_TO_SPRITE_NAME[cell.num_adjacent_mines]
                sprite = self.sprites[sprite_name]
        else:
            if cell.is_flagged:
                sprite = self.sprites['tile_flag']
            elif self.cell_being_held_coordinates == (x, y) and (self.game_state in [Game.State.START, Game.State.PLAY]) and not self.is_agent:
                sprite = self.sprites['tile_uncovered'] 
            else:
                sprite = self.sprites['tile_covered']
        
        # End of game
        if self.game_state == Game.State.WIN:
            if cell.is_mine:
                sprite = self.sprites['tile_flag']
        elif self.game_state == Game.State.LOSE:
            if cell.is_mine:
                if self.mine_clicked_coords == (x, y):
                    sprite = self.sprites['tile_mine_red']
                else:
                    sprite = self.sprites['tile_mine']
            elif cell.is_flagged:
                sprite = self.sprites['tile_mine_crossed']


        return sprite


class IllegalMoveTint():
    def __init__(self, screen, rect):
        colour_red = pygame.Color(255, 0, 0)

        red_tint_surface = pygame.Surface(rect.size)
        red_tint_surface.fill(colour_red)
        red_tint_surface.set_alpha(150)

        self.screen = screen
        self.rect = rect
        self.surface = red_tint_surface

    def draw(self):
        self.screen.blit(self.surface, self.rect)


class MouseButton(Enum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3