import os
from enum import Enum

import pygame

from .game import Game
from .renderer import Renderer

# Sprites is global as most classes in this module need access to it, and it
# only needs to be loaded into memory once. Apart from loading in the sprites,
# no class should make any changes to the sprites.
sprites = {}


class PygameRenderer(Renderer):
    def __init__(self, config, grid, agent):
        self.agent = agent
        self.game_config = config

        # First game's starting conditions
        self.grid = grid
        self.mines_left = self.game_config['num_mines']
        self.game_state = Game.State.START

        self.things_to_draw = []
        self.tile_sprites = None
        self.sprite_being_held = None
        self.mouse_being_held = False
        self.last_action_coords = None
        self.mine_counter = None
        self.timer = None
        self.emote_button = None
        self.illegal_move_tint_group = pygame.sprite.Group()
        self.is_agent = (agent != None)
        self.agent_next_move = None

        # Constants 
        self.TILE_SIZE = 16
        self.CLOCK_TICK_EVENT = pygame.USEREVENT + 1
        self.PLAY_AGENT_MOVE_EVENT = pygame.USEREVENT + 2
        self.AGENT_TIME_BETWEEN_MOVES = 5
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

        icon = sprites['icon_16']
        pygame.display.set_icon(icon)

        background_sprite = BackgroundSprite(self.game_config, self.TILE_SIZE)
        sprites['background'] = background_sprite.image
        group = pygame.sprite.Group()
        group.add(background_sprite)
        self.things_to_draw.append(group)

        screen_width = sprites['background'].get_width()
        screen_height = sprites['background'].get_height()

        # DEBUG: place screen in specific location (so it's easier to work with vs code debugger)
        position = (1920 - screen_width, (1080 - screen_height) // 2 - 50)
        os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
    
        self.screen = pygame.display.set_mode((screen_width, screen_height))
    @staticmethod
    def loadSprites():
        global sprites
        
        # Do nothing if they're already loaded.
        if sprites:
            return

        # Assumes sprites folder is in the same directory as this script.
        DIR_THIS_SCRIPT_IS_IN = os.path.dirname(os.path.realpath(__file__))
        SPRITES_FOLDER_PATH = DIR_THIS_SCRIPT_IS_IN + "\\sprites\\"

        for file_name in os.listdir(SPRITES_FOLDER_PATH):
            # Load
            path = SPRITES_FOLDER_PATH + file_name
            sprite = pygame.image.load(path)

            # Store
            name = file_name.replace('.png', '')
            sprites[name] = sprite

    def initialiseObjects(self):
        self.initialiseGrid()
        self.initialiseHUD()
        self.initialiseIllegalMoveTint()

    def initialiseGrid(self):
        grid_x = sprites['background_middle_left'].get_width()
        grid_y = sprites['background_top_left'].get_height()

        self.tile_sprites = []
        group = pygame.sprite.Group()

        for y, row in enumerate(self.grid):
            sprites_row = []

            for x, tile in enumerate(row):
                pos = (grid_x + (x * self.TILE_SIZE), grid_y + y * self.TILE_SIZE)
                tile_sprite = TileSprite(tile, self.game_state, pos, x, y, self.is_agent, self)
                sprites_row.append(tile_sprite)
            
            self.tile_sprites.append(sprites_row)
            group.add(sprites_row) 

        self.things_to_draw.append(group)

    def initialiseHUD(self):
        COUNTER_X_MARGIN = 5
        COUNTER_Y = 15  # Got by counting pixels. There's probably a better way to figure this out.

        mine_count_x = sprites['background_top_left'].get_width() + COUNTER_X_MARGIN
        mine_count_y = COUNTER_Y
        mine_count_pos = (mine_count_x, mine_count_y)
        mine_count = Counter(mine_count_pos)
        mine_count.updateValue(self.game_config['num_mines'])

        timer_x = sprites['background'].get_width() - sprites['background_top_right'].get_width() - COUNTER_X_MARGIN - sprites['counter_backdrop'].get_width()
        timer_y = COUNTER_Y
        timer_pos = (timer_x, timer_y)
        timer = Counter(timer_pos)

        emote_button_x = ((sprites['background'].get_width() // 2) - 3) - (self.TILE_SIZE // 2)
        emote_button_y = COUNTER_Y
        emote_button_pos = (emote_button_x, emote_button_y)
        emote_button = EmoteButton(emote_button_pos, self.game_state)

        self.mine_counter = mine_count
        self.timer = timer
        self.emote_button = emote_button

        counters_group = pygame.sprite.Group()
        emote_group = pygame.sprite.Group()

        counters_group.add(self.mine_counter, self.timer)
        emote_group.add(self.emote_button)

        self.things_to_draw.extend((counters_group, emote_group))

    def initialiseIllegalMoveTint(self):
        grid_pos = self.tile_sprites[0][0].rect.topleft
        grid_size = (len(self.grid[0]) * self.TILE_SIZE, len(self.grid) * self.TILE_SIZE)
        grid_rect = pygame.Rect(grid_pos, grid_size)
        self.illegal_move_tint = IllegalMoveTint(grid_rect)
        self.things_to_draw.append(self.illegal_move_tint_group)

    def initialiseAgent(self):
        if self.agent:
            self.agent.update(self.grid, self.mines_left, self.game_state)
            self.agent.onGameBegin()
            self.agent.feedRenderer(self)
            pygame.time.set_timer(self.PLAY_AGENT_MOVE_EVENT, self.AGENT_TIME_BETWEEN_MOVES)           

    def getNextMoveFromAgentAndMarkTiles(self):
        next_move = self.agent.nextMove()
        self.markAgentAction(next_move)

        tiles_to_highlight = self.agent.highlightTiles()
        
        if tiles_to_highlight:
            for (x, y, highlight_code) in tiles_to_highlight:
                self.tile_sprites[y][x].addHighlight(highlight_code)
        self.draw()
        
        return next_move

    def getNextMove(self):
        action = None

        self.draw()
        while not action:
            action = self.handleEvents()
            self.update()
            self.draw()
        
        if action == self.ACTION_RESET_GAME:
            self.onGameRestart()
        else:
            self.last_action_coords = (action[0], action[1])
            self.agent_next_move = None

        return action

    # Reset counters and remove illegal-move red tint
    def onGameRestart(self):
        self.timer.updateValue(0)
        self.illegal_move_tint.kill()

    def update(self):
        if not self.agent_next_move and self.agent and self.game_state in [Game.State.PLAY, Game.State.START]:
            self.agent_next_move = self.getNextMoveFromAgentAndMarkTiles()

    def handleEvents(self):
        action = None

        for event in pygame.event.get():
            if event.type in [pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION]:
                self.handleSpriteBeingHeldSwitch(event)
            if event.type == pygame.MOUSEBUTTONUP:
                self.handleSpriteBeingHeldSwitch(event)
                action = self.onMouseButtonUp(event)
            if event.type == self.CLOCK_TICK_EVENT:
                self.onTimerTick()
            if event.type == self.PLAY_AGENT_MOVE_EVENT:
                action = self.playNextMove()
            if event.type == pygame.QUIT:
                pygame.quit()

        return action

    def playNextMove(self):
        if self.game_state in [Game.State.WIN, Game.State.LOSE, Game.State.ILLEGAL_MOVE]:
            # Game ended so update screen and then send a game reset action.
            self.onGameRestart()
            action = self.ACTION_RESET_GAME
        elif self.agent_next_move:
            action = self.agent_next_move
        else:
            action = None
        
        return action

    def onMouseButtonUp(self, event):
        sprite_at_cursor = self.getSpriteAtCursor()
        action = None

        if isinstance(sprite_at_cursor, TileSprite):    
            action = self.onGridClick(event, sprite_at_cursor)      
        elif isinstance(sprite_at_cursor, EmoteButton) and event.button == MouseButton.LEFT.value:
            action = self.ACTION_RESET_GAME

        return action

    def onGridClick(self, event, tile_sprite):
        # Ignore click if AI is playing, or if in an end-game state 
        if self.agent or (self.game_state in [Game.State.WIN, Game.State.LOSE, Game.State.ILLEGAL_MOVE]):
            return None

        # Clicking an uncovered tile is an illegal move. Ignore it.
        if self.grid[tile_sprite.y][tile_sprite.x].uncovered:
            return None
        
        toggle_flag = (event.button == MouseButton.RIGHT.value)

        # Trying to uncover a flagged tile is an illegal move. Ignore it.
        if not toggle_flag and self.grid[tile_sprite.y][tile_sprite.x].is_flagged:
            return None

        # Start clock event on first tile uncovering
        if self.game_state == Game.State.START and not toggle_flag:
            pygame.time.set_timer(self.CLOCK_TICK_EVENT, self.ONE_SECOND) 

        return (tile_sprite.x, tile_sprite.y, toggle_flag)


    def handleSpriteBeingHeldSwitch(self, event):
        if self.sprite_being_held:
            self.sprite_being_held.updateBeingHeld(False, self.game_state)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.mouse_being_held = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.mouse_being_held = False

        # pygame provides buttons tuple for MOUSEMOTION but a button integer for clicks.
        if event.type == pygame.MOUSEMOTION:
            is_LMB = (event.buttons[0] == 1)
        else:
            is_LMB = (event.button == MouseButton.LEFT.value)
        
        sprite = self.getSpriteAtCursor()

        # Right mouse button holds are not considered as holding the sprite
        if sprite and sprite.holdable and self.mouse_being_held and is_LMB:
            self.sprite_being_held = sprite
        else:
            self.sprite_being_held = None

        if self.sprite_being_held:
            self.sprite_being_held.updateBeingHeld(True, self.game_state)

        if isinstance(self.sprite_being_held, TileSprite) and self.sprite_being_held.holdable:
            tile_being_held = self.sprite_being_held
        else:
            tile_being_held = None
        
        self.emote_button.updateTileBeingHeld(tile_being_held)

    def onTimerTick(self):
        self.timer.increment()

    def getSpriteAtCursor(self):
        cursor_pos = pygame.mouse.get_pos()
        
        for sprite_group in self.things_to_draw:
            for sprite in sprite_group:
                if not isinstance(sprite, BackgroundSprite) and sprite.rect.collidepoint(cursor_pos):
                    return sprite

        return None

    def highlightTilesAndDraw(self, tiles_to_highlight, add_highlights):
        if add_highlights == None:
            raise ValueError("Required 'add_highlights' keyword is missing")

        for ((x, y), code) in tiles_to_highlight:
            if add_highlights:
                self.tile_sprites[y][x].addHighlight(code)
            else:
                self.tile_sprites[y][x].removeHighlight(code)
        
        pygame.event.pump()  # Incase OS is not updating display immediately
        self.draw()

    def removeAllTileHighlights(self, tiles):
        for (x, y) in tiles:
            self.tile_sprites[y][x].removeAllHighlights()

        pygame.event.pump()
        self.draw()

    def draw(self):
        for group in self.things_to_draw:
            group.draw(self.screen)

        pygame.display.flip()

    def updateFromResult(self, result):
        self.grid, self.mines_left, self.game_state = result

        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                new_tile = self.grid[y][x]
                self.tile_sprites[y][x].update(new_tile, self.game_state, self.is_agent)

        self.mine_counter.updateValue(self.mines_left)
        self.emote_button.update(self.game_state)

        if self.agent:
            self.agent.update(self.grid, self.mines_left, self.game_state)

            if self.game_state == Game.State.START:
                self.agent.onGameBegin()
        
        if self.game_state in [Game.State.LOSE, Game.State.WIN, Game.State.ILLEGAL_MOVE]:
            pygame.time.set_timer(self.CLOCK_TICK_EVENT, self.NO_TIMER)    # Stop the clock by disabling timer event

            if self.game_state == Game.State.LOSE:
                self.onGameLoss()
            elif self.game_state == Game.State.ILLEGAL_MOVE:
                self.onIllegalMove()
        
        # Remove highlights from tile as turn has finished.
        for row in self.tile_sprites:
            for tile in row:
                tile.removeAllHighlights()

    def markAgentAction(self, next_move):
        (x, y, toggle_flag) = next_move
        if toggle_flag:
            sprite = sprites['tile_mark_flag_1']
        else:
            sprite = sprites['tile_mark_no_flag_1']
        
        self.tile_sprites[y][x].blitSpriteOverTile(sprite)

    # Notifies grid object which mine tile was clicked that lost the game so it can show the appropriate sprite
    def onGameLoss(self):
        (x, y) = self.last_action_coords
        self.tile_sprites[y][x].updateAsFinalClick()

 
    # Put a red tint over the grid to notify user an illegal move was made
    def onIllegalMove(self):
        self.illegal_move_tint_group.add(self.illegal_move_tint)
        
    
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
    

class Counter(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)
        self.value = 0
        self.NUM_DIGITS = 3

        self.rect = self.initialiseRect(pos)
        self.image = self.getSprite()
        self.rect.topleft = pos
        self.holdable = False


    def initialiseRect(self, pos):
        counter_size = sprites['counter_backdrop'].get_size()
        return pygame.Rect(pos, counter_size)


    def getSprite(self):
        # Bound value to be drawn to -99 or 999 if it exceeds either. 
        value_to_draw = min(self.value, 999)
        value_to_draw = max(value_to_draw, -99)

        # Pad with leading 0's, if needed to match the required length
        padded_digits_string = str(value_to_draw).zfill(self.NUM_DIGITS)

        digits = [digit for digit in padded_digits_string]
        digit_width = sprites['num_0'].get_width()
        digit_height = sprites['num_0'].get_height()

        counter_surface = pygame.Surface(self.rect.size)

        # Draw backdrop
        counter_surface.blit(sprites['counter_backdrop'], (0, 0))

        # Draw each digit
        for i in range(self.NUM_DIGITS):
            if digits[i] == '-':
                sprite_name = "num_dash"
            else:
                sprite_name = 'num_' + digits[i]
                
            digit_image = sprites[sprite_name]

            digit_pos = (1 + (i * digit_width), 1)

            counter_surface.blit(digit_image, digit_pos)
        
        return counter_surface
    
    def updateBeingHeld(self, *args):
        pass

    def updateValue(self, value):
        self.value = value
        self.image = self.getSprite()
    
    def increment(self):
        self.value += 1
        self.image = self.getSprite()
    
    def decrement(self):
        self.value -= 1
        self.image = self.getSprite()


class EmoteButton(pygame.sprite.Sprite):
    def __init__(self, pos, game_state):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.getSprite(game_state)
        self.real_image = self.image
        self.rect = self.image.get_rect()
        self.rect.topleft = pos

        self.being_held = False
        self.holdable = True

    def getSprite(self, game_state):
        if game_state in [Game.State.PLAY, Game.State.START]:
            sprite = sprites['emote_smile']
        elif game_state == Game.State.WIN:
            sprite = sprites['emote_cool']
        else:
            sprite = sprites['emote_dead']

        return sprite

    def update(self, game_state):
        self.image = self.real_image = self.getSprite(game_state)

    def updateBeingHeld(self, being_held, game_state):
        self.being_held = being_held

        if being_held and self.holdable:
            self.image = sprites['emote_smile_pressed']
        else:
            self.image = self.real_image
    
    def updateTileBeingHeld(self, tile_being_held):
        if tile_being_held:
            self.image = sprites['emote_shock']
        elif not self.being_held:
            self.image = self.real_image


class TileSprite(pygame.sprite.Sprite):
    def __init__(self, tile, game_state, pos, grid_x, grid_y, is_agent, renderer):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.getSprite(tile, game_state)
        self.real_image = self.image
        self.rect = self.image.get_rect()
        self.rect.topleft = pos

        self.x = grid_x
        self.y = grid_y
        self.TILE_NUM_TO_SPRITE_NAME = ["tile_uncovered", "tile_1", "tile_2", "tile_3", "tile_4", "tile_5", "tile_6", "tile_7", "tile_8"]
        self.holdable = self.isHoldable(tile, game_state, is_agent)

        # DEBUG
        self.highlights = set()
        self.renderer = renderer

    def getSprite(self, tile, game_state):
        if tile.uncovered:
            if tile.is_mine:
                sprite = sprites['tile_mine']
            elif tile.num_adjacent_mines == 0:
                sprite = sprites['tile_uncovered']
            else:
                sprite_name = self.TILE_NUM_TO_SPRITE_NAME[tile.num_adjacent_mines]
                sprite = sprites[sprite_name]
        else:
            if tile.is_flagged:
                sprite = sprites['tile_flag']
            else:
                sprite = sprites['tile_covered']

        # End of game. Override with appropriate sprite if necessary.
        if game_state == Game.State.WIN:
            if tile.is_mine:
                sprite = sprites['tile_flag']
        elif game_state == Game.State.LOSE:
            if tile.is_mine:
                sprite = sprites['tile_mine']
            elif tile.is_flagged:
                sprite = sprites['tile_mine_crossed']

        return sprite

    def update(self, tile, game_state, is_agent):
        self.holdable = self.isHoldable(tile, game_state, is_agent)
        self.image = self.getSprite(tile, game_state)
        self.real_image = self.image

    def isHoldable(self, tile, game_state, is_agent):
        return not is_agent and not tile.uncovered and not tile.is_flagged and game_state in [Game.State.PLAY, Game.State.START]

    def updateBeingHeld(self, being_held, game_state):
        if being_held and self.holdable:
            self.image = sprites['tile_uncovered']
        else:
            self.image = self.real_image

    def updateAsFinalClick(self):
        self.image = sprites['tile_mine_red']

    def addHighlight(self, highlight_code):
        self.highlights.add(str(highlight_code))
        self.highlightImage()

    def removeHighlight(self, highlight_code):
        self.highlights.discard(str(highlight_code))
        self.highlightImage()
    
    def removeAllHighlights(self):
        self.highlights = set()
        self.image = self.real_image

    def highlightImage(self):
        self.image = self.real_image

        for highlight_code in self.highlights:
            sprite_name = 'tile_highlight_' + highlight_code
            sprite = sprites[sprite_name]
            self.blitSpriteOverTile(sprite)

    def blitSpriteOverTile(self, sprite_image):
        # Create new surface as the regular tile sprite image is a reference of the sprite in global sprites array.
        # Blitting directly onto that sprite would change the sprite for every tile.
        marked_tile = pygame.Surface(self.image.get_size())
        marked_tile.blit(self.image, (0, 0))
        marked_tile.blit(sprite_image, (0, 0))

        self.image = marked_tile
        

class IllegalMoveTint(pygame.sprite.Sprite):
    def __init__(self, rect):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.createRedTintSurface(rect)
        self.rect = rect
        self.holdable = False

    def createRedTintSurface(self, rect):
        colour_red = pygame.Color(255, 0, 0)

        red_tint_surface = pygame.Surface(rect.size)
        red_tint_surface.fill(colour_red)
        red_tint_surface.set_alpha(150)

        return red_tint_surface


class BackgroundSprite(pygame.sprite.Sprite):
    def __init__(self, game_config, tile_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = self.generateScaledBackground(game_config, tile_size)
        self.rect = self.image.get_rect()
        self.holdable = False

    # Smallest background is that with an 8x1 grid (even if user sets it to something smaller).
    # Any extra space in the background needed for a larger grid is measured by how many 
    # extra rows/columns of tiles are needed.
    @staticmethod
    def generateScaledBackground(game_config, tile_size):
        rows = max(game_config['rows'], 1)
        columns = max(game_config['columns'], 8)

        return BackgroundSprite.makeBackground(rows, columns, tile_size)

    # Background template is split into a grid of 9 images. This function pieces them together, scaling the appropriate pieces
    # to match the space required (dependendant on grid size) to create the scaled background image.
    @staticmethod
    def makeBackground(rows, columns, tile_size):
        width = sprites['background_top_left'].get_width() + (tile_size * columns) + sprites['background_top_right'].get_width()
        height = sprites['background_top_left'].get_height() + (tile_size * rows) + sprites['background_bottom_left'].get_height()

        background_size = (width, height)
        background = pygame.Surface(background_size)

        background = BackgroundSprite.putOnCornersOfBackground(background)
        background = BackgroundSprite.putOnBordersOfBackground(background, rows, columns)
        background = BackgroundSprite.putOnPlayAreaOfBackground(background, rows, columns)
        
        return background

    # Note that each corner of the background is constant in size, i.e. regardless of grid size, they'll stay the same.
    @staticmethod
    def putOnCornersOfBackground(background):
        # Calculate coordinates that are dependant on background and image sizes
        top_right_x = background.get_width() - sprites['background_top_right'].get_width()
        bottom_left_y = background.get_height() - sprites['background_bottom_left'].get_height()
        bottom_right_x = background.get_width() - sprites['background_bottom_right'].get_width()
        bottom_right_y = background.get_height() - sprites['background_bottom_right'].get_height()

        top_left_pos = (0, 0)
        top_right_pos = (top_right_x, 0)
        bottom_left_pos = (0, bottom_left_y)
        bottom_right_pos = (bottom_right_x, bottom_right_y)

        background.blit(sprites['background_top_left'], top_left_pos)
        background.blit(sprites['background_top_right'], top_right_pos)
        background.blit(sprites['background_bottom_left'], bottom_left_pos)
        background.blit(sprites['background_bottom_right'], bottom_right_pos)

        return background
    
    @staticmethod
    def putOnBordersOfBackground(background, rows, columns):
        left_border = BackgroundSprite.scaleImageOnAxis(sprites['background_middle_left'], rows, scale_x_axis=False)
        right_border = BackgroundSprite.scaleImageOnAxis(sprites['background_middle_right'], rows, scale_x_axis=False)
        top_border = BackgroundSprite.scaleImageOnAxis(sprites['background_top_middle'], columns, scale_x_axis=True)
        bottom_border = BackgroundSprite.scaleImageOnAxis(sprites['background_bottom_middle'], columns, scale_x_axis=True)
        
        # Calcualte coordinates that depend on size of other images
        left_border_y = sprites['background_top_left'].get_height()
        right_border_x = background.get_width() - sprites['background_top_right'].get_width()
        right_border_y = left_border_y
        top_border_x = sprites['background_top_left'].get_width()
        bottom_border_x = top_border_x
        bottom_border_y = background.get_height() - sprites['background_bottom_left'].get_height()

        left_border_pos = (0, left_border_y)
        right_border_pos = (right_border_x, right_border_y)
        top_border_pos = (top_border_x, 0)
        bottom_border_pos = (bottom_border_x, bottom_border_y)

        background.blit(left_border, left_border_pos)
        background.blit(right_border, right_border_pos)
        background.blit(top_border, top_border_pos)
        background.blit(bottom_border, bottom_border_pos)

        return background

    @staticmethod
    def putOnPlayAreaOfBackground(background, rows, columns):
        play_area_scaled_y = BackgroundSprite.scaleImageOnAxis(sprites['background_middle'], rows, scale_x_axis=False)
        play_area = BackgroundSprite.scaleImageOnAxis(play_area_scaled_y, columns, scale_x_axis=True)

        play_area_x = sprites['background_top_left'].get_width()
        play_area_y = sprites['background_top_left'].get_height()

        play_area_pos = (play_area_x, play_area_y)

        background.blit(play_area, play_area_pos)

        return background

    @staticmethod
    def scaleImageOnAxis(image, scale_factor, scale_x_axis):
        new_x = image.get_width()
        new_y = image.get_height()

        if scale_x_axis:
            new_x *= scale_factor
        else:
            new_y *= scale_factor

        scaled_image_dimensions = (new_x, new_y)

        return pygame.transform.scale(image, scaled_image_dimensions)


class MouseButton(Enum):
    LEFT = 1
    MIDDLE = 2
    RIGHT = 3