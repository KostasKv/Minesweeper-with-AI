# Class for handling spritesheets
# This was taken and adapted from https://ehmatthes.github.io/pcc_2e/beyond_pcc/pygame_sprite_sheets
import pygame

class Spritesheet:

    def __init__(self, filename):
        """Load the spritesheet."""
        try:
            self.spritesheet = pygame.image.load(filename).convert()
        except pygame.error as e:
            print(f"Unable to load spritesheet image: {filename}")
            raise SystemExit(e)


    def getImageAt(self, rectangle):
        """Load a specific image from a specific rectangle."""
        # Loads image from x, y, x+offset, y+offset.
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size).convert()
        image.blit(self.spritesheet, (0, 0), rect)

        return image

    def getImagesAt(self, rects):
        """Load a whole bunch of images and return them as a list."""
        return [self.getImageAt(rect) for rect in rects]

    def getAndSplitStripAt(self, rect, image_count, padding=0, left_margin=0, right_margin=0):
        strip = self.getImageAt(rect)
        return self.splitStrip(strip, image_count, padding, left_margin, right_margin)

    def splitStrip(self, strip_img, image_count, padding=0, left_margin=0, right_margin=0):
        width = self.getWidthOfImagesInStrip(strip_img, image_count, padding, left_margin, right_margin)
        height = strip_img.get_height()
        sprite_size = (width, height)

        images = []
        for i in range(image_count):
            top_left_x = (width + padding) * i
            top_left_y = 0
            area_rect = pygame.Rect((top_left_x, top_left_y, width, height))

            # Create a new surface and draw the sprite from the spritesheet onto this surface.
            image = pygame.Surface(area_rect.size).convert()
            image.blit(strip_img, (0, 0), area_rect)

            images.append(image)

        return images

    def getWidthOfImagesInStrip(self, strip_img, image_count, padding=0, left_margin=0, right_margin=0):
        width_with_trimmed_outsides = strip_img.get_width() - left_margin - right_margin

        num_of_paddings = image_count - 1
        width_without_paddings = width_with_trimmed_outsides - (padding * num_of_paddings)

        return width_without_paddings // image_count