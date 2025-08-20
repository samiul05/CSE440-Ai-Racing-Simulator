import pygame
import math
from config import SCREEN_WIDTH, SCREEN_HEIGHT

class Track:
    def __init__(self):
        # Simple rectangular track with clear boundaries
        self.outer_rect = pygame.Rect(100, 100, 600, 400)
        self.inner_rect = pygame.Rect(200, 200, 400, 200)
        
        # Start position
        self.start_position = (400, 150)
        self.start_angle = -90
        
    def draw(self, screen):
        # Draw outer track
        pygame.draw.rect(screen, (255, 255, 255), self.outer_rect, 2)
        # Draw inner track
        pygame.draw.rect(screen, (255, 255, 255), self.inner_rect, 2)
        # Draw start line
        pygame.draw.line(screen, (0, 255, 0), (350, 100), (450, 100), 3)
    
    def get_walls(self):
        # Outer walls
        outer_walls = [
            ((100, 100), (700, 100)),  # Top
            ((700, 100), (700, 500)),  # Right
            ((700, 500), (100, 500)),  # Bottom
            ((100, 500), (100, 100))   # Left
        ]
        
        # Inner walls
        inner_walls = [
            ((200, 200), (600, 200)),  # Top
            ((600, 200), (600, 400)),  # Right
            ((600, 400), (200, 400)),  # Bottom
            ((200, 400), (200, 200))   # Left
        ]
        
        return outer_walls + inner_walls
    
    def check_collision(self, car_x, car_y):
        """Check if car position is out of track bounds"""
        # Check if car is outside outer boundary or inside inner boundary
        if (car_x < 100 or car_x > 700 or car_y < 100 or car_y > 500):
            return True
        if (car_x > 200 and car_x < 600 and car_y > 200 and car_y < 400):
            return True
        return False
    
    def is_on_track(self, car_x, car_y):
        """Check if car is on the track (between inner and outer boundaries)"""
        # Car is on track if it's within outer bounds but outside inner bounds
        in_outer = (car_x >= 100 and car_x <= 700 and car_y >= 100 and car_y <= 500)
        in_inner = (car_x > 200 and car_x < 600 and car_y > 200 and car_y < 400)
        return in_outer and not in_inner
