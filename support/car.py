import pygame
import math
import numpy as np
from config import *

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0
        self.max_speed = CAR_MAX_SPEED
        self.acceleration = CAR_ACCELERATION
        self.rotation_speed = CAR_ROTATION_SPEED
        self.alive = True
        self.distance_traveled = 0
        self.time_alive = 0
        
        # Keep car within screen bounds
        self.x = max(110, min(690, self.x))
        self.y = max(110, min(490, self.y))
        
    def update(self, action, track):
        if not self.alive:
            return
            
        # Action: 0 = left, 1 = right, 2 = forward
        if action == 0:  # Turn left
            self.angle -= self.rotation_speed
        elif action == 1:  # Turn right
            self.angle += self.rotation_speed
        elif action == 2:  # Accelerate
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        
        # Apply friction
        self.speed *= 0.95
        
        # Update position
        old_x, old_y = self.x, self.y
        self.x += math.cos(math.radians(self.angle)) * self.speed
        self.y += math.sin(math.radians(self.angle)) * self.speed
        
        # Keep car within reasonable bounds
        self.x = max(110, min(690, self.x))
        self.y = max(110, min(490, self.y))
        
        # Update distance traveled
        self.distance_traveled += math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
        self.time_alive += 1
        
        # Check if car is on track
        if not track.is_on_track(self.x, self.y):
            self.alive = False
            self.speed = 0
    
    def get_sensor_data(self, track_walls):
        sensor_angles = [-60, -30, 0, 30, 60]  # 5 sensors
        sensor_data = []
        
        for sensor_angle in sensor_angles:
            total_angle = self.angle + sensor_angle
            end_x = self.x + math.cos(math.radians(total_angle)) * SENSOR_LENGTH
            end_y = self.y + math.sin(math.radians(total_angle)) * SENSOR_LENGTH
            
            sensor_line = ((self.x, self.y), (end_x, end_y))
            min_distance = SENSOR_LENGTH
            
            for wall in track_walls:
                intersection = self._line_intersection(sensor_line, wall)
                if intersection:
                    distance = math.sqrt((self.x - intersection[0])**2 + 
                                       (self.y - intersection[1])**2)
                    min_distance = min(min_distance, distance)
            
            sensor_data.append(min_distance / SENSOR_LENGTH)  # Normalize
        
        return sensor_data
    
    def _line_intersection(self, line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        
        return None
    
    def get_state(self, track_walls):
        sensor_data = self.get_sensor_data(track_walls)
        normalized_speed = self.speed / self.max_speed
        normalized_angle = (self.angle % 360) / 360
        
        return sensor_data + [normalized_speed, normalized_angle]
    
    def draw(self, screen):
        if not self.alive:
            return
            
        # Draw car as a simple rectangle
        half_width = CAR_WIDTH // 2
        half_height = CAR_HEIGHT // 2
        
        # Calculate rotated corners
        corners = [
            (self.x - half_width, self.y - half_height),
            (self.x + half_width, self.y - half_height),
            (self.x + half_width, self.y + half_height),
            (self.x - half_width, self.y + half_height)
        ]
        
        # Rotate corners
        center = (self.x, self.y)
        rotated_corners = []
        for corner in corners:
            angle_rad = math.radians(self.angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            x = corner[0] - center[0]
            y = corner[1] - center[1]
            
            new_x = x * cos_angle - y * sin_angle
            new_y = x * sin_angle + y * cos_angle
            
            rotated_corners.append((new_x + center[0], new_y + center[1]))
        
        points = [(int(x), int(y)) for x, y in rotated_corners]
        pygame.draw.polygon(screen, (255, 0, 0), points)
        
        # Draw direction indicator
        front_x = self.x + math.cos(math.radians(self.angle)) * 15
        front_y = self.y + math.sin(math.radians(self.angle)) * 15
        pygame.draw.line(screen, (0, 255, 0), (int(self.x), int(self.y)), 
                       (int(front_x), int(front_y)), 2)
        
        # Draw sensors for visualization
        sensor_angles = [-60, -30, 0, 30, 60]
        for sensor_angle in sensor_angles:
            total_angle = self.angle + sensor_angle
            end_x = self.x + math.cos(math.radians(total_angle)) * 30
            end_y = self.y + math.sin(math.radians(total_angle)) * 30
            pygame.draw.line(screen, (255, 255, 0), (int(self.x), int(self.y)), 
                           (int(end_x), int(end_y)), 1)
