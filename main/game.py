import pygame
import numpy as np
import matplotlib.pyplot as plt
from config import *
from track import Track
from car import Car
from agent import RLAgent
import time
import math

class Game:
    def __init__(self, fast_training=True):
        self.fast_training = fast_training
        self.screen = None
        self.clock = None
        self.pygame_initialized = False
        
        if not self.fast_training:
            self.init_pygame()
        
        self.track = Track()
        self.car = None
        self.agent = RLAgent()
        self.running = True
        self.training = True
        self.episode_scores = []
        self.episode_steps = []
        self.episode_distances = []
        
    def init_pygame(self):
        """Initialize pygame only when needed"""
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Self-Driving Car Racing")
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True
    
    def reset(self):
        self.car = Car(self.track.start_position[0], self.track.start_position[1], 
                      self.track.start_angle)
        self.car.alive = True
        self.car.distance_traveled = 0
        self.car.time_alive = 0
        return self.car.get_state(self.track.get_walls())
    
    def calculate_reward(self):
        if not self.car or not self.car.alive:
            return -50  # Penalty for crashing
        
        reward = 0
        
        # Reward for moving forward
        reward += self.car.speed * 0.3
        
        # Reward for staying alive
        reward += 0.1
        
        # Reward for distance traveled
        reward += self.car.distance_traveled * 0.005
        
        # Bonus for staying on track
        if self.track.is_on_track(self.car.x, self.car.y):
            reward += 0.2
            
        return reward
    
    def step(self, action):
        if not self.car:
            return None, -50, True
            
        walls = self.track.get_walls()
        old_state = self.car.get_state(walls)
        self.car.update(action, self.track)
        new_state = self.car.get_state(walls)
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Episode termination conditions
        done = (not self.car.alive or 
                self.car.time_alive > MAX_STEPS_PER_EPISODE or
                self.car.distance_traveled > 2000)  # Complete a good distance
        
        return new_state, reward, done
    
    def handle_events(self):
        """Handle pygame events to keep window responsive"""
        if not self.pygame_initialized or not self.screen:
            return True
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def run_episode(self, episode_num=0):
        state = self.reset()
        total_reward = 0
        steps = 0
        
        # For fast training, don't render
        render = not self.fast_training and (self.training or not self.training)
        
        # Initialize pygame for testing if needed
        if render and not self.pygame_initialized:
            self.init_pygame()
        
        while self.car and self.car.alive and steps < MAX_STEPS_PER_EPISODE:
            # Handle events for window responsiveness
            if render:
                if not self.handle_events():
                    return False
            
            if self.training:
                action = self.agent.get_action(state)
            else:
                # For testing, use trained model
                action = self.agent.get_action(state)
            
            next_state, reward, done = self.step(action)
            
            if next_state is None:  # Error occurred
                break
                
            if self.training:
                self.agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render game if needed
            if render:
                self.render()
                if FPS > 0 and self.clock:
                    self.clock.tick(FPS)
            
            # Break if done
            if done:
                break
        
        self.episode_scores.append(total_reward)
        self.episode_steps.append(steps)
        distance = self.car.distance_traveled if self.car else 0
        self.episode_distances.append(distance)
        
        if self.training and episode_num % TARGET_UPDATE_FREQUENCY == 0:
            self.agent.update_target_model()
        
        print(f"Episode finished: Steps={steps}, Distance={distance:.1f}")
        return True
    
    def render(self):
        if not self.pygame_initialized or not self.screen or not self.car:
            return
            
        self.screen.fill((0, 0, 0))
        self.track.draw(self.screen)
        if self.car:
            self.car.draw(self.screen)
        
        # Add some text information
        try:
            font = pygame.font.SysFont('Arial', 20)
            score_text = font.render(f"Score: {self.episode_scores[-1] if self.episode_scores else 0:.1f}", True, (255, 255, 255))
            episode_text = font.render(f"Episode: {len(self.episode_scores)}", True, (255, 255, 255))
            distance_text = font.render(f"Distance: {self.car.distance_traveled if self.car else 0:.0f}", True, (255, 255, 255))
            mode_text = font.render(f"Mode: {'Training' if self.training else 'Testing'}", True, (255, 255, 255))
            
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(episode_text, (10, 35))
            self.screen.blit(distance_text, (10, 60))
            self.screen.blit(mode_text, (10, 85))
        except:
            pass  # Font might not be available
        
        pygame.display.flip()
    
    def cleanup(self):
        """Clean up pygame resources"""
        if self.pygame_initialized:
            try:
                pygame.quit()
            except:
                pass
            self.pygame_initialized = False
            self.screen = None
            self.clock = None
    
    def plot_training_progress(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.episode_scores)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 3, 2)
        plt.plot(self.episode_steps)
        plt.title('Episode Steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.subplot(1, 3, 3)
        plt.plot(self.episode_distances)
        plt.title('Distance Traveled')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        
        plt.tight_layout()
        plt.show()
    
    def train(self, episodes=TRAINING_EPISODES):
        print(f"🏁 Starting training for {episodes} episodes...")
        if GPU_AVAILABLE:
            print("Using NVIDIA GPU for accelerated training!")
        else:
            print("Using CPU for training.")
            
        start_time = time.time()
        
        for episode in range(episodes):
            episode_start = time.time()
            
            if not self.run_episode(episode):
                print("Training stopped by user.")
                break
                
            episode_time = time.time() - episode_start
            
            # Print progress every EPISODES_PER_PRINT episodes
            if (episode + 1) % EPISODES_PER_PRINT == 0:
                avg_score = np.mean(self.episode_scores[-EPISODES_PER_PRINT:])
                avg_steps = np.mean(self.episode_steps[-EPISODES_PER_PRINT:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Avg Steps: {avg_steps:.1f} | "
                      f"Time: {episode_time:.2f}s | "
                      f"Epsilon: {self.agent.dqn_agent.epsilon:.3f}")
        
        total_time = time.time() - start_time
        print(f"\n🏁 Training completed in {total_time:.2f} seconds")
        print(f"Average time per episode: {total_time/episodes:.2f} seconds")
        
        self.plot_training_progress()
    
    def test(self, episodes=10):
        self.training = False
        self.agent.dqn_agent.epsilon = 0  # No exploration during testing
        
        print("🏁 Testing trained agent!")
        print("Press ESC or close window to stop testing early")
        test_scores = []
        test_distances = []
        
        for episode in range(episodes):
            if not self.run_episode():
                print("Testing stopped by user.")
                break
            test_scores.append(self.episode_scores[-1])
            test_distances.append(self.episode_distances[-1])
            print(f"Test Episode {episode + 1}: Score = {test_scores[-1]:.2f}, "
                  f"Distance = {test_distances[-1]:.2f}")
        
        if test_scores:
            avg_score = np.mean(test_scores)
            avg_distance = np.mean(test_distances)
            print(f"\n📊 Testing Results:")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Average Distance: {avg_distance:.2f}")
            print(f"Best Score: {max(test_scores):.2f}")
            print(f"Best Distance: {max(test_distances):.2f}")
        
        return test_scores
