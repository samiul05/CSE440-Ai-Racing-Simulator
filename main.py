import pygame
import random
import numpy as np

# Window setup
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Q-learning setup
ACTIONS = ['left', 'right', 'forward']
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1.0  # Exploration rate
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01

# Pygame init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Car Racing")
clock = pygame.time.Clock()

# Car class
class Car:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - 100
        self.angle = 0
        self.speed = 5
        self.size = 20

    def draw(self):
        pygame.draw.rect(screen, (0, 200, 255), (self.x, self.y, self.size, self.size))

    def move(self, action):
        if action == 'left':
            self.x -= self.speed
        elif action == 'right':
            self.x += self.speed
        elif action == 'forward':
            self.y -= self.speed

    def get_state(self):
        return (self.x // 50, self.y // 50)

    def is_off_track(self):
        return self.x < 0 or self.x > WIDTH or self.y < 0

# Initialize Q-table
q_table = {}

# Training loop
def train_ai_car():
    global EPSILON

    car = Car()
    total_reward = 0

    for episode in range(1, 301):  # Train for 300 episodes
        car = Car()
        done = False
        reward = 0
        total_reward = 0

        for step in range(200):
            state = car.get_state()

            # Initialize Q-table entry
            if state not in q_table:
                q_table[state] = np.zeros(len(ACTIONS))

            # Choose action
            if random.uniform(0, 1) < EPSILON:
                action_index = random.randint(0, len(ACTIONS) - 1)
            else:
                action_index = np.argmax(q_table[state])

            action = ACTIONS[action_index]
            car.move(action)

            # Check for crash
            if car.is_off_track():
                reward = -100
                done = True
            else:
                reward = 1  # Reward for surviving

            new_state = car.get_state()

            if new_state not in q_table:
                q_table[new_state] = np.zeros(len(ACTIONS))

            # Q-Learning update
            max_future_q = np.max(q_table[new_state])
            current_q = q_table[state][action_index]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[state][action_index] = new_q

            total_reward += reward
            if done:
                break

        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
        print(f"Episode {episode} - Reward: {total_reward} - Epsilon: {EPSILON:.4f}")

# Game loop to show trained agent (optional)
def run_game():
    car = Car()
    run = True
    while run:
        screen.fill(WHITE)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        state = car.get_state()
        if state in q_table:
            action_index = np.argmax(q_table[state])
            action = ACTIONS[action_index]
        else:
            action = random.choice(ACTIONS)

        car.move(action)
        car.draw()

        if car.is_off_track():
            print("Crashed!")
            car = Car()

        pygame.display.update()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    train_ai_car()
    run_game()
