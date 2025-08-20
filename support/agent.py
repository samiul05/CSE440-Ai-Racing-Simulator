import numpy as np
from dqn import DQNAgent
from config import STATE_SIZE, ACTION_SIZE
import os

class RLAgent:
    def __init__(self):
        self.dqn_agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
        self.scores = []
        self.episode = 0
        
    def get_action(self, state):
        state = np.reshape(state, [1, STATE_SIZE])
        return self.dqn_agent.act(state)
    
    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, STATE_SIZE])
        next_state = np.reshape(next_state, [1, STATE_SIZE])
        self.dqn_agent.remember(state, action, reward, next_state, done)
        
        if len(self.dqn_agent.memory) > 32:
            self.dqn_agent.replay(32)
    
    def update_target_model(self):
        self.dqn_agent.update_target_model()
    
    def save_model(self, filename):
        self.dqn_agent.model.save(filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename):
        try:
            if os.path.exists(filename):
                self.dqn_agent.model.load_weights(filename)
                print(f"Model loaded from {filename}")
                return True
            else:
                print(f"No existing model found at {filename}")
                return False
        except Exception as e:
            print(f"Could not load model: {e}")
            return False
    
    def get_epsilon(self):
        return self.dqn_agent.epsilon
    
    def set_epsilon(self, epsilon):
        self.dqn_agent.epsilon = epsilon

