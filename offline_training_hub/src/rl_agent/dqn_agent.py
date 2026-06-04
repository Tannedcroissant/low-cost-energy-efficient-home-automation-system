import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from .networks import QNetwork


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ReplayBuffer:
    
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(DEVICE),
            torch.tensor(np.array(next_states), dtype=torch.float32).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        )
        
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        
        
        self.q_network = QNetwork(state_size, action_size).to(DEVICE)
        self.target_network = QNetwork(state_size, action_size).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        
    def act(self, state):
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().numpy())
        
    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        current_q = self.q_network(states).gather(1, actions)
     
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))
            
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay