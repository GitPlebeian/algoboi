import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the DNN
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        state, action, reward = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward

    def __len__(self):
        return len(self.buffer)

# Parameters
INPUT_DIM = 2  # Excluding the closing price
OUTPUT_DIM = 3  # buy, sell, hold
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CAPACITY = 10000  # Experience replay buffer capacity

# Initialize network, optimizer, and buffer
network = DNN(INPUT_DIM, OUTPUT_DIM)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(CAPACITY)
loss_fn = nn.MSELoss()

# Assuming you have a method to select action based on epsilon-greedy or other policy
def select_action(state):
    with torch.no_grad():
        return network(state).argmax().item()

# Assuming you have a method to compute reward at the end of the episode
def compute_episode_reward(closing_prices):
    return (closing_prices[-1] - closing_prices[0]) / closing_prices[0]

# Training Loop (for one episode)
for state in states:
    # Extracting only the first two values for decision making
    action_input = torch.tensor(state[:2], dtype=torch.float32)
    action = select_action(action_input)

    # You can then use 'action' to interact with your trading environment, get new state, etc.
    # ...

    # After completing the episode, compute reward
    closing_prices = [x[2] for x in states]
    episode_reward = compute_episode_reward(closing_prices)
    
    # Add all state-action pairs with the reward to the replay buffer
    for s in states:
        buffer.push(s[:2], action, episode_reward)

    # Sample from buffer and update network
    if len(buffer) >= BATCH_SIZE:
        sampled_states, sampled_actions, sampled_rewards = buffer.sample(BATCH_SIZE)
        sampled_states = torch.tensor(sampled_states, dtype=torch.float32)
        sampled_actions = torch.tensor(sampled_actions, dtype=torch.int64)
        sampled_rewards = torch.tensor(sampled_rewards, dtype=torch.float32)

        # Forward pass
        predicted_rewards = network(sampled_states).gather(1, sampled_actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute loss and backpropagate
        loss = loss_fn(predicted_rewards, sampled_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
