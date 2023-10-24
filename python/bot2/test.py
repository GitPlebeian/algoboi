import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import json
import os

# Define the DNN
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
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

    def reset(self, capacity):
        self.buffer = deque(maxlen = capacity)

    def __len__(self):
        return len(self.buffer)

# Parameters
STATE_SIZE = 2  # Excluding the closing price
ACTION_SIZE = 3  # buy, sell, hold
BATCH_SIZE = 2000
LEARNING_RATE = 0.0001
CAPACITY = 10000  # Experience replay buffer capacity

# Initialize network, optimizer, and buffer
network = DNN(STATE_SIZE, ACTION_SIZE)
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(CAPACITY)
loss_fn = nn.MSELoss()

# Environment Parameters
epsilon = 1.0  # exploration rate
epsilon_min = 0.005
epsilon_decay = 0.995
starting_portfolio = 100

random_action_count = 0

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test.json file
json_path = os.path.join(current_dir, '..', '..', 'shared', 'test.json')

# Read and load the JSON data
with open(json_path, 'r') as f:
    states = json.load(f)

# Assuming you have a method to select action based on epsilon-greedy or other policy
def select_action(state):
    action_value = 0
    if random.random() <= epsilon:
        global random_action_count
        action_value = random.randint(0, ACTION_SIZE - 1)
        random_action_count = random_action_count + 1
    else:
        with torch.no_grad():
            action_value = network(torch.tensor(state)).argmax().item()
    return action_value

num_episodes = 100

playback_buys = []
playback_sells = []


for episode in range(num_episodes):
    # print(f"Epsilon {epsilon}")
    current_buy_price = None
    portfolio_value = starting_portfolio
    total_reward = None
    actions = []
    random_action_count = 0
    # buffer.reset(CAPACITY)
    playback_buys.append([])
    playback_sells.append([])
    for index, state in enumerate(states):
        action = select_action(state[1:])
        actions.append(action)
        if action == 0 and current_buy_price == None: # Buy Stock
            current_buy_price = state[0]
            playback_buys[episode].append(index)
            # print(f"{index} Buying at price: {current_buy_price}")
        elif action == 1 and current_buy_price != None: # Sell Stock
            closing_price = state[0]
            percentage_change = (closing_price - current_buy_price) / current_buy_price
            portfolio_value = portfolio_value * (percentage_change + 1)
            current_buy_price = None
            playback_sells[episode].append(index)
            # print(f"{index} Selling {percentage_change * 100}% | ${portfolio_value}")

    total_reward = (portfolio_value - starting_portfolio) / starting_portfolio
    print(f"Ending Value {total_reward} Random Count {random_action_count}")

    if epsilon * epsilon_decay >= epsilon_min:
        epsilon = epsilon * epsilon_decay
    else:
        epsilon = 0

    for index, state in enumerate(states):
        buffer.push(state[1:], actions[index], total_reward)

    # print(buffer.__len__())

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

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test.json file
json_path = os.path.join(current_dir, '..', '..', 'shared', 'playback', 'playback.json')

playback_data = {
    "purchaseIndexs" : playback_buys,
    "sellIndexs" : playback_sells,
    "length" : len(states)
}
# print(playback_data)

with open(json_path, 'w') as json_file:
    json.dump(playback_data, json_file)
