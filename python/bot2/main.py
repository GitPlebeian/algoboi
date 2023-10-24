import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import os

# Hyperparameters
GAMMA = 0.99
EPS_START = 1
EPS_MIN = 0.01
EPS_DECAY = 0.97
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TARGET_UPDATE = 10

current_eps = EPS_START
total_explorations = 0

# Q-network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 actions: Buy, Sell, Do Nothing
        )

    def forward(self, x):
        return self.fc(x)

# Select action
def select_action(state, policy_net):
    sample = random.random()
    # print("Sample", sample)

    if sample > current_eps:
        with torch.no_grad():
            value = policy_net(state).max(1)[1].view(1, 1)
            # print("Action value: ", value)
            return value
    else:
        global total_explorations
        random_value = random.randrange(3)
        total_explorations = total_explorations + 1
        # print("Random Value: ", random_value)
        return torch.tensor([[random_value]], dtype=torch.long)

# Training loop
def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    transitions = random.sample(memory, BATCH_SIZE)
    batch = list(zip(*transitions))
    
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = target_net(state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main loop
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = []

current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test.json file
json_path = os.path.join(current_dir, '..', '..', 'shared', 'test.json')

# Read and load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

candles = list(zip(data["slopeOf9DayEMA"], data["slopeOf25DayEMA"], data["closes"]))

steps_done = 0
holding_price = None
for episode in range(120):  # Number of episodes
    score = 0
    total_explorations = 0
    # print("Current EPS: ", current_eps)
    for candle in candles:
        ema1, ema2, price = candle
        state = torch.tensor([ema1, ema2], dtype=torch.float32).unsqueeze(0)

        # Decide action
        action = select_action(state, policy_net)
        steps_done += 1

        # Get reward
        if action == 0 and holding_price is None:  # Buy
            holding_price = price
            reward = 0
        elif action == 1 and holding_price is not None:  # Sell
            reward = (price - holding_price) / holding_price
            holding_price = None
            # print("Appending reward", reward)
        else:
            reward = 0

        score += reward

        reward = torch.tensor([reward], dtype=torch.float32)
        memory.append((state, action, reward))
        
        # Optimize model
        optimize_model(policy_net, target_net, optimizer, memory)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode: {episode} Score: {score} Total Explorations: {total_explorations} Current EPS: {current_eps}")

    if current_eps * EPS_DECAY >= EPS_MIN:
        current_eps = current_eps * EPS_DECAY
    else:
        current_eps = EPS_MIN
