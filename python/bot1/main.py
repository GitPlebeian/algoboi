
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers.legacy import Adam
import json
import os
import pandas as pd

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # if np.random.rand() <= self.epsilon:
        #     random_value = random.randrange(self.action_size)
        #     print("RANDOM", random_value)
        #     return random_value
        act_values = self.model.predict(state)
        print("ACT VALUES", act_values)
        return np.argmax(act_values[0])

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.train_on_batch(state, target_f)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay


class TradingEnvironment:
    def __init__(self, data):
        self.data = pd.DataFrame({
            'closes': data['closes'],
            'slopeOf9DayEMA': data['slopeOf9DayEMA'],
            'slopeOf25DayEMA': data['slopeOf25DayEMA']
        })
        self.data = self.data.iloc[data['minimumStartIndex']:]
        self.data = self.data.reset_index(drop=True)
        self.current_step = 0
        self.current_average_price = None  # None means not holding, otherwise it's the purchase price
        self.purchased_state = None

    def reset(self):
        self.current_step = 0
        self.current_average_price = None
        self.purchased_state = None
        return self.data.iloc[self.current_step].values[1:]  # return the initial state

    def step(self, action):
        # action = 0 (Hold), 1 (Buy), 2 (Sell)
        reward = None
        if action == 1 and self.current_average_price == None: # Buy the stock:
            current_average_price = self.data.iloc[self.current_step].values[0]
            purchased_state = self.data.iloc[self.current_step].values[1:]
            print("Purchasing Stock At Price: ", current_average_price)
        elif action == 2 and self.current_average_price != None: # Sell the stock and assign reward
            percentage_change = (self.data.iloc[self.current_step].values[0] - self.current_average_price) / self.current_average_price * 100
            reward = percentage_change
            self.current_average_price = None
            print("Selling Stock For Change Of ", percentage_change)



        if action == 1 and self.current_position is None:  # Buy
            self.current_position = self.data.iloc[self.current_step]['price']
        elif action == 2 and self.current_position is not None:  # Sell
            reward = self.data.iloc[self.current_step]['price'] - self.current_position
            self.current_position = None
        
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.data.iloc[self.current_step].values[:3]
        return next_state, reward, done

    def action_space(self):
        return 3

    def state_space(self):
        return len(self.data.iloc[0].values[1:])


# Get the directory of the current python script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the test.json file
json_path = os.path.join(current_dir, '..', '..', 'shared', 'test.json')

# Read and load the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

env = TradingEnvironment(data)
state_size = env.state_space()
print("State Size: ", state_size)
action_size = env.action_space()
agent = DQNAgent(state_size, action_size)
batch_size = 32
episodes = 1

for e in range(episodes):
    state = env.reset();
    state = np.reshape(state, [1, state_size])
    for candle in range(len(env.data)):
        action = agent.act(state)
        # next_state, reward

# if __name__ == "__main__":



#     # Load your data into 'data'
#     env = TradingEnvironment(data)
#     state_size = env.state_space()
#     action_size = env.action_space()
#     agent = DQNAgent(state_size, action_size)
#     batch_size = 32
#     episodes = 100

#     for e in range(episodes):
#         state = env.reset()
#         state = np.reshape(state, [1, state_size])
#         for time in range(len(env.data)):
#             action = agent.act(state)
#             next_state, reward, done = env.step(action)
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)

