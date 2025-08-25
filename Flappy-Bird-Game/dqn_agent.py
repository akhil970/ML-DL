import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
from collections import deque
from flappy_bird_env import FlappyBirdEnv

# Hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 10000
SAVED_SCORE = 0
TRACK = {}

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, steps_done, action_dim):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return random.randrange(action_dim)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()

def evaluate(num_episodes=5):
    env = FlappyBirdEnv(render_mode=True)
    state_dim = 5
    action_dim = 2
    policy_net = DQN(state_dim, action_dim)
    if os.path.exists("dqn_flappy_bird.pth"):
        policy_net.load_state_dict(torch.load("dqn_flappy_bird.pth"))
        print("Loaded trained weights for evaluation.")
    else:
        print("No trained model found. Exiting evaluation.")
        return
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        print(f"[EVAL] Episode {episode+1}: Score={info['score']} Total Reward={total_reward} Steps={steps}")

def train():
    env = FlappyBirdEnv(render_mode=False)
    state_dim = 5
    action_dim = 2
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    epsilon = EPS_START
    steps_done = 0
    # Load previous weights and epsilon if available
    if os.path.exists("dqn_flappy_bird.pth"):
        policy_net.load_state_dict(torch.load("dqn_flappy_bird.pth"))
        print("Loaded previous weights from dqn_flappy_bird.pth")
        if os.path.exists("dqn_flappy_bird_meta.json"):
            with open("dqn_flappy_bird_meta.json", "r") as f:
                meta = json.load(f)
                epsilon = meta.get("epsilon", EPS_END)
                steps_done = meta.get("steps_done", 0)
            print(f"Loaded previous epsilon: {epsilon:.4f}, steps_done: {steps_done}")
        else:
            epsilon = EPS_END
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    num_episodes = 2000
    global SAVED_SCORE
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            # Use current epsilon for exploration
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            epsilon = eps_threshold
            if random.random() < epsilon:
                action = random.randrange(action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps_done += 1
            # Learn
            if len(memory) >= BATCH_SIZE:
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(BATCH_SIZE)
                batch_state = torch.FloatTensor(batch_state)
                batch_action = torch.LongTensor(batch_action).unsqueeze(1)
                batch_reward = torch.FloatTensor(batch_reward)
                batch_next_state = torch.FloatTensor(batch_next_state)
                batch_done = torch.FloatTensor(batch_done)
                q_values = policy_net(batch_state).gather(1, batch_action).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(batch_next_state).max(1)[0]
                expected_q = batch_reward + GAMMA * next_q_values * (1 - batch_done)
                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update target network
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode+1}: Total Reward={total_reward} Score={info['score']}")
        # Save epsilon and steps_done after each episode
        with open("dqn_flappy_bird_meta.json", "w") as f:
            json.dump({"epsilon": float(epsilon), "steps_done": int(steps_done)}, f)
        if SAVED_SCORE < info['score']:
            SAVED_SCORE = info['score']
            TRACK["Episode"] = episode+1
            TRACK["Reward"] = total_reward
            TRACK["Score"] = info['score']
            torch.save(policy_net.state_dict(), "dqn_flappy_bird.pth")
            print(f"New high score! Model saved at score {SAVED_SCORE}.")
    print("Training complete. Highest Achievement: Episode - {} Reward - {} Score - {}".format(
        TRACK.get("Episode", "-"), TRACK.get("Reward", "-"), TRACK.get("Score", "-")))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate(num_episodes=5)
    else:
        train()
