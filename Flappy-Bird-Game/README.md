# Flappy Bird RL Agent

This project implements a Flappy Bird game in Python and trains a reinforcement learning (RL) agent using Deep Q-Networks (DQN) to play the game. The project is modular, allowing you to play manually, run a random agent, or train and evaluate a DQN agent.

## Project Structure

- `flappy_bird.py` — Play Flappy Bird manually using Pygame.
- `flappy_bird_env.py` — Gym-like environment for RL agents.
- `random_agent.py` — Runs a random agent for baseline comparison.
- `dqn_agent.py` — Trains and evaluates a DQN agent using PyTorch.
- `dqn_flappy_bird.pth` — Saved model weights (created after training).
- `dqn_flappy_bird_meta.json` — Stores epsilon and steps_done for resuming training.

## How to Use

### 1. Install Requirements
```sh
pip install -r requirements.txt
```

### 2. Play the Game Manually
```sh
python flappy_bird.py
```
Press SPACE to flap. Avoid pipes and the ground!

### 3. Run a Random Agent
```sh
python random_agent.py
```

### 4. Train the DQN Agent
```sh
python dqn_agent.py
```
The agent will train and save the best model automatically.

### 5. Evaluate the Trained Agent
```sh
python dqn_agent.py eval
```
This will run the agent in greedy mode and render the game.

## RL Details
- **State**: Bird position, velocity, and next pipe info (5 features)
- **Actions**: 0 = do nothing, 1 = flap
- **Reward**: +1 for staying alive, +10 for passing a pipe, -100 for crashing
- **Algorithm**: DQN (Deep Q-Network) with experience replay and target network

## Requirements
See `requirements.txt` for all dependencies.

## Credits
- Pygame for game rendering
- PyTorch for deep learning
- OpenAI Gym for environment inspiration
