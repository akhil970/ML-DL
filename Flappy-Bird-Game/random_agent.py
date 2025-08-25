from flappy_bird_env import FlappyBirdEnv
import numpy as np

if __name__ == "__main__":
    env = FlappyBirdEnv(render_mode=True)
    num_episodes = 5
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # Random agent: choose action randomly (0 = do nothing, 1 = flap)
            action = np.random.choice([0, 1])
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        print(f"Episode {episode+1}: Score={info['score']} Total Reward={total_reward} Steps={steps}")
