import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random

# Initialize the environment
env = gym.make('FrozenLake-v1', desc=[
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
], is_slippery=False)

# Manually specify state and action space sizes
state_space_size = 64  # Example value for an 8*8 grid
action_space_size = 4  # Typically 4 actions: left, down, right, up

# Initialize Q-table with specified sizes
q_table = np.zeros((state_space_size, action_space_size))

# Hyperparameters
num_episodes = 15000
max_steps_per_episode = 1000
learning_rate = 0.08  # 𝛼 (alpha)
discount_rate = 0.9  # 𝛾 (gamma)
exploration_rate = 1  # ε (epsilon)
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0005

# List to store rewards for each episode
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()[0]
    done = False
    total_rewards = 0
    for step in range(max_steps_per_episode):
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        # Take action and observe the outcome
        new_state, reward, done, truncated, _ = env.step(action)

        # Update Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        total_rewards += reward

        if done or truncated:
            break

    # Apply exploration rate decay only after 1000 episodes
    if episode < 1000:
        exploration_rate = max_exploration_rate
    else:
        exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * (episode - 1000))

    # Store total rewards for this episode
    rewards_all_episodes.append(total_rewards)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_all_episodes[-100:])  # Average of last 100 episodes
        print(f"Episode: {episode + 1}, Average Reward: {avg_reward:.2f}, Exploration Rate: {exploration_rate:.2f}")

# Close the environment
env.close()
