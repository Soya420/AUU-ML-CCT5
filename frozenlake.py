import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialize the environment
env = gym.make('FrozenLake-v1', desc=[
        "SFFFFFFFFF",
        "FFFFFFFFFF",
        "FFHFFFFFFF",
        "FFFFHFFFFF",
        "FFFHFFFFFF",
        "FHHFFFHFFF",
        "FHFFHFHFFF",
        "FFFHFFFFHF",
        "FFFFFFFHFF",
        "FFFFHFFFFG"
    ], is_slippery=False)

# Manually specify state and action space sizes
state_space_size = 100  # 10x10 grid has 100 states
action_space_size = 4  # Typically 4 actions: left, down, right, up

# Initialize Q-table with specified sizes
q_table = np.zeros((state_space_size, action_space_size))

# Hyperparameters
num_episodes = 30000
max_steps_per_episode = 400
learning_rate = 0.1  # 𝛼 (alpha)
discount_rate = 0.95  # 𝛾 (gamma)
exploration_rate = 1  # ε (epsilon)
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0005

# List to store rewards for each episode
rewards_all_episodes = []

# Define a function to calculate the Manhattan distance to the goal
def get_manhattan_distance(state, goal_state, grid_size=10):
    state_row, state_col = divmod(state, grid_size)
    goal_row, goal_col = divmod(goal_state, grid_size)
    return abs(goal_row - state_row) + abs(goal_col - state_col)

# Get goal state from the map
goal_state = 99  # For the given grid, the goal 'G' is at position 99 (last position)

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

        # Calculate the new Manhattan distance to the goal
        old_distance = get_manhattan_distance(state, goal_state)
        new_distance = get_manhattan_distance(new_state, goal_state)

        # Update rewards manually
        if new_state == goal_state:
            reward = 10  # Large reward for reaching the goal
        elif done:  # Falling into a hole
            reward = -5  # High penalty for falling into a hole
        elif new_distance < old_distance:
            reward = 1  # Small positive reward for getting closer to the goal
        elif new_distance > old_distance:
            reward = -1  # Small penalty for getting further away from the goal
        else:
            reward = -0.1  # Small penalty for neutral moves to encourage exploration

        # Update Q-table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        total_rewards += reward

        if done or truncated:
            break

    # Apply exploration rate decay only after 1000 episodes
    if episode < 6000:
        exploration_rate = max_exploration_rate
    else:
        exploration_rate = min_exploration_rate + (
            max_exploration_rate - min_exploration_rate
        ) * np.exp(-exploration_decay_rate * (episode - 6000))

    # Store total rewards for this episode
    rewards_all_episodes.append(total_rewards)

    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards_all_episodes[-1000:])  # Average of last 1000 episodes
        print(f"Episode: {episode + 1}, Average Reward: {avg_reward:.3f}, Exploration Rate: {exploration_rate:.2f}")

# Close the environment
env.close()

# Plotting the rewards
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
average_rewards = [np.mean(rewards) for rewards in rewards_per_thousand_episodes]

plt.plot(range(1, len(average_rewards) + 1), average_rewards)
plt.xlabel('Episodes (in thousands)')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.grid(True)
plt.show()
