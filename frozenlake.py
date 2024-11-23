import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
#Import the environment configurations from the env_config.py file, you can also add more environments to the file
from env_config import env_12x12, env_20x20

class CustomFrozenLakeEnv(gym.Env):
    #A custom environment for the Frozen Lake game that modifies rewards based on agent's progress.

    def __init__(self, env):
        """
        Initialize the custom environment with the given Gym environment.

        Args:
            env (gym.Env): The original Gym environment to wrap.
        """
        # Initialize the base Gym environment
        super(CustomFrozenLakeEnv, self).__init__()
        self.env = env

        # Set the action and observation spaces to match the wrapped environment
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        # Extract the environment's map description
        desc = env.unwrapped.desc.tolist()

        # Find all positions of holes ('H') in the map
        self.hole_positions = self._get_positions(desc, b'H')

        # Find the position of the goal ('G'); assuming there's only one goal
        self.goal_position = self._get_positions(desc, b'G')[0]

        # Initialize the last distance tracker
        self.last_distance = None

    def _get_positions(self, desc, item):
        """
        Get the positions of a specific item in the map.

        Args:
            desc (list): The map description as a list of lists.
            item (bytes): The item to search for (e.g., b'H' for holes).

        Returns:
            list: A list of (row, col) tuples where the item is located.
        """
        positions = []
        # Loop over each row and column in the map
        for row_idx, row in enumerate(desc):
            for col_idx, val in enumerate(row):
                if val == item:
                    # If the item matches, add its position to the list
                    positions.append((row_idx, col_idx))
        return positions

    def _get_coordinates(self, state):
        """
        Convert the linear state index to 2D coordinates.

        Args:
            state (int): The linear state index from the environment.

        Returns:
            tuple: A (row, col) tuple representing the agent's position.
        """
        # Calculate the number of columns in the map
        ncol = self.env.unwrapped.ncol
        # Compute the row and column based on the state index
        row = state // ncol
        col = state % ncol
        return (row, col)

    def _manhattan_distance(self, pos1, pos2):
        """
        Calculate the Manhattan distance between two positions.

        Args:
            pos1 (tuple): The first position (row, col).
            pos2 (tuple): The second position (row, col).

        Returns:
            int: The Manhattan distance between pos1 and pos2.
        """
        # Compute the sum of absolute differences of the coordinates
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def step(self, action):
        """
        Take an action in the environment and receive modified rewards.

        Args:
            action (int): The action to take.

        Returns:
            tuple: Contains the new state, modified reward, done flag, truncated flag, and info dict.
        """
        # Take a step using the original environment
        state, reward, done, truncated, info = self.env.step(action)
        # Increment the step count
        self.step_count += 1
        # Get the current position of the agent
        position = self._get_coordinates(state)

        # Penalize each step to encourage shorter paths
        if self.step_count > 25:
            # Increase the penalty a bit after 25 steps
            step_penalty = 0.02  # Increased penalty
        elif self.step_count > 50:
             # Increase the penalty a bit after 50 steps
            step_penalty = 0.05  # Increased penalty
        else:
            step_penalty = 0.01  # Regular penalty
        reward -= step_penalty

        # Check if the agent has fallen into a hole
        if position in self.hole_positions:
            # Apply a significant penalty for falling into a hole
            reward -= 1

        # Calculate the current distance to the goal
        distance = self._manhattan_distance(position, self.goal_position)

        # Initialize last_distance on the first step
        if self.last_distance is None:
            self.last_distance = distance

        # Reward the agent for moving closer to the goal
        if distance < self.last_distance:
            reward += 0.1  # Small reward for getting closer
        else:
            reward -= 0.07  # Small penalty for moving away or not getting closer

        # Update the last_distance for the next step
        self.last_distance = distance

        return state, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment to start a new episode.

        Args:
            **kwargs: Additional arguments for the reset method.

        Returns:
            tuple: The initial state and an info dictionary.
        """
        # Reset the original environment and get the initial state
        state, info = self.env.reset(**kwargs)
        # Get the starting position of the agent
        position = self._get_coordinates(state)
        # Initialize the last_distance to the distance from the start to the goal
        self.last_distance = self._manhattan_distance(position, self.goal_position)
        # Initialize the step count
        self.step_count = 0
        return state, info


env_use = env_12x12  # Change this to change what environment to use
env = gym.make('FrozenLake-v1', desc=env_use, is_slippery=False)
env = CustomFrozenLakeEnv(env)

# Specify state and action space sizes
state_space_size = env.observation_space.n  # Example value for an 8x8 grid = 64
action_space_size = 4  #How many actions/choices at each step? = 4 actions: left, down, right, up

# Initialize Q-table with specified sizes
q_table = np.zeros((state_space_size, action_space_size))

# Hyperparameters
num_episodes = 35000
max_steps_per_episode = 500
learning_rate = 0.05  # 𝛼 (alpha)
discount_rate = 0.95  # 𝛾 (gamma)
exploration_rate = 1  # ε (epsilon)
max_exploration_rate = 1
min_exploration_rate = 0.00001
exploration_decay_rate = 0.00052

# List to store rewards for each episode
rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    state, info = env.reset()
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

    # Apply exploration rate decay only after "ep_rate" episodes
    ep_rate = 10000 #Change this value to adjust when to start decaying the exploration rate
    if episode < ep_rate:
        exploration_rate = max_exploration_rate
    else:
        exploration_rate = min_exploration_rate + (
        max_exploration_rate - min_exploration_rate
    ) * np.exp(-exploration_decay_rate * (episode - ep_rate))

    # Store total rewards for this episode
    rewards_all_episodes.append(total_rewards)

    # Print progress every "pr_prog" episodes
    pr_prog = 1000 #Change this value to adjust how often to print progress
    if (episode + 1) % pr_prog == 0:
        avg_reward = np.mean(rewards_all_episodes[-pr_prog:])  # Average of last 100 episodes
        print(f"Episode: {episode + 1}, Average Reward: {avg_reward:.3f}, Exploration Rate: {exploration_rate:.5f}")


# Plot the rewards for all episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
average_rewards = [np.mean(rewards) for rewards in rewards_per_thousand_episodes]

plt.plot(range(1, len(average_rewards) + 1), average_rewards)
plt.xlabel('Episodes (in thousands)')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.grid(True)
plt.show()

# Close the environment
env.close()

# Set the render mode to 'human' for visualization of last episode, but use the same environment
env = gym.make('FrozenLake-v1', desc=env_use, is_slippery=False, render_mode='human')
env = CustomFrozenLakeEnv(env)


# Run a few episodes to watch the trained agent in action
for episode in range(2):
    state, info = env.reset()
    done = False
    total_rewards = 0

    print(f"Episode {episode+1}")
    time.sleep(0.25)  # Adjust delay as needed

    while not done:
        # Choose the best action based on the Q-table
        action = np.argmax(q_table[state, :])

        # Take the action (renders automatically)
        state, reward, done, truncated, info = env.step(action)
        total_rewards += reward
        time.sleep(0.25)  # Adjust delay as needed

    print(f"Total rewards: {total_rewards:.3f}")
    time.sleep(1.5)  # Pause before the next episode

env.close()
