# Plot the rewards for all episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 100)
average_rewards = [np.mean(rewards) for rewards in rewards_per_thousand_episodes]

plt.plot(range(1, len(average_rewards) + 1), average_rewards)
plt.xlabel('Episodes (times 100)')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.grid(True)
plt.show()