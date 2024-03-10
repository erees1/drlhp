import gymnasium as gym

# Create the Hopper environment
env = gym.make("Hopper-v4", render_mode="human")

# Set render mode to humnan

# Run for a few episodes
for _ in range(5):
    observation = env.reset()
    done = False
    while not done:
        # Render the environment
        env.render()

        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation)
        break

# Close the environment
env.close()
