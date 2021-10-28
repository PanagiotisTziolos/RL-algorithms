import gym
import numpy as np
import matplotlib.pyplot as plt

from train import train
from test import test

from ppo import Agent


GAMES = ["CartPole-v0", "LunarLander-v2"]
game = GAMES[0]

def main():
    # Create the environment
    env = gym.make(game)
    
    # Get the state and action space
    state_shape = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # Initialize the agent
    agent = Agent(state_shape, num_actions)
    
    # Train the agent
    rewards_per_episode = train(agent, env, episodes=50)
    
    # Save the rewards to a text file
    save_to_file(rewards_per_episode)
    
    # Test the agent
    test(agent, env)
    
    # Plot the rewards obtained from the episode
    episodes = range(len(rewards_per_episode))
    plt.plot(episodes, rewards_per_episode)
    plt.show()


def save_to_file(rewards_per_episode):
    with open(f"{game}_rewards_per_episode.txt", "w") as rewards_file:
        for r in rewards_per_episode:
            rewards_file.write(str(r) + "\n")
    

if __name__ == "__main__":
    main()


