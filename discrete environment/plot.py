import gym
import numpy as np
import matplotlib.pyplot as plt

"""
The directory should be like this
"results/{algorithm}/{game}/iteration{i}.txt".

"algorithm" and "game" are items 
from the lists "ALGORITHMS" and "GAMES".

The algorithm iterates over all games and
algorithms.

TRAINING_TIMES defines the number of times
that the agent of a single algorithm was trained
and the number of .txt files that contain the 
rewards for each episode.

iteration{i}.txt is the i_th .txt file that
contains the rewards of each episode.

For example, if the agent was trained for
4 times on the same environmet (game) then
TRAINING_TIMES = 4 and the .txt files that
contain the rewards for each episode are
iteration0.txt, iteration1.txt, 
iteration3.txt, iteration3.txt,

smooth defines the maximum window size 
that is used to calculate the moving average
the minimum window size is 1

plt.fill_between creates the shading around
the curve
"""


TRAINING_TIMES = 4

GAMES = ["CartPole-v0"] # , "LunarLander-v2"

ALGORITHMS = ['dqn', 'reinforce', 'a2c', 'ppo']

COLORS = ["green", "orange", "red", "purple"]

def calculate_moving_average_and_std(rewards , smooth=10):
    ma = []
    std = []
    for i in range(len(rewards)):
        d = 0
        if i < smooth-1:
            elements = rewards[: i+1]
            moving_average = np.mean(elements)
            for e in elements:
                d += pow(e - moving_average, 2)
            d = np.sqrt(d / (i+1))
            ma.append(moving_average)
            std.append(d)
        else:
            elements = rewards[i+1-smooth : i+1]
            moving_average = np.mean(elements)
            for e in elements:
                d += pow(e - moving_average, 2)
            d = np.sqrt(d / smooth)
            ma.append(moving_average)
            std.append(d)

    ma_m, ma_p = [], []
    for r, d in zip(ma, std):
        ma_m.append(r - d)
        ma_p.append(r + d)
        
    return ma, ma_m, ma_p
    
    
def get_mean_and_std_of_rewards():
    rewards_per_game = []
    
    for game in GAMES:
        rewards_per_algorithm = []

        for algorithm in ALGORITHMS:
            rewards_per_episode = []

            for i in range(TRAINING_TIMES):
                rewards = []
                
                with open(f"results/{algorithm}/{game}/iteration{i}.txt", "r") as rewards_file:
                    for index, r in enumerate(rewards_file):
                        rewards.append(float(r.strip("\n")))
                
                    rewards_per_episode.append(rewards)
                    
            mean_rewards = np.mean(rewards_per_episode, axis=0)
            ma, ma_m, ma_p = calculate_moving_average_and_std(mean_rewards)
            rewards_per_algorithm.append((ma, ma_m, ma_p))
            
        rewards_per_game.append(rewards_per_algorithm)
        
    rewards_dic = {}
    
    for index, game in enumerate(GAMES):
        rewards_dic[game] = rewards_per_game[index]
        
    return rewards_dic


def plot_results(rewards_dic):
    for game in GAMES:
        r = rewards_dic[game]
        
        rewards = []
        
        for index in range(len(ALGORITHMS)):
            mean_av_rewards = r[index][0]
            std_minus = r[index][1]
            std_plus = r[index][2]
            rewards.append((mean_av_rewards, std_minus, std_plus))
        
        episodes = range(len(rewards[0][0]))
        
        for index, alg in enumerate(ALGORITHMS):
            plt.plot(episodes, rewards[index][0], label=alg, color=COLORS[index])
            plt.fill_between(episodes,
                            rewards[index][1],
                            rewards[index][2],
                            alpha=0.4, color=COLORS[index])
                            
        plt.legend()

        plt.savefig(f"{game} plots")
        
        plt.show()

if __name__ == "__main__":
    plot_results(get_mean_and_std_of_rewards())