from plot import Plot
import numpy as np
import parameters
import torch
import time

from agent import Agent
from env import Env

from collections import deque

# Uses NVIDIA-CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# Sets the seed
torch.manual_seed(parameters.SEED)
torch.cuda.manual_seed(parameters.SEED)
np.random.seed(parameters.SEED)


# Calculates the current timestamp
def get_timestamp(time_start):
    time_diff = int(time.time() - time_start)

    h = time_diff // 3600
    m = time_diff % 3600 // 60
    s = time_diff % 60

    return f'{h:02}:{m:02}:{s:02}'


if __name__ == '__main__':

    # Initializes agent and environment
    agent = Agent(device)
    env = Env()

    # Loads parameters from a given episode
    if parameters.LOAD_PARAMETERS:
        agent.load_paramaters(parameters.LOAD_PARAMETER_EPISODE)

    # Scores
    scores = deque(maxlen=100)
    running_score = 0
    average_scores = []

    # Keeps track of time
    time_start = time.time()

    # Trains the agent
    for episode in range(parameters.MAX_EPISODES):

        # Resets state and variables for each episode
        state = env.reset()
        timestep = 0
        score = 0

        # Performs steps in an episode
        while True:

            # Selects and performs action
            action, a_logp = agent.select_action(state)
            next_state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), False)

            # Stores the data
            if agent.store((state, action, a_logp, reward, next_state)):
                print('\nUpdating\n')
                agent.update()

            # Reward function correction
            if die:
                reward -= 100

            # Increments score and timestep
            score += reward
            timestep += 1

            # Sets new state
            state = next_state

            if done or die:
                env.env.stats_recorder.save_complete()
                env.env.stats_recorder.done = True
                break

        # Calculates scores
        scores.append(score)
        average_score = np.mean(scores)
        average_scores.append(average_score)
        running_score = running_score * 0.99 + score * 0.01

        # Prints statistics
        if episode % parameters.LOG_INTERVAL == 0:
            print(f'Episode: {episode:<5} Timesteps: {timestep:<8} Score: {score:<12.2f} Average score: '
                  f'{average_score:<12.2f} Running score: {running_score:<12.2f}Time: {get_timestamp(time_start)}')

        # Saves parameters
        if episode % parameters.SAVE_INTERVAL == 0:
            agent.save_parameters(episode)

        # Saves plot
        if episode % parameters.PLOT_INTERVAL == 0:
            Plot.save(average_scores, 'Baseline')

        # Stops if the reward threshold is met
        if average_score > env.reward_threshold:
            print(f'\nSolved! Episode: {episode}')
            break
