import numpy as np
import parameters
import torch
import time

from agent import Agent
from env import Env

# Uses NVIDIA-CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# Sets the seed
torch.manual_seed(parameters.SEED)
torch.cuda.manual_seed(parameters.SEED)
np.random.seed(parameters.SEED)


if __name__ == '__main__':

    # Initializes agent and environment
    agent = Agent(device)
    env = Env()

    # Loads parameters from a given episode
    agent.load_paramaters(parameters.LOAD_PARAMETER_EPISODE)

    # Keeps track of time
    time_start = time.time()

    # Trains the agent
    for episode in range(parameters.RENDER_EPISODES + 1):

        # Resets state and variables for each episode
        state = env.reset()
        score = 0

        # Performs steps in an episode
        for _ in range(parameters.MAX_STEPS):

            # Selects and performs action
            action, a_logp = agent.select_action(state)
            next_state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), True)

            # Reward function correction
            if die:
                reward -= 100

            # Increments score
            score += reward

            # Sets new state
            state = next_state

            # Saves recording
            if done or die:
                env.env.stats_recorder.save_complete()
                env.env.stats_recorder.done = True
                break

        # Prints statistics
        if episode % parameters.LOG_INTERVAL == 0:
            print(f'Episode: {episode:<5} Score: {score:<12.2f}')
