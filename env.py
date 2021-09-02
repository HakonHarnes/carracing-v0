import matplotlib.pyplot as plt

import parameters
import numpy as np
import gym

if parameters.GRAYSCALE:
    channels = 1
else:
    channels = 3


class Env:
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self):
        self.env = gym.make('CarRacing-v0', verbose=0)
        self.env = gym.wrappers.Monitor(self.env, 'video', video_callable=lambda episode_id: episode_id % parameters.VIDEO_INTERVAL == 0, force=True)
        self.env.seed(parameters.SEED)

        self.die = False
        self.stack = []
        self.counter = 0
        self.av_r = self.reward_memory()
        self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.die = False
        self.counter = 0
        self.av_r = self.reward_memory()
        self.stack = []

        rgb = self.env.reset()
        rgb = self.remove_bar(rgb)

        # Adds the state to the stack
        if parameters.GRAYSCALE:
            gray = self.rgb_to_gray(rgb)
            self.stack = [gray] * parameters.FRAME_STACK
        else:
            for _ in range(parameters.FRAME_STACK):
                for channel in range(channels):
                    self.stack.append(rgb[:, :, channel])

        return np.array(self.stack)

    def step(self, action, render):
        total_reward = 0

        for _ in range(parameters.ACTION_REPEAT):
            rgb, reward, die, _ = self.env.step(action)
            
            if parameters.REDUCED_INPUT:
                rgb = self.remove_bar(rgb)

            # Don't penalize 'die state'
            if die:
                reward += 100

            # Penalize for hitting the grass
            if np.mean(rgb[:, :, 1]) > 185.0:
                reward -= 0.05

            total_reward += reward

            # Renders the step
            if render:
                self.render()

            # If no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break

        # Converts the state to grayscale
        if parameters.GRAYSCALE:
            gray = self.rgb_to_gray(rgb)
            self.stack.pop(0)
            self.stack.append(gray)
        else:
            for channel in range(channels):
                self.stack.pop(0)
                self.stack.append(rgb[:, :, channel])

        # Checks that the length of the stack is correct
        assert len(self.stack) == parameters.FRAME_STACK * channels

        # Returns state, reward, done and die
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def remove_bar(rgb):
        no_bar = np.zeros(rgb.shape, dtype=int)
        no_bar[:83, :, :] = rgb[:83, :, :]
        return no_bar

    # Converts RGB state to grayscale
    @staticmethod
    def rgb_to_gray(rgb, norm=True):
        gray = np.dot(rgb, [0.299, 0.587, 0.114])

        # Normalize
        if norm:
            gray = gray / 128. - 1.

        return gray

    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory
