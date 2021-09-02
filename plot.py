import matplotlib.pyplot as plt
import numpy as np


class Plot:
    """
    Class for drawing and saving plots
    """

    @staticmethod
    def save(data, title):
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        plt.plot(data, label='Score')
        plt.savefig(f'data/{title.lower()}_plot')
        np.save('data/average_scores.npy', data)
