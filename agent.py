import torch
import torch.optim as optim
import torch.nn.functional as f
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from model import Net
import numpy as np
import parameters

if parameters.GRAYSCALE:
    channels = 1
else:
    channels = 3

transition = np.dtype([('s', np.float64, (parameters.FRAME_STACK * channels, 96, 96)),
                       ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (parameters.FRAME_STACK * channels, 96, 96))])


class Agent:
    """
    Agent for training
    """
    
    def __init__(self, device):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(parameters.BUFFER_SIZE, dtype=transition)
        self.counter = 0
        self.device = device
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=parameters.LEARNING_RATE)

    # Selects an action given a state
    def select_action(self, state):

        # Converts shape: (4, 96, 96) -> (1, 4, 96, 96)
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)

        # Gets alpha and beta parameters
        with torch.no_grad():
            alpha, beta = self.net(state)[0]

        dist = Beta(alpha, beta)

        # Calculates action and log probability
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        # Reshapes action and log probability
        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()

        # Returns action and log probability
        return action, a_logp

    # Stores transition in buffer
    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1

        # If buffer is full, start writing to the start of the buffer again
        if self.counter == parameters.BUFFER_SIZE:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        # Allocates buffer space
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        next_s = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + parameters.GAMMA * self.net(next_s)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(parameters.EPOCHS):
            for index in BatchSampler(SubsetRandomSampler(range(parameters.BUFFER_SIZE)), parameters.BATCH_SIZE, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - parameters.EPSILON, 1.0 + parameters.EPSILON) * adv[index]

                # Calculates losses
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = f.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # Saves parameters to file
    def save_parameters(self, episode):
        torch.save(self.net.state_dict(), f'data/parameters_{episode}.pkl')

    # Loads parameters from file
    def load_paramaters(self, episode):
        self.net.load_state_dict(torch.load(f'data/parameters_{episode}.pkl'))
        self.net.eval()


