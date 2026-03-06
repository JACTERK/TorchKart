import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes weights with orthogonal init and biases to zero.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    The PPO Actor-Critic network.
    It shares a common "body" and has two "heads":
    1. The Actor (policy), which outputs action probabilities.
    2. The Critic (value), which estimates the state's value.
    """

    def __init__(self, envs):
        super().__init__()
        obs_shape = envs._single_observation_space.shape
        num_actions = envs._single_action_space.n

        # The "body" of the network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # The "head" for the Critic (value)
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            # The "head" for the Actor (policy)
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )

    def get_value(self, x):
        """
        Gets the estimated value of a state.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Gets an action (and its log-probability) and the state value.
        If an action is provided, it also returns the log-prob and entropy
        of that action (used during training).
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)
