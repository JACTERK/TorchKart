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
    The PPO Actor-Critic network with a multi-head (factored) action space.
    It shares a common "body" and has multiple "heads":
    1. The Actor heads (policy), one per action dimension (throttle, steering, drift, item).
    2. The Critic (value), which estimates the state's value.
    """

    def __init__(self, envs):
        super().__init__()
        obs_shape = envs._single_observation_space.shape

        # Get the sizes of each action dimension from MultiDiscrete
        # e.g., [3, 5, 2, 2] for [throttle, steering, drift, item]
        self.action_nvec = envs._single_action_space.nvec

        # The "body" of the network (shared feature extractor)
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(64, 64, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # The Critic head (value function)
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # One Actor head per action dimension
        self.actor_heads = nn.ModuleList([
            layer_init(nn.Linear(64, n), std=0.01)
            for n in self.action_nvec
        ])

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        """
        Gets the estimated value of a state.
        """
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic_head(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        """
        Gets an action (and its log-probability) and the state value.
        If an action is provided, it also returns the log-prob and entropy
        of that action (used during training).

        Returns:
            action: Tensor of shape (batch, num_heads) — one index per head
            log_prob: Tensor of shape (batch,) — sum of log-probs across heads
            entropy: Tensor of shape (batch,) — sum of entropies across heads
            value: Tensor of shape (batch, 1) — the estimated state value
            head_entropies: Tensor of shape (batch, num_heads) — entropy per head (for logging)
            lstm_state: Tuple of updated (hidden_state, cell_state) tensors
        """
        hidden, lstm_state = self.get_states(x, lstm_state, done)

        # Build a Categorical distribution for each head
        multi_logits = [head(hidden) for head in self.actor_heads]
        multi_dists = [Categorical(logits=logits) for logits in multi_logits]

        if action is None:
            # Sample from each head independently
            action = torch.stack([d.sample() for d in multi_dists], dim=-1)

        # Compute log-prob and entropy per-head, then sum
        log_prob = torch.stack(
            [d.log_prob(action.reshape(-1, len(self.action_nvec))[:, i]) for i, d in enumerate(multi_dists)], dim=-1
        ).sum(dim=-1)

        head_entropies = torch.stack(
            [d.entropy() for d in multi_dists], dim=-1
        )
        entropy = head_entropies.sum(dim=-1)

        if action.ndim == 3: # (seq_len, batch_size, num_heads)
            log_prob = log_prob.reshape(*action.shape[:2])
            entropy = entropy.reshape(*action.shape[:2])

        value = self.critic_head(hidden)

        return action, log_prob, entropy, value, head_entropies, lstm_state
