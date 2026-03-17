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
    PPO Actor-Critic with a factored multi-head action space and LSTM for the actor.

    Architecture:
      Actor path:  obs -> network (MLP 128) -> lstm (128) -> actor_heads
      Critic path: obs -> critic_mlp (128->64) -> critic_head

    The critic has its own dedicated MLP and bypasses the LSTM entirely.
    This prevents the actor and critic from fighting over the LSTM representation
    and gives the value function a clean, uncontested gradient path.
    """

    def __init__(self, envs):
        super().__init__()
        obs_shape = envs._single_observation_space.shape
        obs_dim = int(np.array(obs_shape).prod())

        # Get the sizes of each action dimension from MultiDiscrete
        # e.g., [3, 5, 2, 2] for [throttle, steering, drift, item]
        self.action_nvec = envs._single_action_space.nvec

        # --- Actor path ---

        # Shared MLP body feeding into the LSTM
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
        )

        # Single-layer LSTM: 128 -> 128
        self.lstm = nn.LSTM(128, 128, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        # Actor heads: one per action dimension (128 -> n_i)
        self.actor_heads = nn.ModuleList([
            layer_init(nn.Linear(128, n), std=0.01)
            for n in self.action_nvec
        ])

        # --- Critic path (no LSTM) ---

        # Dedicated MLP that reads directly from raw observations
        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
        )

        # Critic head: 64 -> 1
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

    def get_states(self, x, lstm_state, done):
        """Runs the actor MLP + LSTM, returning the LSTM output and updated state."""
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
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],  # zero h on done
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],  # zero c on done
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        """
        Estimates state value using the dedicated critic MLP (no LSTM).
        lstm_state and done are accepted for interface compatibility but not used.
        """
        x_flat = x.reshape(-1, x.shape[-1])
        return self.critic_head(self.critic_mlp(x_flat))

    def get_action_and_value(self, x, lstm_state, done, action=None):
        """
        Actor uses the MLP + LSTM path.
        Critic uses the dedicated critic_mlp path (no LSTM).

        Returns:
            action:        (batch, num_heads)
            log_prob:      (batch,)
            entropy:       (batch,)
            value:         (batch, 1)
            head_entropies:(batch, num_heads)
            lstm_state:    updated (h, c) tuple
        """
        # Actor path: MLP -> LSTM -> heads
        hidden, lstm_state = self.get_states(x, lstm_state, done)

        multi_logits = [head(hidden) for head in self.actor_heads]
        multi_dists = [Categorical(logits=logits) for logits in multi_logits]

        if action is None:
            action = torch.stack([d.sample() for d in multi_dists], dim=-1)

        log_prob = torch.stack(
            [d.log_prob(action.reshape(-1, len(self.action_nvec))[:, i])
             for i, d in enumerate(multi_dists)], dim=-1
        ).sum(dim=-1)

        head_entropies = torch.stack([d.entropy() for d in multi_dists], dim=-1)
        entropy = head_entropies.sum(dim=-1)

        if action.ndim == 3:  # (seq_len, batch_size, num_heads)
            log_prob = log_prob.reshape(*action.shape[:2])
            entropy = entropy.reshape(*action.shape[:2])

        # Critic path: dedicated MLP, bypasses LSTM
        x_flat = x.reshape(-1, x.shape[-1])
        value = self.critic_head(self.critic_mlp(x_flat))

        return action, log_prob, entropy, value, head_entropies, lstm_state
