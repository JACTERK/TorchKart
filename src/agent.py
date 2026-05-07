import math

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


class ActorCriticMLP(nn.Module):
    """
    The original PPO Actor-Critic network with a multi-head (factored) action space.
    Uses a simple MLP with frame stacking for temporal context.
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

        # The Critic head (value function)
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # One Actor head per action dimension
        self.actor_heads = nn.ModuleList([
            layer_init(nn.Linear(64, n), std=0.01)
            for n in self.action_nvec
        ])

    def get_value(self, x):
        return self.critic_head(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)

        multi_logits = [head(hidden) for head in self.actor_heads]
        multi_dists = [Categorical(logits=logits) for logits in multi_logits]

        if action is None:
            action = torch.stack([d.sample() for d in multi_dists], dim=-1)

        log_prob = torch.stack(
            [d.log_prob(action[:, i]) for i, d in enumerate(multi_dists)], dim=-1
        ).sum(dim=-1)

        head_entropies = torch.stack(
            [d.entropy() for d in multi_dists], dim=-1
        )
        entropy = head_entropies.sum(dim=-1)

        value = self.critic_head(hidden)

        return action, log_prob, entropy, value, head_entropies


class ActorCriticTransformer(nn.Module):
    """
    A Transformer-based Actor-Critic network for long-term temporal reasoning.
    Uses a causal transformer to process a sequence of observation frames,
    replacing frame stacking with attention-based temporal context.
    """

    def __init__(self, envs, seq_length=64, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        obs_shape = envs._single_observation_space.shape
        self.action_nvec = envs._single_action_space.nvec

        self.seq_length = seq_length
        self.embed_dim = embed_dim

        # Calculate per-frame feature count from total obs shape
        # obs_shape is (seq_length * num_features,) when flattened
        total_obs_dim = np.array(obs_shape).prod()
        self.num_features = total_obs_dim // seq_length

        # Input projection: per-frame features -> embed_dim
        self.input_proj = nn.Linear(self.num_features, embed_dim)

        # Learned positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim) * 0.02)

        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        )

        # Post-transformer projection
        self.post_proj = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim)),
            nn.ReLU(),
        )

        # Critic head
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(embed_dim, 1), std=1.0),
        )

        # Actor heads
        self.actor_heads = nn.ModuleList([
            layer_init(nn.Linear(embed_dim, n), std=0.01)
            for n in self.action_nvec
        ])

    def _encode(self, x):
        """
        Encodes a flat observation vector through the transformer.
        x: (batch, seq_length * num_features) flat vector
        Returns: (batch, embed_dim) hidden representation from the last timestep
        """
        batch_size = x.shape[0]

        # Reshape from flat to sequence: (batch, seq_length, num_features)
        x = x.view(batch_size, self.seq_length, self.num_features)

        # Project each frame to embed_dim
        x = self.input_proj(x)  # (batch, seq_length, embed_dim)

        # Add positional encoding
        x = x + self.pos_embedding

        # Apply causal transformer
        x = self.transformer(x, mask=self.causal_mask)  # (batch, seq_length, embed_dim)

        # Take the last position (the "current" timestep)
        x = x[:, -1, :]  # (batch, embed_dim)

        # Post-projection
        x = self.post_proj(x)

        return x

    def get_value(self, x):
        hidden = self._encode(x)
        return self.critic_head(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self._encode(x)

        multi_logits = [head(hidden) for head in self.actor_heads]
        multi_dists = [Categorical(logits=logits) for logits in multi_logits]

        if action is None:
            action = torch.stack([d.sample() for d in multi_dists], dim=-1)

        log_prob = torch.stack(
            [d.log_prob(action[:, i]) for i, d in enumerate(multi_dists)], dim=-1
        ).sum(dim=-1)

        head_entropies = torch.stack(
            [d.entropy() for d in multi_dists], dim=-1
        )
        entropy = head_entropies.sum(dim=-1)

        value = self.critic_head(hidden)

        return action, log_prob, entropy, value, head_entropies


# Backward-compatible alias
ActorCritic = ActorCriticMLP
