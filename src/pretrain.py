"""
Behavioral cloning pre-training from human demonstrations.

Usage:
    python -m src.pretrain --demo-files demos/my_demo.npz --output demos/bc_pretrained.pth

Then start PPO training from the pre-trained checkpoint:
    python main.py --load-checkpoint demos/bc_pretrained.pth
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.environment import MK64Env


def parse_pretrain_args():
    parser = argparse.ArgumentParser(description="Behavioral cloning pre-training for MK64")
    parser.add_argument("--demo-files", type=str, nargs="+", default=None,
                        help="Path(s) to .npz demo files. If not specified, uses all files in demos/")
    parser.add_argument("--output", type=str, default="demos/bc_pretrained.pth",
                        help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--seq-length", type=int, default=8,
                        help="Sequence length for frame stacking (8 for MLP, 64 for transformer)")
    parser.add_argument("--use-transformer", action="store_true", default=False,
                        help="Pre-train a transformer model instead of MLP")
    parser.add_argument("--cuda", action="store_true", default=True)
    return parser.parse_args()


def load_demos(file_paths):
    """Load and concatenate demo files."""
    all_obs = []
    all_actions = []

    for path in file_paths:
        data = np.load(path)
        all_obs.append(data["observations"])
        all_actions.append(data["actions"])
        print(f"  Loaded {path}: {data['observations'].shape[0]} frames")

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    return observations, actions


def create_stacked_dataset(observations, actions, seq_length):
    """
    Create frame-stacked observations from sequential demo data.
    Each sample becomes (seq_length * num_features,) to match the training observation format.
    """
    num_frames = observations.shape[0]
    num_features = observations.shape[1]

    stacked_obs = []
    stacked_actions = []

    for i in range(seq_length - 1, num_frames):
        # Stack the last seq_length frames
        stack = observations[i - seq_length + 1:i + 1]  # (seq_length, num_features)
        stacked_obs.append(stack.flatten())
        stacked_actions.append(actions[i])

    return np.stack(stacked_obs), np.stack(stacked_actions)


class DummyEnvSpaces:
    """Minimal object to satisfy ActorCritic constructor's interface."""
    def __init__(self, obs_dim, action_nvec):
        self._single_observation_space = type('Space', (), {'shape': (obs_dim,)})()
        self._single_action_space = type('Space', (), {'nvec': np.array(action_nvec)})()


def main():
    args = parse_pretrain_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Find demo files
    if args.demo_files:
        demo_files = args.demo_files
    else:
        demo_files = sorted(glob.glob("demos/*.npz"))

    if not demo_files:
        print("No demo files found. Record some demos first with: python -m src.record_demo")
        return

    print(f"Loading {len(demo_files)} demo file(s)...")
    observations, actions = load_demos(demo_files)
    print(f"Total frames: {observations.shape[0]}")

    # Create frame-stacked dataset
    seq_length = args.seq_length if not args.use_transformer else args.seq_length
    print(f"Creating stacked dataset with seq_length={seq_length}...")
    stacked_obs, stacked_actions = create_stacked_dataset(observations, actions, seq_length)
    print(f"Stacked dataset: {stacked_obs.shape[0]} samples, obs_dim={stacked_obs.shape[1]}")

    # Train/val split
    num_samples = stacked_obs.shape[0]
    num_val = int(num_samples * args.val_split)
    num_train = num_samples - num_val

    indices = np.random.permutation(num_samples)
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train_obs = torch.FloatTensor(stacked_obs[train_idx]).to(device)
    train_act = torch.LongTensor(stacked_actions[train_idx]).to(device)
    val_obs = torch.FloatTensor(stacked_obs[val_idx]).to(device)
    val_act = torch.LongTensor(stacked_actions[val_idx]).to(device)

    train_dataset = TensorDataset(train_obs, train_act)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    num_features = MK64Env.NUM_OBS_FEATURES
    obs_dim = seq_length * num_features
    action_nvec = MK64Env.ACTION_DIMS
    dummy_envs = DummyEnvSpaces(obs_dim, action_nvec)

    if args.use_transformer:
        from src.agent import ActorCriticTransformer
        agent = ActorCriticTransformer(dummy_envs, seq_length=seq_length).to(device)
        print(f"Created Transformer model (seq_length={seq_length})")
    else:
        from src.agent import ActorCriticMLP
        agent = ActorCriticMLP(dummy_envs).to(device)
        print("Created MLP model")

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float("inf")
    print(f"\nTraining for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # Train
        agent.train()
        train_loss = 0.0
        train_correct = [0] * len(action_nvec)
        train_total = 0

        for batch_obs, batch_act in train_loader:
            # Forward pass through the network body
            if args.use_transformer:
                hidden = agent._encode(batch_obs)
            else:
                hidden = agent.network(batch_obs)

            # Compute cross-entropy loss for each action head
            loss = torch.tensor(0.0, device=device)
            for i, head in enumerate(agent.actor_heads):
                logits = head(hidden)
                loss = loss + criterion(logits, batch_act[:, i])

                # Track accuracy
                preds = logits.argmax(dim=-1)
                train_correct[i] += (preds == batch_act[:, i]).sum().item()

            train_total += batch_obs.shape[0]
            train_loss += loss.item() * batch_obs.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= train_total

        # Validate
        agent.eval()
        with torch.no_grad():
            if args.use_transformer:
                val_hidden = agent._encode(val_obs)
            else:
                val_hidden = agent.network(val_obs)

            val_loss = 0.0
            val_correct = [0] * len(action_nvec)
            for i, head in enumerate(agent.actor_heads):
                logits = head(val_hidden)
                val_loss += criterion(logits, val_act[:, i]).item()
                preds = logits.argmax(dim=-1)
                val_correct[i] += (preds == val_act[:, i]).sum().item()

        # Print progress
        head_names = ["throttle", "steering", "drift", "item"]
        train_accs = [c / train_total * 100 for c in train_correct]
        val_accs = [c / len(val_idx) * 100 for c in val_correct]

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            for name, ta, va in zip(head_names, train_accs, val_accs):
                print(f"  {name:>10s}: train {ta:.1f}% | val {va:.1f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
            torch.save({
                'update': 0,
                'global_step': 0,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.output)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Pre-trained checkpoint saved to: {args.output}")
    print(f"\nTo start PPO training from this checkpoint:")
    print(f"  python main.py --load-checkpoint {args.output}")


if __name__ == "__main__":
    main()
