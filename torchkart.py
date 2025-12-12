import socket
import struct
import time
import argparse
import os
from typing import List, Tuple, Dict, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Box, Discrete
from tqdm import tqdm


# CLI Args
def parse_args():
    parser = argparse.ArgumentParser(description="PPO Trainer for Mario Kart 64")

    # Environment Args
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel BizHawk clients to connect to.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host IP address to bind the server to.")
    parser.add_argument("--port", type=int, default=65432,
                        help="Port to listen on.")

    # PPO Hyperparameters
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="The learning rate of the optimizer.")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="Number of steps to run in each environment per policy rollout.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="The discount factor gamma.")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="The lambda for the GAE calculation.")
    parser.add_argument("--num-minibatches", type=int, default=32,
                        help="The number of mini-batches.")
    parser.add_argument("--update-epochs", type=int, default=10,
                        help="The K epochs to update the policy.")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="The surrogate clipping coefficient.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Coefficient of the entropy.")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Coefficient of the value function.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="The maximum norm for the gradient clipping.")

    # Training Args
    parser.add_argument("--total-timesteps", type=int, default=5_000_000,
                        help="Total timesteps of the experiments.")
    parser.add_argument("--torch-deterministic", type=bool, default=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=True`")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="if toggled, cuda will be enabled by default")

    # Misc Configuration
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save a model checkpoint every N updates.")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to a .pth checkpoint file to load and resume training.")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


# --- Gymnasium Environment ---
class MK64Env(gym.vector.VectorEnv):
    """
    A Gymnasium Environment that acts as a server
    and manages multiple BizHawk socket clients.
    """
    # Total bytes to read, based on the Lua script's MEMORY_MAP
    # 5 * 4 (float) + 4 * 4 (int) + 1 * 2 (short uint) = 38
    STATE_SIZE_BYTES = 38

    # The struct format string for parsing the 46 bytes.
    # '>' means big-endian (which N64 is)
    # f = 4-byte float
    # i = 4-byte signed int
    # H = 2-byte unsigned short
    # Fixed-Point 16.16 is read as 'i' (signed int)
    STATE_FORMAT = ">fffiiHiiff"

    # The names for each raw value read from the struct
    STATE_NAMES = [
        "x_vel",
        "y_vel",
        "y_vel",
        "path_progress",
        "lap",
        "orientation",
        "wall_1",
        "wall_2",
        "track_center_dist",
        "speed"
    ]

    # The number of processed features sent to the network (some are combined)
    NUM_OBS_FEATURES = 9

    def __init__(self, num_envs, host, port, stack_size=8):
        # Define how many frames to "stack" for temporal awareness
        self.stack_size = stack_size

        # Observation dimensionality (Number of features * the stack size)
        obs_dim = self.NUM_OBS_FEATURES * self.stack_size

        # Define the observation and action spaces for Gymnasium
        single_observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # What discrete actions can the agent do (Defined in LUA)
        single_action_space = Discrete(9)

        # Call the base object's __init__
        super().__init__()

        # Set the attributes used by Gymnasium
        self.clients = []
        self.num_envs = num_envs
        self._single_observation_space = single_observation_space
        self._single_action_space = single_action_space

        # Set the attributes used for other parts of the program
        self.stuck_counter = np.zeros(num_envs, dtype=np.int32)
        self.obs_stacks = np.zeros((num_envs, self.stack_size, self.NUM_OBS_FEATURES), dtype=np.float32)

        # Set the attributes used to connect to BizHawk sockets
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(num_envs)

        print(f"Server listening on {host}:{port}...")
        print(f"Waiting for {num_envs} BizHawk clients to connect.")

        # Wait for all num_envs clients to connect
        for i in range(self.num_envs):
            conn, addr = self.server_socket.accept()
            conn.settimeout(300)  # 30-second timeout
            self.clients.append(conn)
            print(f"Client {i + 1}/{num_envs} connected from {addr}")

        print("All clients connected! Starting training.")

        # Storage for calculating rewards
        self.last_state_dicts = [{} for _ in range(num_envs)]
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)


    # Getter for getting stacked observations
    def _get_stacked_obs(self):
        # Flatten the last two dimensions: (Num_Envs, Stack_Size * Features)
        return self.obs_stacks.reshape(self.num_envs, -1)

    def _parse_and_preprocess(self, state_bytes: bytes, old_wall_values: Tuple[int, int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Takes the raw state bytes, and preprocesses it for the network. (Network likes when the values are between
        -1 and 1).
        Also turns the raw state bytes into a dictionary for the reward function.
        """
        # Try to unpack the state, error if unsuccessful
        try:
            raw_values = struct.unpack(self.STATE_FORMAT, state_bytes)
        except struct.error:
            # Return zero state on error
            return np.zeros(self.NUM_OBS_FEATURES, dtype=np.float32), {}

        state = dict(zip(self.STATE_NAMES, raw_values))

        # --- Normalize values for network ---

        # Velocities: Range -5 to 5 -> Map to -1.0 to 1.0
        norm_x_vel = state["x_vel"] / 5.0
        norm_y_vel = state["y_vel"] / 5.0

        # Speed: Range 0 to 67 -> Map to 0.0 to 1.0
        norm_speed = state["speed"] / 67.0

        # Orientation: Range 0 to 65536. Convert to Sin/Cos components
        angle_radians = (state["orientation"] / 65536.0) * 2 * np.pi
        norm_sin_angle = np.sin(angle_radians)
        norm_cos_angle = np.cos(angle_radians)

        # Center Distance: Range -1 to 1 (Already within that range)
        norm_center = state["track_center_dist"]

        # Progress: Range 0 to 1890 -> Map to 0.0 to ~1.0
        norm_progress = state["path_progress"] / 1900.0

        # Lap: Range 0 to 3 -> Map to 0.0 to 1.0
        norm_lap = state["lap"] / 3.0

        # --- Wall Hit Logic ---
        # If Mario hits the wall, the current value will differ from the old value.
        wall_hit = 0.0
        if old_wall_values is not None:
            old_w1, old_w2 = old_wall_values
            # Check if raw values changed
            if state["wall_1"] != old_w1 or state["wall_2"] != old_w2:
                wall_hit = 1.0

        # --- Build Observation Vector ---
        obs = np.array([
            norm_x_vel,
            norm_y_vel,
            norm_speed,
            norm_sin_angle,
            norm_cos_angle,
            norm_center,
            wall_hit,
            norm_progress,
            norm_lap
        ], dtype=np.float32)

        # Return processed state dict for reward calculation later
        processed_state_dict = {
            "progress": state["path_progress"],
            "lap": state["lap"],
            "speed": state["speed"],
            "track_center_dist": state["track_center_dist"],
            "wall_hit": wall_hit,  # Boolean 0.0 or 1.0
            "raw_wall_1": state["wall_1"],
            "raw_wall_2": state["wall_2"]
        }

        return obs, processed_state_dict

    def _calculate_reward(self, old_state: Dict, new_state: Dict) -> Tuple[float, bool, bool]:
        """
        Calculates the reward based on the change in state.
        """
        # Small penalty for existing
        reward = -0.1

        # Keep track of if the agent has been terminated or if the race is over
        terminated = False
        is_finished_race = False

        if not old_state:
            # No old state to compare to (first frame after reset)
            return reward, terminated, is_finished_race

        # Reward for forward progress, punish equivalently for negative progress
        progress_delta = new_state["progress"] - old_state["progress"]
        reward += progress_delta * 0.5

        # Reward for speed (Between 0 and 67)
        speed = new_state["speed"]

        if speed > 67:
            reward += (speed / 67.0) * 0.5

            # Punishment for hitting a wall at full speed
            if new_state["wall_hit"] > 0.5:
                reward -= 5.0

        else:
            reward += (speed / 67.0) * 0.5

            # Punishment for hitting a wall at full speed
            if new_state["wall_hit"] > 0.5:
                reward -= 2.0

        # # Punishment for going off the track
        # deviation = abs(new_state["track_center_dist"])
        # if deviation > 1.0:
        #     reward += (1.0 - deviation) * 0.125

        # Check for termination (finished race)
        if new_state["lap"] >= 3:
            reward += 100.0  # Big bonus for finishing!
            terminated = True
            is_finished_race = True

        return float(reward), terminated, is_finished_race

    def reset_at(self, index: int) -> Tuple[np.ndarray, Dict]:
        """
        Resets a single environment at the given index.
        This can happen if the agent gets stuck, or if the race is finished.
        """
        try:
            # Send 'R' (Reset) command
            self.clients[index].sendall(b'R')
            # Receive state from emulator
            state_bytes = self.clients[index].recv(self.STATE_SIZE_BYTES)

            if len(state_bytes) < self.STATE_SIZE_BYTES:
                raise ConnectionAbortedError(f"Client {index} sent incomplete state on reset.")

            # Preprocess the state
            obs, state_dict = self._parse_and_preprocess(state_bytes, old_wall_values=None)

            # Flush the observation stack
            for i in range(self.stack_size):
                self.obs_stacks[index, i] = obs

            # Flatten the stack to 1D array
            stacked_obs = self.obs_stacks[index].flatten()

            self.last_state_dicts[index] = state_dict
            self.episode_rewards[index] = 0.0
            self.episode_lengths[index] = 0
            self.stuck_counter[index] = 0

            return stacked_obs, {}

        except (socket.timeout, ConnectionAbortedError, ConnectionResetError) as e:
            print(f"Error resetting client {index}: {e}. Stopping.")
            self.close()
            raise e

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets all environments.
        """
        obs_list = []
        for i in range(self.num_envs):
            obs, _ = self.reset_at(i)
            obs_list.append(obs)

        return np.stack(obs_list), {}

    def step(self, s_actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Steps all environments with the given actions.
        """
        obs_list = []
        rew_list = []
        term_list = []
        trunc_list = []
        info_list = []

        try:
            # Send action command to all clients
            for i, action_id in enumerate(s_actions):
                # Send 'S' (Step) command + 1-byte action
                cmd = b'S' + bytes([action_id])
                self.clients[i].sendall(cmd)

            # Receive new state from all clients
            for i in range(self.num_envs):
                state_bytes = self.clients[i].recv(self.STATE_SIZE_BYTES)

                if len(state_bytes) < self.STATE_SIZE_BYTES:
                    raise ConnectionAbortedError(f"Client {i} sent incomplete state on step.")

                # Process state and calculate reward
                old_state = self.last_state_dicts[i]
                old_walls = None
                if old_state:
                    old_walls = (old_state["raw_wall_1"], old_state["raw_wall_2"])

                new_frame, new_state = self._parse_and_preprocess(state_bytes, old_walls)

                # Shift existing frames: [0,1,2,3] -> [1,2,3,0]
                self.obs_stacks[i] = np.roll(self.obs_stacks[i], shift=-1, axis=0)

                # Overwrite the last element with new data
                self.obs_stacks[i, -1] = new_frame

                # Create the flat vector for the network
                stacked_obs = self.obs_stacks[i].flatten()

                reward, terminated, is_finished_race = self._calculate_reward(old_state, new_state)
                truncated = False # Might add a time limit, for now the stuck check seems to work fine.

                # Check if player is stuck
                if old_state:
                    # Check progress, accounting for lap crossovers
                    progress_delta = new_state["progress"] - old_state["progress"]
                    if new_state["lap"] > old_state["lap"]:
                        progress_delta += 1000  # Made positive progress

                    if progress_delta < 0.1:  # Not making meaningful progress
                        self.stuck_counter[i] += 1
                    else:
                        self.stuck_counter[i] = 0  # Reset counter, we're moving

                # Check if stuck for too long (600 steps = 10 seconds at 60fps) @TODO Make this lower
                if self.stuck_counter[i] > 60:
                    terminated = True  # End the episode
                    reward -= 20.0  # Apply a large penalty for being stuck
                    self.stuck_counter[i] = 0  # Reset counter for next episode
                    is_finished_race = False

                # Update episode trackers
                self.last_state_dicts[i] = new_state
                self.episode_rewards[i] += reward
                self.episode_lengths[i] += 1

                # Handle "done" state (terminated or truncated)
                info = {}
                if terminated or truncated:
                    ep_len = self.episode_lengths[i]

                    info["final_info"] = {
                        "episode": {
                            "r": self.episode_rewards[i],
                            "l": ep_len,
                            "final_lap": new_state.get("lap", -99)
                        }
                    }
                    if is_finished_race:
                        # Assuming 60 steps/sec based on stuck counter
                        race_time_seconds = ep_len / 60.0
                        info["final_info"]["episode"]["race_time"] = race_time_seconds

                    # Auto-reset this environment
                    stacked_obs, _ = self.reset_at(i)

                # Append results
                obs_list.append(stacked_obs)
                rew_list.append(reward)
                term_list.append(terminated)
                trunc_list.append(truncated)
                info_list.append(info)

            # Convert lists to stacked numpy arrays
            return (
                np.stack(obs_list),
                np.array(rew_list, dtype=np.float32),
                np.array(term_list, dtype=np.bool_),
                np.array(trunc_list, dtype=np.bool_),
                info_list,
            )

        except (socket.timeout, ConnectionAbortedError, ConnectionResetError) as e:
            print(f"Error during step: {e}. Stopping.")
            self.close()
            raise e

    def close(self):
        """
        Sends the 'C' (Close) command to all clients and closes sockets.
        """
        print("Closing connections...")
        for client in self.clients:
            try:
                client.sendall(b'C')
                client.close()
            except Exception as e:
                print(f"Error closing a client: {e}")
        self.server_socket.close()
        print("Server shut down.")


# --- PPO Agent ---

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


# --- Main Training Loop ---

if __name__ == "__main__":
    args = parse_args()

    # Setup for saving data (checkpoints and TensorBoard data)
    run_name = f"MK64_PPO_{int(time.time())}"
    run_dir = f"runs/{run_name}"
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    np.random.seed(0)
    torch.manual_seed(0)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    # --- Setup Environment ---
    # Pause the script and wait for num_envs clients to connect
    envs = MK64Env(num_envs=args.num_envs, host=args.host, port=args.port)

    # --- Setup Agent ---
    agent = ActorCritic(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Setup Storage ---
    # This buffer will store the rollouts
    obs = torch.zeros((args.num_steps, args.num_envs) + envs._single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs._single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # --- Start Training ---
    global_step = 0
    start_update = 1

    start_time = time.time()

    # Load from checkpoint if one is defined
    if args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            print(f"Loading checkpoint from {args.load_checkpoint}")
            checkpoint = torch.load(args.load_checkpoint, map_location=device)
            agent.load_state_dict(checkpoint['agent_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_update = checkpoint['update'] + 1
            global_step = checkpoint['global_step']
            print(f"Resuming from update {start_update} (global_step {global_step})")
        else:
            print(f"Checkpoint file not found: {args.load_checkpoint}. Starting from scratch.")

    # Get initial observation
    # Note: `envs.reset()` returns a tuple (obs, info)
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size

    print(f"Starting training for {num_updates} updates...")

    try:
        for update in range(start_update, num_updates + 1):

            print(f"\n--- Update {update}/{num_updates} ---")

            # --- Collect Rollouts ---
            pbar = tqdm(range(0, args.num_steps), desc="Collecting Rollout")
            for step in pbar:
                global_step += 1 * args.num_envs

                obs[step] = next_obs
                dones[step] = next_done

                # Get action from the agent
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # Send action to the environment
                # `step` returns: next_obs, reward, terminated, truncated, info
                next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

                # Handle `done`
                done = np.logical_or(terminated, truncated)

                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(done).to(device)

                # Check for final episode info (logged on `terminated` or `truncated`)
                for i, item in enumerate(info):
                    if "final_info" in item and item["final_info"] is not None:
                        ep_info = item["final_info"]["episode"]
                        ep_reward = ep_info['r']
                        ep_len = ep_info['l']
                        ep_final_lap = ep_info.get("final_lap", -99)

                        print(
                            f"  [Env {i}] Episode Finish. Reward: {ep_reward:.2f}, Length: {ep_len}, Final Lap: {ep_final_lap}")

                        writer.add_scalar("charts/episodic_return", ep_reward, global_step)
                        writer.add_scalar("charts/episodic_length", ep_len, global_step)
                        writer.add_scalar("charts/final_lap", ep_final_lap, global_step)

                        if "race_time" in ep_info:
                            race_time = ep_info["race_time"]
                            print(f"  [Env {i}] *** RACE COMPLETED *** Time: {race_time:.2f}s")
                            writer.add_scalar("charts/completed_race_time", race_time, global_step)

                        break  # Only log one per step to avoid spam

            # --- Calculate Advantages (GAE) ---
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # --- Update Policy (PPO Epochs) ---

            # Flatten the batch
            b_obs = obs.reshape((-1,) + envs._single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs._single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            pbar_update = tqdm(range(args.update_epochs), desc="Updating Policy")
            # Display epoch progress on TQDM
            for epoch in pbar_update:
                # Use a random selection of data to optimize on
                b_inds = np.random.permutation(args.batch_size)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    # Evaluate the moves made by the 'old' network against the 'new' one
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )

                    # Calculate the ratio between the old and new policy (The proximal part)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss (Actor)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (Critic)
                    newvalue = newvalue.view(-1)
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    # Entropy loss (Exploration)
                    entropy_loss = entropy.mean()

                    # Total loss
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            # --- Log Metrics ---
            sps = int(global_step / (time.time() - start_time))
            print(f"  SPS: {sps}")
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("charts/SPS", sps, global_step)

            if update % args.save_interval == 0:
                checkpoint_path = f"{run_dir}/checkpoint_update_{update}.pth"
                # Ensure the directory exists (SummaryWriter usually makes it, but good to be safe)
                os.makedirs(run_dir, exist_ok=True)

                torch.save({
                    'update': update,
                    'global_step': global_step,
                    'agent_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nTraining interrupted: {e}")
    finally:
        # --- Clean Up ---
        print("Training finished. Closing environment.")
        envs.close()
        writer.close()
        print("Done.")