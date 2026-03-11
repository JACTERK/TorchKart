import socket
import struct
from typing import Tuple, Dict, Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, MultiDiscrete


class MK64Env(gym.vector.VectorEnv):
    """
    A Gymnasium Environment that acts as a server
    and manages multiple BizHawk socket clients.
    """
    # Total bytes to read, based on the Lua script's MEMORY_MAP
    # 5 * 4 (float) + 5 * 4 (int) + 1 * 2 (short uint) = 42
    STATE_SIZE_BYTES = 42

    # The struct format string for parsing the 42 bytes.
    # '>' means big-endian (which N64 is)
    # f = 4-byte float
    # i = 4-byte signed int
    # H = 2-byte unsigned short
    # Fixed-Point 16.16 is read as 'i' (signed int)
    STATE_FORMAT = ">fffiiHiiffi"

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
        "speed",
        "mushroom_raw"
    ]

    # The number of processed features sent to the network (some are combined)
    NUM_OBS_FEATURES = 10

    # Action dimensions: [throttle, steering, drift, item]
    ACTION_DIMS = [3, 5, 2, 2]
    NUM_ACTION_BYTES = len(ACTION_DIMS)  # 4 bytes sent to Lua

    def __init__(self, num_envs, host, port, frame_skip=4, stack_size=8):
        # Define how many frames to "stack" for temporal awareness
        self.stack_size = stack_size
        self.frame_skip = frame_skip

        # Observation dimensionality (Number of features * the stack size)
        obs_dim = self.NUM_OBS_FEATURES * self.stack_size

        # Define the observation and action spaces for Gymnasium
        single_observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Multi-discrete action space: [throttle(3), steering(5), drift(2), item(2)]
        single_action_space = MultiDiscrete(self.ACTION_DIMS)

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

        # Per-episode metric trackers
        self.episode_speed_sum = np.zeros(num_envs, dtype=np.float32)
        self.episode_wall_hits = np.zeros(num_envs, dtype=np.int32)
        self.episode_progress_start = np.zeros(num_envs, dtype=np.float32)


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

        # --- Mushroom Count ---
        # Fixed-point 16.16: raw value 14=3, 13=2, 12=1, 0=0
        mushroom_raw = state["mushroom_raw"] >> 16  # Get integer part of fixed-point
        if mushroom_raw >= 14:
            mushroom_count = 3
        elif mushroom_raw >= 13:
            mushroom_count = 2
        elif mushroom_raw >= 12:
            mushroom_count = 1
        else:
            mushroom_count = 0
        norm_mushrooms = mushroom_count / 3.0  # Normalize to 0.0 - 1.0

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
            norm_lap,
            norm_mushrooms
        ], dtype=np.float32)

        # Return processed state dict for reward calculation later
        processed_state_dict = {
            "progress": state["path_progress"],
            "lap": state["lap"],
            "speed": state["speed"],
            "track_center_dist": state["track_center_dist"],
            "wall_hit": wall_hit,  # Boolean 0.0 or 1.0
            "raw_wall_1": state["wall_1"],
            "raw_wall_2": state["wall_2"],
            "mushrooms": mushroom_count
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
            self.episode_speed_sum[index] = 0.0
            self.episode_wall_hits[index] = 0
            self.episode_progress_start[index] = state_dict.get("progress", 0.0)

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
            for i in range(self.num_envs):
                # Send 'S' (Step) command + 1-byte frame_skip + 4-byte multi-action
                action_bytes = bytes(s_actions[i].astype(np.uint8).tolist())
                cmd = b'S' + bytes([self.frame_skip]) + action_bytes
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
                self.episode_speed_sum[i] += new_state.get("speed", 0.0)
                if new_state.get("wall_hit", 0.0) > 0.5:
                    self.episode_wall_hits[i] += 1

                # Handle "done" state (terminated or truncated)
                info = {}
                if terminated or truncated:
                    ep_len = self.episode_lengths[i]

                    info["final_info"] = {
                        "episode": {
                            "r": self.episode_rewards[i],
                            "l": ep_len,
                            "final_lap": new_state.get("lap", -99),
                            "avg_speed": self.episode_speed_sum[i] / max(ep_len, 1),
                            "wall_hits": int(self.episode_wall_hits[i]),
                            "avg_progress_per_step": (new_state.get("progress", 0) - self.episode_progress_start[i]) / max(ep_len, 1),
                            "stuck_termination": not is_finished_race and terminated,
                        }
                    }
                    if is_finished_race:
                        # Account for frame_skip: each step = frame_skip game frames
                        race_time_seconds = (ep_len * self.frame_skip) / 60.0
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
