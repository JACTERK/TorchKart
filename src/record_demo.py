"""
Records a human demonstration for behavioral cloning.

Usage:
    python -m src.record_demo --output demos/my_demo.npz

Play 3 laps normally with a controller in BizHawk. The script records
(observation, action) pairs and saves them when the race finishes or
you press Ctrl+C.
"""
import argparse
import os
import socket
import struct
import time

import numpy as np

from src.environment import MK64Env


def parse_demo_args():
    parser = argparse.ArgumentParser(description="Record a human demonstration for MK64")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=54321)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz file path. Defaults to demos/<timestamp>.npz")
    return parser.parse_args()


def main():
    args = parse_demo_args()

    if args.output is None:
        os.makedirs("demos", exist_ok=True)
        args.output = f"demos/demo_{int(time.time())}.npz"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # We reuse MK64Env's parsing constants
    state_size = MK64Env.STATE_SIZE_BYTES  # 50 bytes
    state_format = MK64Env.STATE_FORMAT
    state_names = MK64Env.STATE_NAMES
    num_features = MK64Env.NUM_OBS_FEATURES  # 14
    action_size = 4  # 4 bytes for multi-discrete action

    # Setup server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    server.settimeout(300)

    print(f"Demo recorder listening on {args.host}:{args.port}")
    print("Launch BizHawk with the Lua script and connect...")

    conn, addr = server.accept()
    conn.settimeout(300)
    print(f"Client connected from {addr}")

    # Send reset command to load save state
    conn.sendall(b'R')
    init_state = conn.recv(state_size)
    print(f"Save state loaded ({len(init_state)} bytes). Start playing!")
    print("Press Ctrl+C to stop and save the recording.\n")

    # Storage for recorded data
    observations = []
    actions = []

    # Tracking for derived features (mimics environment logic)
    drift_counter = 0.0
    last_angle = 0.0

    frame_count = 0

    try:
        while True:
            # Send demo command
            conn.sendall(b'D')

            # Receive state bytes + 4 action bytes
            data = conn.recv(state_size + action_size)
            if len(data) < state_size + action_size:
                print(f"Incomplete data ({len(data)} bytes). Stopping.")
                break

            state_bytes = data[:state_size]
            action_bytes = data[state_size:]

            # Parse state
            try:
                raw_values = struct.unpack(state_format, state_bytes)
            except struct.error as e:
                print(f"Struct error: {e}. Stopping.")
                break

            state = dict(zip(state_names, raw_values))

            # Normalize features (same logic as environment._parse_and_preprocess)
            norm_x_vel = state["x_vel"] / 5.0
            norm_y_vel = state["y_vel"] / 5.0
            norm_speed = state["speed"] / 80.0

            angle_radians = (state["orientation"] / 65536.0) * 2 * np.pi
            norm_sin_angle = np.sin(angle_radians)
            norm_cos_angle = np.cos(angle_radians)

            norm_center = state["track_center_dist"]
            norm_progress = state["path_progress"] / 1900.0
            norm_lap = state["lap"] / 3.0

            # Mushroom count
            mushroom_raw = state["mushroom_raw"] >> 16
            if mushroom_raw >= 14:
                mushroom_count = 3
            elif mushroom_raw >= 13:
                mushroom_count = 2
            elif mushroom_raw >= 12:
                mushroom_count = 1
            else:
                mushroom_count = 0
            norm_mushrooms = mushroom_count / 3.0

            # Drift state
            drift_active = 1.0 if state["drift_state"] != 0 else 0.0
            if drift_active > 0.5:
                drift_counter += 1.0
            else:
                drift_counter = 0.0
            drift_duration = min(drift_counter / 180.0, 1.0)

            # Mushroom boost
            mushroom_boost_active = 1.0 if state["mushroom_boost_raw"] == 8192 else 0.0

            # Angular velocity
            angular_diff = angle_radians - last_angle
            if angular_diff > np.pi:
                angular_diff -= 2 * np.pi
            elif angular_diff < -np.pi:
                angular_diff += 2 * np.pi
            last_angle = angle_radians
            angular_velocity = np.clip(angular_diff / 0.5, -1.0, 1.0)

            # Build observation (14 features)
            obs = np.array([
                norm_x_vel, norm_y_vel, norm_speed,
                norm_sin_angle, norm_cos_angle,
                norm_center, 0.0,  # wall_hit not tracked in demo mode
                norm_progress, norm_lap, norm_mushrooms,
                drift_active, drift_duration,
                mushroom_boost_active, angular_velocity
            ], dtype=np.float32)

            # Parse action bytes
            action = np.array([
                action_bytes[0],  # throttle
                action_bytes[1],  # steering
                action_bytes[2],  # drift
                action_bytes[3],  # item
            ], dtype=np.int32)

            observations.append(obs)
            actions.append(action)
            frame_count += 1

            # Print progress
            if frame_count % 60 == 0:
                lap = state["lap"]
                progress = state["path_progress"]
                speed = state["speed"]
                print(f"  Frame {frame_count}: Lap {lap}, Progress {progress:.0f}, Speed {speed:.1f}")

            # Check if race finished
            if state["lap"] >= 3:
                print(f"\nRace finished! Recorded {frame_count} frames.")
                break

    except KeyboardInterrupt:
        print(f"\nRecording stopped by user. Recorded {frame_count} frames.")
    finally:
        # Save recording
        if frame_count > 0:
            obs_array = np.stack(observations)
            act_array = np.stack(actions)
            np.savez(args.output, observations=obs_array, actions=act_array)
            print(f"Saved demo to {args.output}")
            print(f"  Observations shape: {obs_array.shape}")
            print(f"  Actions shape: {act_array.shape}")
        else:
            print("No frames recorded. Nothing saved.")

        # Cleanup
        try:
            conn.sendall(b'C')
            conn.close()
        except:
            pass
        server.close()


if __name__ == "__main__":
    main()
