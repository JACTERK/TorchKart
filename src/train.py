import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.args import parse_args
from src.environment import MK64Env
from src.agent import ActorCritic
from src.emulator_manager import EmulatorManager


def main():
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

    # --- Setup Emulators ---
    print("Preparing emulator manager...")
    emulator_manager = EmulatorManager(
        num_envs=args.num_envs,
        bizhawk_exe=args.bizhawk_exe,
        rom_path=args.rom_path,
        lua_script=args.lua_script,
        grid_cols=args.grid_cols,
        grid_fraction=args.grid_fraction,
    )

    # --- Setup Environment ---
    # The environment will launch the emulators after it starts listening
    envs = MK64Env(
        num_envs=args.num_envs, 
        host=args.host, 
        port=args.port, 
        frame_skip=args.frame_skip,
        emulator_manager=emulator_manager
    )

    # --- Setup Agent ---
    agent = ActorCritic(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # --- Setup Storage ---
    # This buffer will store the rollouts
    obs = torch.zeros((args.num_steps, args.num_envs) + envs._single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, len(envs.ACTION_DIMS))).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Counters for per-update aggregate metrics
    head_names = ["throttle", "steering", "drift", "item"]

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

            # Linear LR annealing: decay from initial LR to 0 over training
            frac = 1.0 - (update - 1) / num_updates
            lr_now = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

            # --- Collect Rollouts ---
            pbar = tqdm(range(0, args.num_steps), desc="Collecting Rollout")
            for step in pbar:
                global_step += 1 * args.num_envs

                obs[step] = next_obs
                dones[step] = next_done

                # Get action from the agent
                with torch.no_grad():
                    action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
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

                        # Game performance metrics
                        writer.add_scalar("charts/avg_speed", ep_info.get("avg_speed", 0), global_step)
                        writer.add_scalar("charts/wall_hits_per_episode", ep_info.get("wall_hits", 0), global_step)
                        writer.add_scalar("charts/avg_progress_per_step", ep_info.get("avg_progress_per_step", 0), global_step)
                        writer.add_scalar("charts/stuck_termination", float(ep_info.get("stuck_termination", False)), global_step)

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
            b_actions = actions.reshape((-1, len(envs.ACTION_DIMS)))
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
                    _, newlogprob, entropy, newvalue, head_ents = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )

                    # Calculate the ratio between the old and new policy (The proximal part)
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                        # Track clip fraction
                        clipfracs = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

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

                    # Per-head entropy (for logging, taken from last minibatch)
                    last_head_entropies = head_ents.mean(dim=0)

                    # Total loss
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    # Capture grad norm before clipping
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            # --- Log Metrics ---
            sps = int(global_step / (time.time() - start_time))
            print(f"  SPS: {sps}")

            # Existing metrics
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)

            # Loss metrics
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

            # Policy health metrics
            writer.add_scalar("losses/clipfrac", clipfracs.item(), global_step)
            writer.add_scalar("charts/grad_norm", grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, global_step)

            # Explained variance: how well the critic predicts returns
            # 1.0 = perfect prediction, 0 = no better than mean, negative = worse than mean
            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            # Per-head entropy
            for h, name in enumerate(head_names):
                writer.add_scalar(f"entropy/{name}", last_head_entropies[h].item(), global_step)

            # --- LSTM metrics (uncomment when LSTM is implemented) ---
            # writer.add_scalar("charts/hidden_state_norm", hidden_state.norm().item(), global_step)
            # writer.add_histogram("charts/episode_length_histogram", episode_lengths_buffer, global_step)

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
        # --- Clean Up ---
        print("Training finished. Closing environment.")
        envs.close()
        emulator_manager.shutdown()
        writer.close()
        print("Done.")
