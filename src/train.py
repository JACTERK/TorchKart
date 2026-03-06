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
