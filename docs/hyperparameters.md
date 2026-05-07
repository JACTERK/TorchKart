# Hyperparameters

## Setting Hyperparameters

When calling `main.py`, you can set these values as such:

```
python main.py --num-envs=20 --total-timesteps=250000000 ...
```


## PPO Core Hyperparameters

| Parameter       | Default | What it does                                                                             | Impact on Agent                                                                                                    |
|-----------------|---------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `learning-rate` | `2.5e4` | The step size for the Adam optimizer                                                     | Agent changes behaviour fast but might have difficulty stabilizing (shows in jittery steering)                     |
| `gamma`         | `0.99`  | Discount factor: How much the agent cares about immediate vs future rewards              | At the default 0.99, the agent heavily cares about long term rewards (finishing lap)                               |
| `gae-lambda`    | `0.95`  | Generalized Advantage Estimation: Balances bias vs variance when calculating advantage   | Lowering it reduces variance but introduces bias. Leave at 0.95.                                                   |
| `clip-coef`     | `0.2`   | Surrogate clipping: Prevents the new policy from being too different from the old one.   | Prevents 'catastrophic forgetting', where the policy collapses if a fatal error is made by one of the agents.      |
| `ent-coef`      | `0.01`  | Entropy Coefficient: How much to reward 'randomness' or exploration                      | This percentage of actions the kart drives erratically to try new things.                                          |
| `vf-coef`       | `0.5`   | Value Function Coefficient: The weight of the critic's loss compared to the actor's loss | Balances how much the network focuses on predicting the score vs choosing the action.                              |
| `max-grad-norm` | `0.5`   | Gradient Clipping: Caps the gradient size during backpropagation                         | Stability safety net. Prevents the 'exploding gradient' problem where a single bad update breaks the whole network |


## Rollout and Training Loop

| Parameter         | Default | What it does                                                            | Impact on Training                                                                                      |
|-------------------|---------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `num-envs`        | `4`     | Number of BizHawk instances running in parallel                         | More environments = more diverse data per second (Running on a 9800X3d I can manage ~20 clients)        |
| `num-steps`       | `2048`  | How many frames to collect _per environment_ before updating the policy | Total batch size = `num-envs` * `num-steps`                                                             |
| `num-minibatches` | `128`   | Splits the total batch size into smaller chunks for the GPU             | 8192 / 128 = 64 samples at a time to save memory                                                        |
| `update-epochs`   | `10`    | How many times to re-use the collected batch for training               | The agent will look at the same frames 10 times to get as much learning out of them before discarding.  |
| `total-timesteps` | `5M`    | Total duration of the experiment                                        | The loop will stop after 5M steps have been processed. For best results increase this to 100M+.         |
| `frame-skip`      | `4`     | How many emulator frames to repeat each action                          | Reduces network traffic and speeds up training. Each step advances the emulator by `frame-skip` frames. |
| `save-interval`   | `10`    | How often (in updates) to save a `.pth` model file to disk              | Checkpoints are saved to `runs/<experiment>/checkpoint_update_N.pth`                                    |
| `load-checkpoint` | `None`  | Path to a `.pth` checkpoint file to resume training from                | Restores agent weights, optimizer state, update counter, and global step                                |


## Architecture

| Parameter         | Default | What it does                                                                | Impact on Training                                                                                         |
|-------------------|---------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `use-transformer` | `False` | Switch from MLP to Transformer policy architecture                          | Transformer uses causal attention over a sequence of frames instead of frame stacking                      |
| `seq-length`      | `1024`  | Context window length for the Transformer (number of frames to attend over) | Only used when `--use-transformer` is set. Also determines the observation stack size for the Transformer. |


## System and Environment

| Parameter             | Default                 | What it does                                                                                          |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------------|
| `host`                | `127.0.0.1`             | The IP address the Python server binds to (localhost)                                                 |
| `port`                | `54321`                 | The TCP port used for the socket connection                                                           |
| `cuda`                | `True`                  | Whether to use NVIDIA GPU if True (`torch.device("cuda")`), or CPU if False                           |
| `torch-deterministic` | `True`                  | Forces PyTorch to use deterministic algorithms (makes the result reproducible if using the same seed) |
| `rom-path`            | `rom/marioKart.n64`     | Path to the Mario Kart 64 ROM file                                                                    |
| `bizhawk-exe`         | `./bizhawk/EmuHawk.exe` | Path to the BizHawk EmuHawk executable                                                                |
| `lua-script`          | `mk64_interface.lua`    | Path to the Lua interface script                                                                      |
| `grid-cols`           | `5`                     | Number of columns for the emulator window grid layout                                                 |
| `grid-fraction`       | `0.33`                  | Fraction of screen width to use for the emulator grid (0.0-1.0)                                       |
