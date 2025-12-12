# Hyperparameters

## Setting Hyperparameters

When calling `torchkart.py`, you can set these values as such:

```
python torchkart.py --num-envs=20 --total-timesteps=250000000 ...
```


## PPO Core Hyperparameters

| Parameter       | Default | What it does                                                                             | Impact on Agent                                                                                                    |
|-----------------|---------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `learning-rate` | `2.5e4` | The step size for the Adam optimizer                                                     | Agent changes behaviour fast but might have difficulty stabilizing (shows in jittery steering)                     |
| `gamma`         | `0.99`  | Discount factor: How muich the agent cares about immediate vs future rewards             | At the default 0.99, the agent heavily cares about long term rewards (finishing lap)                               |
| `gae-lambda`    | `0.95`  | Genralized Advantage Estimation: Balances bias vs variance when calculating advantage    | Lowering it reduces variance but introduces bias. Leave at 0.95.                                                   |
| `clip-coef`     | `0.2`   | Surrogate clipping: Prevents the new policy from being too different from the old one.   | Prevents 'catastrophic forgetting', where the policy collapses if a fatal error is made by one of the agents.      |
| `ent-coef`      | `0.01`  | Entropy Coefficient: How much to reward 'randomness' or exploration                      | This percentage of actions the kart  drives erratically to try new things.                                         |
| `vf-coef`       | `0.5`   | Value Function Coefficient: The weight of the critic's loss compared to the actor's loss | Balances how much the network focuses on predicting the score vs choosing the action.                              |
| `max-grad-norm` | `0.5`   | Gradient Clipping: Caps the gradient size during backpropogation                         | Stability safety net. Prevents the 'exploding gradient' problem where a single bad update breaks the whole network |


## Rollout and Training Loop

| Parameter         | Default | What it does                                                            | Impact on Training                                                                                     |
|-------------------|---------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `num-envs`        | `4`     | Number of BizHawk instances running in parallel                         | More environments = more diverse data per second (Running on a 9800X3d I can manage ~20 clients)       |
| `num-steps`       | `2048`  | How many frames to collect _per environment_ before updating the policy | Total batch size = `num-envs` * `num-steps`                                                            |
| `num-minibatches` | `32`    | Splits the total batch size into smaller chunks for the GPU             | 8192 / 32 = 256 samples at a time to save memory                                                       |
| `update-epochs`   | `10`    | How many times to re-use the collected batch for training               | The agent will look at the same frames 10 times to get as much learning out of them before discarding. |
| `total-timesteps` | `5M`    | Total duration of the experiment                                        | The loop will stop after 5M steps have been processed. For best results increase this to 100M+.        |


## System and Environment

| Parameter             | Default     | What it does                                                                                          |
|-----------------------|-------------|-------------------------------------------------------------------------------------------------------|
| `host`                | `127.0.0.1` | The IP address the python server binds to (localhost)                                                 |
| `port`                | `65432`     | The TCP port used for the socket connection                                                           |
| `cuda`                | `True`      | Whether to use NVIDIA GPU if True (`torch.device("cuda"))`, or CPU if False                           |
| `torch-deterministic` | `True`      | Forces PyTorch to use deterministic algorithms (makes the result reproducible if using the same seed) |
| `save-interval`       | `10`        | How often (in updates) to save a `.pth` model file to disk.                                           |

