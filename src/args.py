import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Trainer for Mario Kart 64")

    # Environment Args
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of parallel BizHawk clients to connect to.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host IP address to bind the server to.")
    parser.add_argument("--port", type=int, default=54321,
                        help="Port to listen on.")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Number of frames to repeat each action (reduces network traffic).")
    parser.add_argument("--rom-path", type=str, default="rom/marioKart.n64",
                        help="Path to the .n64 ROM file.")
    parser.add_argument("--bizhawk-exe", type=str, default="./bizhawk/EmuHawk.exe",
                        help="Path to the BizHawk EmuHawk executable.")
    parser.add_argument("--lua-script", type=str, default="mk64_interface.lua",
                        help="Path to the Lua interface script.")
    parser.add_argument("--grid-cols", type=int, default=5,
                        help="Number of columns for the emulator window grid layout.")
    parser.add_argument("--grid-fraction", type=float, default=0.33,
                        help="Fraction of screen width to use for the emulator grid (0.0-1.0).")

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
