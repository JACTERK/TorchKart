"""
Standalone emulator launcher for TorchKart.
Run this script to start all BizHawk emulators, then type 'exit' to close them.

Usage:
    python launch_emulators.py --rom-path="path/to/marioKart.n64" --num-envs=20
"""
import argparse
from src.emulator_manager import EmulatorManager


def parse_launcher_args():
    parser = argparse.ArgumentParser(description="Launch and manage BizHawk emulators for TorchKart")

    parser.add_argument("--num-envs", type=int, default=4,
                        help="Number of BizHawk emulator instances to launch.")
    parser.add_argument("--rom-path", type=str, required=True,
                        help="Path to the .n64 ROM file.")
    parser.add_argument("--bizhawk-exe", type=str, default="./bizhawk/EmuHawk.exe",
                        help="Path to the BizHawk EmuHawk executable.")
    parser.add_argument("--lua-script", type=str, default="mk64_interface.lua",
                        help="Path to the Lua interface script.")
    parser.add_argument("--grid-cols", type=int, default=5,
                        help="Number of columns for the emulator window grid layout.")
    parser.add_argument("--grid-fraction", type=float, default=0.33,
                        help="Fraction of screen width to use for the emulator grid (0.0-1.0).")

    return parser.parse_args()


def main():
    args = parse_launcher_args()

    manager = EmulatorManager(
        num_envs=args.num_envs,
        bizhawk_exe=args.bizhawk_exe,
        rom_path=args.rom_path,
        lua_script=args.lua_script,
        grid_cols=args.grid_cols,
        grid_fraction=args.grid_fraction,
    )
    manager.launch()

    print("\n" + "=" * 50)
    print("All emulators are running.")
    print("Type 'exit' and press Enter to close them all.")
    print("=" * 50 + "\n")

    try:
        while True:
            user_input = input("> ").strip().lower()
            if user_input == "exit":
                break
    except (KeyboardInterrupt, EOFError):
        print()  # Newline after ^C

    manager.shutdown()


if __name__ == "__main__":
    main()
