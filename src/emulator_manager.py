import subprocess
import time
import math
import ctypes
import ctypes.wintypes
import os

# Win32 constants
SW_SHOWNOACTIVATE = 4
WNDENUMPROC = ctypes.WINFUNCTYPE(
    ctypes.wintypes.BOOL,
    ctypes.wintypes.HWND,
    ctypes.wintypes.LPARAM,
)

user32 = ctypes.windll.user32


def _get_screen_size():
    """Returns (width, height) of the primary monitor."""
    SM_CXSCREEN = 0
    SM_CYSCREEN = 1
    return user32.GetSystemMetrics(SM_CXSCREEN), user32.GetSystemMetrics(SM_CYSCREEN)


def _get_window_title(hwnd):
    """Get the title text of a window handle."""
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _find_window_by_pid(pid):
    """Find the main emulator window handle for a given process ID (ignores Lua Console)."""
    result = []

    def enum_callback(hwnd, _lparam):
        # Only consider visible windows
        if not user32.IsWindowVisible(hwnd):
            return True
        # Check if this window belongs to our PID
        window_pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid:
            # Skip the Lua Console window
            title = _get_window_title(hwnd)
            if "Lua Console" not in title:
                result.append(hwnd)
        return True

    user32.EnumWindows(WNDENUMPROC(enum_callback), 0)

    # Return the first match (main window), or None
    return result[0] if result else None


class EmulatorManager:
    """
    Manages BizHawk emulator processes: launching, arranging windows
    in a grid, and shutting them down cleanly.
    """

    def __init__(self, num_envs, bizhawk_exe, rom_path, lua_script, grid_cols=5, grid_fraction=0.5):
        self.num_envs = num_envs
        self.bizhawk_exe = os.path.abspath(bizhawk_exe)
        self.rom_path = os.path.abspath(rom_path)
        self.lua_script = os.path.abspath(lua_script)
        self.grid_cols = grid_cols
        self.grid_fraction = grid_fraction
        self.processes = []

    def launch(self):
        """Spawn all BizHawk emulator instances and arrange their windows."""
        print(f"Launching {self.num_envs} BizHawk instance(s)...")

        for i in range(self.num_envs):
            proc = subprocess.Popen(
                [self.bizhawk_exe, self.rom_path, f"--lua={self.lua_script}"],
                # Prevent child stdin/stdout from interfering with the training script
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.processes.append(proc)
            print(f"  Instance {i + 1}/{self.num_envs} launched (PID {proc.pid})")

        print("All instances launched. Waiting for windows to appear...")
        self.arrange_windows()

    def arrange_windows(self, timeout=60, poll_interval=2):
        """
        Tile all emulator windows into a grid on the primary monitor.
        Polls until all windows are found or the timeout is reached.
        """
        cols = min(self.grid_cols, self.num_envs)
        rows = math.ceil(self.num_envs / cols)

        # Use only a fraction of the screen width for the grid
        screen_w, screen_h = _get_screen_size()
        usable_w = int(screen_w * self.grid_fraction)
        win_w = usable_w // cols
        win_h = screen_h // rows

        # Track which windows we've already positioned
        arranged_set = set()
        deadline = time.time() + timeout

        while len(arranged_set) < self.num_envs and time.time() < deadline:
            for i, proc in enumerate(self.processes):
                if i in arranged_set:
                    continue

                # Check if the process exited
                if proc.poll() is not None:
                    print(f"  Warning: Instance {i + 1} (PID {proc.pid}) exited early, skipping.")
                    arranged_set.add(i)  # Mark as handled so we don't block on it
                    continue

                hwnd = _find_window_by_pid(proc.pid)
                if hwnd is None:
                    continue  # Not ready yet, will retry next poll

                col = i % cols
                row = i // cols
                x = col * win_w
                y = row * win_h

                # MoveWindow(hwnd, x, y, width, height, repaint)
                user32.MoveWindow(hwnd, x, y, win_w, win_h, True)
                arranged_set.add(i)
                print(f"  Arranged instance {i + 1}/{self.num_envs}")

            if len(arranged_set) < self.num_envs:
                remaining = self.num_envs - len(arranged_set)
                print(f"  Waiting for {remaining} window(s) to appear... ({int(deadline - time.time())}s remaining)")
                time.sleep(poll_interval)

        successful = sum(1 for i in arranged_set if self.processes[i].poll() is None or i in arranged_set)
        print(f"Arranged {len(arranged_set)}/{self.num_envs} window(s) in a {cols}x{rows} grid.")

    def shutdown(self):
        """Terminate all emulator processes."""
        print("Shutting down emulator instances...")
        for i, proc in enumerate(self.processes):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                print(f"  Instance {i + 1} (PID {proc.pid}) terminated.")
            else:
                print(f"  Instance {i + 1} (PID {proc.pid}) already exited.")
        self.processes.clear()
        print("All emulator instances shut down.")

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
