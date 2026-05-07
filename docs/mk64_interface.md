# Mario Kart 64 BizHawk Lua Interface

## Requirements

- [Lua Sockets](https://lunarmodules.github.io/luasocket/) (included in `../Lua`)

## Configuration

At the top of `mk64_interface.lua` is a configuration block. This contains information regarding:

- Host IP
- Port
- Savestate Path
- Memory Domain

The host IP if running the emulators on the same system as the control server will be localhost, and the port is the start of the range of ports the BizHawk clients will use. (Ex. If 20 clients are connecting, they will use ports 54321 - 54340)

The savestate path is an absolute path to the savestate you create in the setup instructions in the `README.md` file.

The memory domain (used by BizHawk to read memory from the emulator) for N64 emulation is `RDRAM`.

## Architecture

### `MEMORY_MAP`

This is a dictionary of key-value pairs, where the key is the memory address to read, and the value is the number of bytes to read from each memory address.

Each dictionary entry is labeled with the name and data type.

The current memory map reads 50 bytes across 13 addresses:

| Address      | Bytes | Name                  | Type              | Description                                             |
|--------------|-------|---------------------- |-------------------|---------------------------------------------------------|
| `0x0F69C4`   | 4     | x velocity            | float             | Kart's X-axis velocity                                  |
| `0x0F69CC`   | 4     | y velocity            | float             | Kart's Y-axis velocity                                  |
| `0x0F69C8`   | 4     | z velocity            | float             | Kart's Z-axis velocity                                  |
| `0x163288`   | 4     | path progress         | signed int        | Progress along the track (0 to ~1000)                   |
| `0x164390`   | 4     | lap                   | signed int        | Current lap (-1 = behind start line, 3 = finished)      |
| `0x0F69BC`   | 2     | orientation           | unsigned short    | Heading angle (0 to 65535)                              |
| `0x0F6BC4`   | 4     | wall_1                | fixed-point 16.16 | Wall proximity sensor 1                                 |
| `0x0F6BE4`   | 4     | wall_2                | fixed-point 16.16 | Wall proximity sensor 2                                 |
| `0x163068`   | 4     | track center distance | float             | Distance from track center (-1 to 1 is on-road)         |
| `0x18CFE4`   | 4     | speed                 | float             | Current speed                                           |
| `0x0F69A0`   | 4     | mushroom count        | fixed-point 16.16 | Mushroom count (14=3, 13=2, 12=1, 0=0)                  |
| `0x0E9E74`   | 4     | drift state           | unsigned int      | 0=not drifting, 2=drifting on road, 4=drifting off-road |
| `0x0F6A4C`   | 4     | mushroom boost        | unsigned int      | 0=inactive, 8192=active (resets on expiry or wall hit)  |


### `STEERING_VALUES`

A lookup table mapping steering indices to analog stick X-axis values:

| Index | Value | Direction    |
|-------|-------|--------------|
| 0     | -80   | Hard Left    |
| 1     | -40   | Slight Left  |
| 2     | 0     | Center       |
| 3     | 40    | Slight Right |
| 4     | 80    | Hard Right   |


### `execute_multi_action`

This function defines the multi-discrete action space. It receives four separate action parameters and maps them to BizHawk controller inputs:

| Parameter  | Values                        | Mapping                            |
|------------|-------------------------------|------------------------------------|
| `throttle` | 0=nothing, 1=forward, 2=brake | A button, B button, Y-axis         |
| `steering` | 0-4 (hard left to hard right) | X-axis via `STEERING_VALUES` table |
| `drift`    | 0=no drift, 1=drift           | R button                           |
| `item`     | 0=no item, 1=use item         | Z button                           |

The function starts by resetting all digital and analog controls to their defaults:

```lua
local digital_controls = {
    A = false,
    B = false,
    R = false,
    Z = false
}

local analog_controls = {
    ["X Axis"] = 0.0,
    ["Y Axis"] = 0.0
}
```

Then applies the appropriate inputs based on the four action parameters. At the end, it calls `joypad.set()` and `joypad.setanalog()` with the control dictionaries, targeting controller port 1.

The corresponding Python-side action space is defined in `environment.py` as:

```python
ACTION_DIMS = [3, 5, 2, 2]  # [throttle, steering, drift, item]
```


### `get_state_bytes`

A helper function that collects the state values from memory (via `MEMORY_MAP`) and returns them as a single byte string.


### Initialization

When started, the script will wait for a connection to the control server, retrying every 2 seconds.


### Connected

Once connected, the client waits for the server to send a command. These commands can be one of the following:

- `R`: Reset the client to the savestate
- `S`: Step the emulator forward with a given action, and send back the state to the control server
- `D`: Demo mode — advance one frame with human input, then send back the state plus the observed action (used for imitation learning recording)
- `C`: Close the connection with the control server

For the `S` command, the client receives 5 bytes: 1 byte for `frame_skip` followed by 4 bytes for the multi-discrete action (`throttle`, `steering`, `drift`, `item`). The action is executed for `frame_skip` consecutive frames before reading and returning the state.

Frame skip is used to speed up training time by repeating the same action for `n` number of frames, combining the feature data over those frames, and returning them to the server. 

For the `D` command, the script reads the human's joypad input after frame advance, maps it back to the multi-discrete action encoding, and sends 54 bytes total (50 state bytes + 4 action bytes).

The emulators will freeze until the control server successfully sends a command to each emulator. Under normal operation, they all wait for a successful step, and send the `emu.frameadvance()` command to the emulator to advance the game one frame.
