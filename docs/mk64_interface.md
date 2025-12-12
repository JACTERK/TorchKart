# Mario Kart 64 BizHawk LUA Interface

## Requirements

- [LUA Sockets](https://lunarmodules.github.io/luasocket/)

## Configuration

At the top of `mk64_interface.lua` is a configuration block. This contains information regarding:

- Host IP
- Port
- Savestate Path
- Memory Domain

The host IP if running the emulators on the same system as the control server will be localhost, and the port is the start of the range of ports the BizHawk clients will use. (Ex. If 20 clients are connecting, they will use ports 65432 - 65451)

The savestate path is an absolute path to the savestate you create in the setup instructions in the `README.md` file. 

The memory domain (used by BizHawk to read memory from the emulator) for N64 emulation is `RDRAM`. 

## Architecture

### `MEMORY_MAP`

This is a dictionary of key-value pairs, where the key is the memory address to read, and the value is the number of bytes to read from each memory address. 

As per convention, the each dictionary entry is labeled with the name and data type. 


### `execute_action`

This function defines the discrete action space. 

It starts out with two blocks of code resetting digital and analogue controls from prior movement. 

```
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

After which it defines the set of discrete actions the agent can take. If modifying the number of actions, ensure to also update this line in `torchkart.py`, which keeps track of the range of actions the agent can send to the client:

```
single_action_space = Discrete(9)
```

By default the agent can choose one of nine discrete actions (numbered 0-8). 

At the end of the function we call `joypad.set()` and `joypad.setanalog()`, each with the control dictionary, and an integer representing the controller port (default=1).


### `get_state_bytes`

A helper function that collects the state values from memory and returns them as a single byte string. 


### Initialization

When started, the script will wait for a connection to the control server, and will retry a connection every 10 seconds. 


### Connected

Once connected, the client waits for the server to send a command. These commands can be one of the following:

- `R`: Reset the client to the savestate
- `S`: Step the emulator forward with a given action, and send back the state to the control server
- `C`: Close the connection with the control server. 

The emulators will freeze until the control server successfully sends a command to each emulator. Under normal operation, they all wait for a successful step, and send the `emu.frameadvance()` command to the emulator to advance the game one frame. 
