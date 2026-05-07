-- bizhawk/mk64_interface.lua

-- Load socket library
package.cpath = package.cpath .. ";?.dll"
local socket = require("socket")

-- --- Configuration ---
local HOST = "127.0.0.1"
local PORT = 54321
local SAVESTATE_PATH = "C:/Users/jacobterkuc/Desktop/TorchKart/mk64_start.state"
local MEMORY_DOMAIN = "RDRAM"

-- {address (hex), num_bytes}
local MEMORY_MAP = {
    {0x0F69C4, 4}, -- x velocity -- float
    {0x0F69CC, 4}, -- y velocity -- float
    {0x0F69C8, 4}, -- z velocity -- float
    {0x163288, 4}, -- path progress (Starts at 0, will go up to around 1000) -- sint
    {0x164390, 4}, -- lap (-1 if behind the start line, 3 is finished.) -- sint
    {0x0F69BC, 2}, -- orientation (value between 0 and 65535) -- uint
    {0x0F6BC4, 4}, -- wall_1 -- Fixed-Point 16.16 (Changes by very small number)
    {0x0F6BE4, 4}, -- wall_2 -- Fixed-Point 16.16 (Changes by very small number)
    {0x163068, 4}, -- Distance from the center of the track (0 is centered, negative is how far left, positive is how far right, road is between -1 and 1) -- Float
    {0x18CFE4, 4}, -- speed -- float
    {0x0F69A0, 4}, -- mushroom count -- Fixed-Point 16.16 (14=3, 13=2, 12=1, 0=0)
    {0x0E9E74, 4}, -- drift state -- uint32 (0=not drifting, 2=drifting on road, 4=drifting off-road)
    {0x0F6A4C, 4}, -- mushroom boost status -- uint32 (0=inactive, 8192=active; resets on expiry or wall hit)
}

-- Steering analog values for each steering index (0-4)
local STEERING_VALUES = {
    [0] = -80,  -- Hard Left
    [1] = -40,  -- Slight Left
    [2] = 0,    -- Center
    [3] = 40,   -- Slight Right
    [4] = 80,   -- Hard Right
}

-- Multi-head action execution
-- throttle: 0=nothing, 1=forward(A), 2=brake(B)
-- steering: 0=hard left, 1=slight left, 2=center, 3=slight right, 4=hard right
-- drift:    0=no drift, 1=drift(R)
-- item:     0=no item, 1=use item(Z)
local function execute_multi_action(throttle, steering, drift, item)
    -- Table for digital buttons
    local digital_controls = {
        A = false,
        B = false,
        R = false,
        Z = false
    }

    -- Table for analog axes (X, Y)
    local analog_controls = {
        ["X Axis"] = 0.0,
        ["Y Axis"] = 0.0
    }

    -- Throttle
    if throttle == 1 then       -- Forward
        digital_controls.A = true
        analog_controls["Y Axis"] = 80  -- Stick "up"
    elseif throttle == 2 then   -- Brake
        digital_controls.B = true
    end

    -- Steering
    analog_controls["X Axis"] = STEERING_VALUES[steering] or 0

    -- Drift
    if drift == 1 then
        digital_controls.R = true
    end

    -- Item
    if item == 1 then
        digital_controls.Z = true
    end

    -- Send digital inputs to Player 1
    joypad.set(digital_controls, 1)
    
    -- Send analog inputs to Player 1
    joypad.setanalog(analog_controls, 1)
end

-- Read all memory addresses and return as a single byte string
local function get_state_bytes()
    local byte_chunks = {}
    for i, item in ipairs(MEMORY_MAP) do
        local addr = item[1]
        local num_bytes = item[2]
        local byte_array = memory.read_bytes_as_array(addr, num_bytes, MEMORY_DOMAIN)
        for j = 1, num_bytes do
            table.insert(byte_chunks, string.char(byte_array[j]))
        end
    end
    return table.concat(byte_chunks)
end

-- --- Main ---
console.clear()
console.log("Attempting to connect to Python server...")

local client

while true do
    client = socket.tcp()
    
    -- Check if the connection was successful
    local success, err = client:connect(HOST, PORT)

    if success then
        console.log("Connected to " .. HOST .. ":" .. PORT)
        break
    else
        console.log("Failed to connect: " .. tostring(err))
        console.log("Retrying in 2 seconds...")
        client:close()
        -- Wait for 2 seconds
        socket.select(nil, nil, 2)
    end
end

-- If we're here, the connection worked.
client:settimeout(nil) -- Block until data is received

-- Main communication loop
while true do
    -- Use 'pcall' (protected call) to safely handle disconnects
    local status, command_char_or_err = pcall(client.receive, client, 1)

    if not status then
        -- pcall failed, which means the connection was lost
        console.log("Connection error: " .. tostring(command_char_or_err))
        console.log("Disconnecting. Waiting to reconnect...")
        client:close()
        break -- Exit the inner while loop to trigger reconnection
    end

    -- If pcall succeeded, command_char_or_err holds the character
    local command_char = command_char_or_err

    if command_char == "R" then
        -- Reset command
        savestate.load(SAVESTATE_PATH)
        console.log("Loaded state: " .. SAVESTATE_PATH)

        local state_bytes = get_state_bytes()
        pcall(client.send, client, state_bytes)

    elseif command_char == "S" then
        -- Step command
        -- Receive 5 bytes: 1 byte frame_skip + 4 bytes action (throttle, steering, drift, item)
        local action_status, action_data = pcall(client.receive, client, 5)

        if not action_status then
            console.log("Connection error (receiving action): " .. tostring(action_data))
            client:close()
            break -- Exit the inner while loop to trigger reconnection
        end

        local frame_skip = string.byte(action_data, 1)
        local throttle   = string.byte(action_data, 2)
        local steering   = string.byte(action_data, 3)
        local drift      = string.byte(action_data, 4)
        local item_use   = string.byte(action_data, 5)

        -- Execute the action for frame_skip frames
        for f = 1, frame_skip do
            execute_multi_action(throttle, steering, drift, item_use)
            emu.frameadvance()
        end

        local state_bytes = get_state_bytes()
        pcall(client.send, client, state_bytes)

    elseif command_char == "D" then
        -- Demo mode: advance one frame with human input, send state + observed actions
        emu.frameadvance()

        -- Read what the human pressed
        local input = joypad.get(1)
        local analog = joypad.getanalog(1)

        -- Map human input back to multi-discrete action encoding
        local throttle = 0  -- nothing
        if input.A then throttle = 1 end  -- forward
        if input.B then throttle = 2 end  -- brake

        -- Map analog X to nearest steering index
        local x_axis = analog["X Axis"] or 0
        local steering = 2  -- center
        if x_axis < -60 then steering = 0       -- hard left
        elseif x_axis < -20 then steering = 1   -- slight left
        elseif x_axis > 60 then steering = 4    -- hard right
        elseif x_axis > 20 then steering = 3    -- slight right
        end

        local drift = 0
        if input.R then drift = 1 end
        local item = 0
        if input.Z then item = 1 end

        -- Send state bytes + 4 action bytes
        local state_bytes = get_state_bytes()
        local action_bytes = string.char(throttle, steering, drift, item)
        pcall(client.send, client, state_bytes .. action_bytes)

    elseif command_char == "C" then
        -- Close command
        console.log("Close command received. Disconnecting.")
        client:close()
        break -- Exit the inner while loop
        
    elseif command_char == nil then
        -- This can happen if the server closes the connection cleanly
        console.log("Server disconnected.")
        break -- Exit the inner while loop to trigger reconnection
    end
end