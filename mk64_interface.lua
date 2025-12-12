-- bizhawk/mk64_interface.lua

-- Load socket library
package.cpath = package.cpath .. ";?.dll"
local socket = require("socket")

-- --- Configuration ---
local HOST = "127.0.0.1"
local PORT = 65432
local SAVESTATE_PATH = "C:/Users/jacobterkuc/Documents/Github/project-group-94/mk64_start.state"
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
}

-- Define discrete actions
-- 0: Forward
-- 1: Forward + Left
-- 2: Forward + Right
-- 3: Drift Left
-- 4: Drift Right
-- 5: Turn Left (no gas)
-- 6: Turn Right (no gas)
-- 7: No action
local function execute_action(action_id)
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

    if action_id == 0 then      -- Forward
        digital_controls.A = true
        analog_controls["Y Axis"] = 80 -- Stick "up"
        
    elseif action_id == 1 then -- Forward + Left
        digital_controls.A = true
        analog_controls["Y Axis"] = 80 -- Stick "up"
        analog_controls["X Axis"] = -80 -- Stick "left"
        
    elseif action_id == 2 then -- Forward + Right
        digital_controls.A = true
        analog_controls["Y Axis"] = 80 -- Stick "up"
        analog_controls["X Axis"] = 80 -- Stick "right"

    elseif action_id == 3 then  -- Drift Left (A + R + Left)
        digital_controls.A = true
        digital_controls.R = true
        analog_controls["Y Axis"] = 80
        analog_controls["X Axis"] = -80

    elseif action_id == 4 then  -- Drift Right (A + R + Right)
        digital_controls.A = true
        digital_controls.R = true
        analog_controls["Y Axis"] = 80
        analog_controls["X Axis"] = 80

    elseif action_id == 5 then  -- Hard Turn Left (no gas)
        analog_controls["X Axis"] = -80

    elseif action_id == 6 then  -- Hard Turn Right (no gas)
        analog_controls["X Axis"] = 80

    elseif action_id == 7 then  -- Boost
        digital_controls.Z = true

    elseif action_id == 8 then
        -- Do nothing...
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
        console.log("Retrying in 10 seconds...")
        client:close()
        -- Wait for 10 seconds
        socket.select(nil, nil, 10)
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
        console.log("Disconnecting.")
        client:close()
        break -- Exit the while loop
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
        -- Receive the action ID (1 byte)
        local action_status, action_char = pcall(client.receive, client, 1)

        if not action_status then
            console.log("Connection error (receiving action): " .. tostring(action_char))
            client:close()
            break
        end

        local action_id = string.byte(action_char)
        execute_action(action_id)
        emu.frameadvance()

        local state_bytes = get_state_bytes()
        pcall(client.send, client, state_bytes)

    elseif command_char == "C" then
        -- Close command
        console.log("Close command received. Disconnecting.")
        client:close()
        break

    elseif command_char == nil then
        -- This can happen if the server closes the connection cleanly
        console.log("Server disconnected.")
        break
    end
end

console.log("Script finished.")