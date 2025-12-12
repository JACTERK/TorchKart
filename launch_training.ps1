param (
    # Default to 1 environment if not specified
    [int]$NumEnvs = 20
)

# --- Configuration ---
# Adjust these paths if your directory structure is different
$BizHawkExe = ".\bizhawk\EmuHawk.exe"
$RomPath    = "C:\Users\jacobterkuc\Documents\GitHub\project-group-94\rom\marioKart.n64"
$LuaScript  = "mk64_interface.lua"

# --- Validation ---
if (-not (Test-Path $BizHawkExe)) {
    Write-Error "Could not find EmuHawk.exe at $BizHawkExe"
    exit
}

# --- Execution Loop ---
Write-Host "Starting $NumEnvs environment(s)..." -ForegroundColor Cyan

for ($i = 1; $i -le $NumEnvs; $i++) {
    Write-Host "Launching Instance $i..."

    # We use Start-Process to ensure they run asynchronously (in parallel)
    Start-Process -FilePath $BizHawkExe -ArgumentList @(
        "`"$RomPath`"",        # Quote the path in case of spaces
        "--lua=`"$LuaScript`"" # syntax required by BizHawk
    )
}

Write-Host "All instances launched." -ForegroundColor Green