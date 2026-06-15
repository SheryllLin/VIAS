# VIAS Real-Time System Startup Script
# ============================================================================
# Purpose: Start backend, frontend, and open browser automatically
# Author: VIAS Development Team
# Date: 2026-06-15
# Usage: .\START_REALTIME.ps1
# ============================================================================

param(
    [switch]$NoOpen,  # Don't open browser
    [switch]$Verbose  # Show detailed output
)

# Configuration
$ProjectRoot = $PSScriptRoot
$BackendPort = 8000
$FrontendPort = 7860
$BackendURL = "http://127.0.0.1:$BackendPort"
$FrontendURL = "http://127.0.0.1:$FrontendPort"

# Colors for output
$Green = 'Green'
$Yellow = 'Yellow'
$Red = 'Red'
$Cyan = 'Cyan'

# ============================================================================
# FUNCTIONS
# ============================================================================

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════════╗" -ForegroundColor $Cyan
    Write-Host "║ $Message" -ForegroundColor $Cyan
    Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor $Cyan
    Write-Host ""
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor $Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $Red
}

function Test-Port {
    param([int]$Port)
    $Connection = @{
        ComputerName = '127.0.0.1'
        Port = $Port
        ErrorAction = 'SilentlyContinue'
    }
    
    if (Test-NetConnection @Connection -WarningAction SilentlyContinue | Where-Object { $_.TcpTestSucceeded }) {
        return $true
    }
    return $false
}

function Kill-ProcessOnPort {
    param([int]$Port)
    
    try {
        $Process = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
        if ($Process) {
            $PID = $Process.OwningProcess
            Stop-Process -Id $PID -Force -ErrorAction SilentlyContinue
            Write-Warning "Killed process on port $Port (PID: $PID)"
            Start-Sleep -Seconds 1
        }
    }
    catch {
        # Process might not exist
    }
}

function Wait-For-Service {
    param(
        [string]$URL,
        [string]$ServiceName,
        [int]$MaxWait = 30
    )
    
    Write-Host "⏳ Waiting for $ServiceName to be ready..." -ForegroundColor $Cyan
    
    $StartTime = Get-Date
    $Ready = $false
    
    while ((Get-Date) - $StartTime -lt (New-TimeSpan -Seconds $MaxWait)) {
        try {
            $Response = Invoke-WebRequest -Uri $URL -TimeoutSec 1 -ErrorAction SilentlyContinue
            if ($Response.StatusCode -eq 200) {
                $Ready = $true
                break
            }
        }
        catch {
            # Still waiting...
        }
        
        Write-Host "." -NoNewline -ForegroundColor $Cyan
        Start-Sleep -Seconds 1
    }
    
    Write-Host ""
    
    if ($Ready) {
        Write-Success "$ServiceName is ready!"
        return $true
    }
    else {
        Write-Error-Custom "$ServiceName failed to start within $MaxWait seconds"
        return $false
    }
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================

Clear-Host

Write-Header "VIAS Real-Time System Startup"

# Check if running in correct directory
if (-not (Test-Path "$ProjectRoot\VIAS\backend\main.py")) {
    Write-Error-Custom "Error: Not in correct directory!"
    Write-Error-Custom "Expected: VIAS-main directory"
    Write-Error-Custom "Current: $ProjectRoot"
    exit 1
}

Write-Success "Found VIAS project in: $ProjectRoot"

# Check for Python
Write-Host ""
Write-Host "🔍 Checking prerequisites..." -ForegroundColor $Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error-Custom "Python not found in PATH"
    Write-Warning "Install Python or add to PATH"
    exit 1
}
Write-Success "Python found: $(python --version)"

# ============================================================================
# CLEAR PORTS (Optional)
# ============================================================================

Write-Host ""
Write-Host "🧹 Checking ports..." -ForegroundColor $Cyan

if (Test-Port $BackendPort) {
    Write-Warning "Port $BackendPort already in use"
    $Kill = Read-Host "Kill process? (y/n)"
    if ($Kill -eq 'y') {
        Kill-ProcessOnPort $BackendPort
    }
}
else {
    Write-Success "Port $BackendPort is free"
}

if (Test-Port $FrontendPort) {
    Write-Warning "Port $FrontendPort already in use"
    $Kill = Read-Host "Kill process? (y/n)"
    if ($Kill -eq 'y') {
        Kill-ProcessOnPort $FrontendPort
    }
}
else {
    Write-Success "Port $FrontendPort is free"
}

# ============================================================================
# START BACKEND
# ============================================================================

Write-Header "Starting Backend"

Write-Host "🚀 Starting FastAPI backend on $BackendURL" -ForegroundColor $Cyan
Write-Host "Command: python -m uvicorn backend.main:app --host 127.0.0.1 --port $BackendPort --reload" -ForegroundColor $Yellow
Write-Host ""

$BackendProcess = Start-Process -FilePath python `
    -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port $BackendPort --reload" `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -PassThru

Write-Success "Backend process started (PID: $($BackendProcess.Id))"

# Wait for backend to be ready
if (-not (Wait-For-Service -URL "$BackendURL/health" -ServiceName "Backend")) {
    Write-Error-Custom "Backend failed to start. Check the error above."
    Stop-Process -Id $BackendProcess.Id -ErrorAction SilentlyContinue
    exit 1
}

# ============================================================================
# START FRONTEND
# ============================================================================

Write-Header "Starting Frontend"

Write-Host "🎨 Starting Gradio frontend on $FrontendURL" -ForegroundColor $Cyan
Write-Host "Command: python VIAS/frontend/gradio_ui/app.py" -ForegroundColor $Yellow
Write-Host ""

$FrontendProcess = Start-Process -FilePath python `
    -ArgumentList "VIAS/frontend/gradio_ui/app.py" `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -PassThru

Write-Success "Frontend process started (PID: $($FrontendProcess.Id))"

# Wait for frontend to be ready
if (-not (Wait-For-Service -URL $FrontendURL -ServiceName "Frontend")) {
    Write-Error-Custom "Frontend failed to start. Check the error above."
    Stop-Process -Id $BackendProcess.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $FrontendProcess.Id -ErrorAction SilentlyContinue
    exit 1
}

# ============================================================================
# OPEN BROWSER
# ============================================================================

Write-Header "System Ready!"

Write-Success "Backend: $BackendURL"
Write-Success "Frontend: $FrontendURL"
Write-Success "API Docs: $BackendURL/docs"

Write-Host ""
Write-Host "📋 What to do now:" -ForegroundColor $Cyan
Write-Host "  1. Open browser to $FrontendURL"
Write-Host "  2. Go to Tab 1: Upload a video"
Write-Host "  3. Go to Tab 2: Register a person"
Write-Host "  4. Go to Tab 3: Try a query"
Write-Host "  5. Go to Tab 4: Check analytics"
Write-Host ""

if (-not $NoOpen) {
    Write-Host "🌐 Opening browser..." -ForegroundColor $Cyan
    Start-Process $FrontendURL
    Write-Host ""
}

# ============================================================================
# MONITORING
# ============================================================================

Write-Host "📊 System monitoring (press Ctrl+C to stop):" -ForegroundColor $Cyan
Write-Host ""

# Create a hashtable to track process info
$Processes = @{
    Backend = $BackendProcess
    Frontend = $FrontendProcess
}

# Monitor processes
try {
    while ($true) {
        $BackendStatus = if ($BackendProcess.HasExited) { "STOPPED" } else { "RUNNING" }
        $FrontendStatus = if ($FrontendProcess.HasExited) { "STOPPED" } else { "RUNNING" }
        
        $BackendColor = if ($BackendStatus -eq "RUNNING") { $Green } else { $Red }
        $FrontendColor = if ($FrontendStatus -eq "RUNNING") { $Green } else { $Red }
        
        Write-Host "`rBackend: $BackendStatus | Frontend: $FrontendStatus" -NoNewline -ForegroundColor $Cyan
        
        if ($BackendProcess.HasExited -or $FrontendProcess.HasExited) {
            Write-Host ""
            Write-Error-Custom "A service has stopped!"
            break
        }
        
        Start-Sleep -Seconds 2
    }
}
catch {
    # Ctrl+C pressed or error
}
finally {
    Write-Host ""
    Write-Host ""
    Write-Header "Shutting Down"
    
    Write-Host "🛑 Stopping services..." -ForegroundColor $Yellow
    
    if (-not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -ErrorAction SilentlyContinue
        Write-Success "Backend stopped"
    }
    
    if (-not $FrontendProcess.HasExited) {
        Stop-Process -Id $FrontendProcess.Id -ErrorAction SilentlyContinue
        Write-Success "Frontend stopped"
    }
    
    Write-Success "Cleanup complete"
    Write-Host ""
}
