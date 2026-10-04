$ErrorActionPreference = 'Stop'
$flyRoot = $PSScriptRoot
$flyPython = Join-Path $flyRoot '.venv\Scripts\python.exe'
$flyPort = 8765
$flyStorageRoot = $flyRoot
$flyRuntimeConfig = Join-Path $flyRoot 'runtime\active-server.json'
if (Test-Path -LiteralPath $flyRuntimeConfig) {
    $flyRuntime = Get-Content -LiteralPath $flyRuntimeConfig -Raw | ConvertFrom-Json
    if ($flyRuntime.port -lt 1024 -or $flyRuntime.port -gt 65535) { throw 'Invalid configured local port.' }
    $flyPort = [int]$flyRuntime.port
    $flyStorageRoot = [System.IO.Path]::GetFullPath((Join-Path $flyRoot $flyRuntime.storage_root))
    $flyRuntimeBoundary = [System.IO.Path]::GetFullPath((Join-Path $flyRoot 'runtime')) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $flyStorageRoot.StartsWith($flyRuntimeBoundary, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Memory storage must stay in the project runtime directory.' }
}
$flyRuns = Join-Path $flyStorageRoot 'runs'
$flyAddress = "http://127.0.0.1:$flyPort"
if (-not (Test-Path -LiteralPath $flyPython)) {
    throw 'Project Python environment is missing. Follow README.md setup instructions.'
}
New-Item -ItemType Directory -Force -Path $flyRuns | Out-Null
$flyExisting = $null
try { $flyExisting = Invoke-RestMethod -Uri "$flyAddress/api/state" -TimeoutSec 2 } catch { }
if ($flyExisting -and $flyExisting.control_source -like 'MaleCNS*') {
    if (-not $flyExisting.autosave) {
        # Upgrade an already-running older instance without interrupting its brain.
        $flyCompanion = $null
        try { $flyCompanion = Invoke-RestMethod -Uri 'http://127.0.0.1:8766/status' -TimeoutSec 2 } catch { }
        if ($flyCompanion.mode -ne 'compatibility') {
            $flyCompanionStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $flyCompanionProcess = Start-Process -FilePath $flyPython -ArgumentList @('-u','scripts/autosave_current_session.py') -WorkingDirectory $flyRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $flyRuns "autosave-$flyCompanionStamp.stdout.log") -RedirectStandardError (Join-Path $flyRuns "autosave-$flyCompanionStamp.stderr.log")
            $flyCompanionProcess.Id | Set-Content -LiteralPath (Join-Path $flyRuns 'autosave-companion.pid') -Encoding ascii
        }
    }
    Start-Process $flyAddress
    Write-Host "Fly Brain Snake is already running: $flyAddress"
    exit 0
}
if (Get-NetTCPConnection -LocalPort $flyPort -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $flyPort is used by another application. Choose another local port."
}
$flyStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$flyProcess = Start-Process -FilePath $flyPython -ArgumentList @('-u','-m','flysnake.server','--port',"$flyPort",'--storage-root',('"' + $flyStorageRoot + '"')) -WorkingDirectory $flyRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $flyRuns "server-$flyStamp.stdout.log") -RedirectStandardError (Join-Path $flyRuns "server-$flyStamp.stderr.log")
$flyProcess.Id | Set-Content -LiteralPath (Join-Path $flyRuns 'server.pid') -Encoding ascii
$flyReady = $false
for ($flyAttempt=0; $flyAttempt -lt 20; $flyAttempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $flyState = Invoke-RestMethod -Uri "$flyAddress/api/state" -TimeoutSec 2
        if ($flyState.control_source -like 'MaleCNS*') { $flyReady=$true; break }
    } catch { }
    if ($flyProcess.HasExited) { break }
}
if (-not $flyReady) {
    throw "Server did not start. See $flyRuns for the startup logs."
}
Start-Process $flyAddress
Write-Host "Fly Brain Snake: $flyAddress"
Write-Host 'The page shows real neural loading and execution status.'
