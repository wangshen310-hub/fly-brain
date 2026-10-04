param([switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$gameRoot = $PSScriptRoot
$gamePython = Join-Path $gameRoot '.venv\Scripts\python.exe'
$gameRun = Join-Path $gameRoot 'runtime\live-paired-snake-v1'
$gameUrl = 'http://127.0.0.1:8769/'
if (-not (Test-Path -LiteralPath $gamePython)) { throw 'Project Python environment is missing.' }
$listener = Get-NetTCPConnection -LocalPort 8769 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    try { $gameState = Invoke-RestMethod -Uri ($gameUrl + 'api/state') -TimeoutSec 3 } catch { }
    if ($gameState.protocol -ne 'live-native-paired-snake-v1') {
        throw 'Port 8769 is in use by a different program.'
    }
} else {
    New-Item -ItemType Directory -Force -Path $gameRun | Out-Null
    $gameStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Start-Process -FilePath $gamePython `
        -ArgumentList @('-u','-m','flysnake.research.live_server','--port','8769') `
        -WorkingDirectory $gameRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $gameRun "server-$gameStamp.out.log") `
        -RedirectStandardError (Join-Path $gameRun "server-$gameStamp.err.log") | Out-Null
    $gameState = $null
    for ($gameTry = 0; $gameTry -lt 30; $gameTry++) {
        Start-Sleep -Milliseconds 500
        try { $gameState = Invoke-RestMethod -Uri ($gameUrl + 'api/state') -TimeoutSec 2; break } catch { }
    }
    if (-not $gameState) { throw 'Game server did not start; inspect the runtime log.' }
}
if (-not $NoBrowser) { Start-Process $gameUrl }
Write-Host ('Fly-brain Snake: ' + $gameUrl)
