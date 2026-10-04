$ErrorActionPreference = 'Stop'
$labRoot = $PSScriptRoot
$labPython = Join-Path $labRoot '.venv\Scripts\python.exe'
$labStorage = Join-Path $labRoot 'runtime\lab-v4'
$labAddress = 'http://127.0.0.1:8772'
$labExisting = $null
try { $labExisting = Invoke-RestMethod -Uri "$labAddress/api/state" -TimeoutSec 2 } catch { }
if ($labExisting -and ($labExisting.version -like 'malecns-lab-*' -or $labExisting.phase -eq 'not_started')) {
    Start-Process $labAddress
    exit 0
}
if (Get-NetTCPConnection -LocalPort 8772 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 8772 is already occupied.' }
New-Item -ItemType Directory -Force -Path $labStorage | Out-Null
$labStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$labProcess = Start-Process -FilePath $labPython -ArgumentList @('-u','-m','flysnake.lab.server','--port','8772') -WorkingDirectory $labRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $labStorage "server-$labStamp.stdout.log") -RedirectStandardError (Join-Path $labStorage "server-$labStamp.stderr.log")
$labProcess.Id | Set-Content -LiteralPath (Join-Path $labStorage 'server.pid') -Encoding ascii
for ($labAttempt=0; $labAttempt -lt 20; $labAttempt++) {
    Start-Sleep -Milliseconds 500
    try { $labReady=Invoke-RestMethod -Uri "$labAddress/api/state" -TimeoutSec 2; break } catch { }
}
if (-not $labReady) { throw 'Lab did not start; inspect runtime/lab-v4 logs.' }
Start-Process $labAddress
