param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectDirectory=$PSScriptRoot
$ordinaryRunName='ordinary-chemical-snake-v1'
$ordinaryRunRelative='runtime/'+$ordinaryRunName
$ordinaryRunDirectory=Join-Path $projectDirectory $ordinaryRunRelative
New-Item -ItemType Directory -Path $ordinaryRunDirectory -Force | Out-Null
$ordinaryStatusPath=Join-Path $ordinaryRunDirectory 'status.json'
$ordinaryMatching=Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match 'scripts[/\\]run_ordinary_chemical_snake.py' -and
    $_.CommandLine -match [regex]::Escape($ordinaryRunName)
}
if (-not $ordinaryMatching) {
    if (Test-Path -LiteralPath $ordinaryStatusPath) {
        $ordinarySaved=Get-Content -LiteralPath $ordinaryStatusPath -Raw | ConvertFrom-Json
        if ($ordinarySaved.status -eq 'complete') {Write-Host 'Bounded ordinary observation complete; preserving results.'}
        elseif (Test-Path -LiteralPath (Join-Path $ordinaryRunDirectory 'STOP_AFTER_WINDOW')) {Write-Host 'Window stop request exists; memory is preserved.'}
        else {
            $ordinarySettings=$ordinarySaved.settings
            # Keep the original parent memory and run directory: reconnecting
            # must not pick a newer training checkpoint and reset game progress.
            $ordinaryArguments=@('--run',$ordinarySettings.parent_run,'--output',$ordinaryRunRelative,
                '--checkpoint-cycle',[string]$ordinarySettings.checkpoint_cycle,
                '--arm',$ordinarySettings.arm,'--seed',[string]$ordinarySettings.seed,
                '--max-windows',[string]$ordinarySettings.max_windows,'--resume')
            $ordinaryStamp=Get-Date -Format 'yyyyMMdd-HHmmss'
            & (Join-Path $projectDirectory 'Start-NativeWorker.ps1') -Script 'scripts/run_ordinary_chemical_snake.py' `
                -WorkerArguments $ordinaryArguments -RunDirectory $ordinaryRunRelative -LogPrefix "resume-$ordinaryStamp" | Out-Null
        }
    } else {
        $ordinaryParentRelative='runtime/native-chemical-causal-controls-20260930-pilot-v1'
        $ordinaryParent=Get-Content -LiteralPath (Join-Path $projectDirectory ($ordinaryParentRelative+'/status.json')) -Raw | ConvertFrom-Json
        if (-not $ordinaryParent.checkpoints) {throw 'No committed native chemical memory exists yet.'}
        $ordinaryCycle=$ordinaryParent.checkpoints[-1].cycles_completed
        & (Join-Path $projectDirectory 'Start-NativeWorker.ps1') -Script 'scripts/run_ordinary_chemical_snake.py' `
            -WorkerArguments @('--run',$ordinaryParentRelative,'--output',$ordinaryRunRelative,
                '--checkpoint-cycle',[string]$ordinaryCycle,'--arm','learning','--seed','930901') `
            -RunDirectory $ordinaryRunRelative -LogPrefix ('initial-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) | Out-Null
    }
} else {Write-Host 'Ordinary native runner already active; no duplicate started.'}
$ordinaryListener=Get-NetTCPConnection -State Listen -LocalPort 8774 -ErrorAction SilentlyContinue
if ($ordinaryListener) {
    $ordinaryServer=Get-CimInstance Win32_Process -Filter "ProcessId=$($ordinaryListener.OwningProcess)"
    if ($ordinaryServer.CommandLine -notmatch 'serve_chemical_snake.py' -or $ordinaryServer.CommandLine -notmatch [regex]::Escape($ordinaryRunName)) {
        throw 'Port 8774 belongs to another service; that service was preserved.'
    }
} else {
    & (Join-Path $projectDirectory 'Start-NativeWorker.ps1') -Script 'scripts/serve_chemical_snake.py' `
        -WorkerArguments @('--run',$ordinaryRunRelative,'--port','8774') -RunDirectory $ordinaryRunRelative `
        -LogPrefix ('web-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) | Out-Null
}
if (-not $NoBrowser) {Start-Process 'http://127.0.0.1:8774/'}
Write-Host 'Ordinary frozen native Snake: http://127.0.0.1:8774/'
