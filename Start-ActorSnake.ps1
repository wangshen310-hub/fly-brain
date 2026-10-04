param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectDirectory=$PSScriptRoot
$pythonPath=Join-Path $projectDirectory '.venv\Scripts\python.exe'
$actorRunName='native-actor-credit-controls-20260930-pilot-v1'
$actorRunRelative='runtime/'+$actorRunName
$actorRunDirectory=Join-Path $projectDirectory $actorRunRelative
New-Item -ItemType Directory -Path $actorRunDirectory -Force | Out-Null
$actorStatusPath=Join-Path $actorRunDirectory 'status.json'
$actorMatching=Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'scripts[/\\]train_actor_credit.py' -and
    $_.CommandLine -match [regex]::Escape($actorRunName)
}
if (-not $actorMatching) {
    if (Test-Path -LiteralPath $actorStatusPath) {
        $actorSaved=Get-Content -LiteralPath $actorStatusPath -Raw | ConvertFrom-Json
        if ($actorSaved.status -eq 'complete') {Write-Host 'Finite experiment complete; opening saved results.'}
        elseif (Test-Path -LiteralPath (Join-Path $actorRunDirectory 'STOP_AFTER_CYCLE')) {Write-Host 'Saved boundary stop request exists; opening memory.'}
        else {
            $actorSettings=$actorSaved.settings
            $actorArguments=@('-u','scripts/train_actor_credit.py','--output',$actorRunRelative,
                '--baseline-run','runtime/native-sensory-basis-controls-20260930-pilot-v1',
                '--seed',[string]$actorSettings.seed,'--size',[string]$actorSettings.size,'--stage',[string]$actorSettings.stage,
                '--limit',[string]$actorSettings.limit,'--cycles',[string]$actorSettings.cycles,
                '--batch-size',[string]$actorSettings.batch_size,'--evaluation-n',[string]$actorSettings.evaluation_n,
                '--noise-sigma',[string]$actorSettings.noise_sigma,'--edge-step',[string]$actorSettings.edge_step,'--resume')
            $actorStamp=Get-Date -Format 'yyyyMMdd-HHmmss'
            Start-Process -FilePath $pythonPath -ArgumentList $actorArguments -WorkingDirectory $projectDirectory -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $actorRunDirectory "resume-$actorStamp.out.log") `
                -RedirectStandardError (Join-Path $actorRunDirectory "resume-$actorStamp.err.log") | Out-Null
            Write-Host 'Resuming committed native memory.'
        }
    } else {
        Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/train_actor_credit.py','--output',$actorRunRelative,
            '--baseline-run','runtime/native-sensory-basis-controls-20260930-pilot-v1','--cycles','8','--batch-size','6','--seed','930601' `
            -WorkingDirectory $projectDirectory -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $actorRunDirectory 'out.log') `
            -RedirectStandardError (Join-Path $actorRunDirectory 'err.log') | Out-Null
    }
} else {Write-Host 'Native trainer already running; no duplicate started.'}
$actorListener=Get-NetTCPConnection -State Listen -LocalPort 8773 -ErrorAction SilentlyContinue
if ($actorListener) {
    $actorServer=Get-CimInstance Win32_Process -Filter "ProcessId=$($actorListener.OwningProcess)"
    if ($actorServer.CommandLine -notmatch 'serve_actor_snake.py' -or $actorServer.CommandLine -notmatch [regex]::Escape($actorRunName)) {
        throw 'Port 8773 belongs to another service; that service was preserved.'
    }
} else {
    Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/serve_actor_snake.py','--run',$actorRunRelative,'--port','8773' `
        -WorkingDirectory $projectDirectory -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $actorRunDirectory 'web.out.log') `
        -RedirectStandardError (Join-Path $actorRunDirectory 'web.err.log') | Out-Null
}
if (-not $NoBrowser) {Start-Process 'http://127.0.0.1:8773/'}
Write-Host 'Native reward Snake: http://127.0.0.1:8773/'
