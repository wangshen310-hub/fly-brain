param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectDirectory=$PSScriptRoot
$chemicalRunName='native-chemical-causal-controls-20260930-pilot-v1'
$chemicalRunRelative='runtime/'+$chemicalRunName
$chemicalRunDirectory=Join-Path $projectDirectory $chemicalRunRelative
New-Item -ItemType Directory -Path $chemicalRunDirectory -Force | Out-Null
$chemicalStatusPath=Join-Path $chemicalRunDirectory 'status.json'
$chemicalMatching=Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -match 'scripts[/\\]train_chemical_controls.py' -and
    $_.CommandLine -match [regex]::Escape($chemicalRunName)
}
if (-not $chemicalMatching) {
    if (Test-Path -LiteralPath $chemicalStatusPath) {
        $chemicalSaved=Get-Content -LiteralPath $chemicalStatusPath -Raw | ConvertFrom-Json
        if ($chemicalSaved.status -eq 'complete') {Write-Host 'Finite experiment complete; opening saved results.'}
        elseif (Test-Path -LiteralPath (Join-Path $chemicalRunDirectory 'STOP_AFTER_CYCLE')) {Write-Host 'Saved boundary stop request exists; opening memory.'}
        else {
            $chemicalSettings=$chemicalSaved.settings
            $chemicalArguments=@('--output',$chemicalRunRelative,
                '--baseline-run','runtime/native-sensory-basis-controls-20260930-pilot-v1',
                '--seed',[string]$chemicalSettings.seed,'--size',[string]$chemicalSettings.size,
                '--stage',[string]$chemicalSettings.stage,'--limit',[string]$chemicalSettings.limit,
                '--cycles',[string]$chemicalSettings.cycles,'--poses-per-heading',[string]$chemicalSettings.poses_per_heading,
                '--synaptic-step',[string]$chemicalSettings.synaptic_step,'--evaluation-n',[string]$chemicalSettings.evaluation_n,'--resume')
            $chemicalStamp=Get-Date -Format 'yyyyMMdd-HHmmss'
            & (Join-Path $projectDirectory 'Start-NativeWorker.ps1') -Script 'scripts/train_chemical_controls.py' `
                -WorkerArguments $chemicalArguments -RunDirectory $chemicalRunRelative -LogPrefix "resume-$chemicalStamp" | Out-Null
            Write-Host 'Resuming committed native memory.'
        }
    } else {
        & (Join-Path $projectDirectory 'Start-NativeWorker.ps1') -Script 'scripts/train_chemical_controls.py' `
            -WorkerArguments @('--output',$chemicalRunRelative,'--baseline-run','runtime/native-sensory-basis-controls-20260930-pilot-v1',
                '--cycles','16','--seed','930701') -RunDirectory $chemicalRunRelative -LogPrefix ('initial-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) | Out-Null
    }
} else {Write-Host 'Native chemical trainer already running; no duplicate started.'}
$chemicalListener=Get-NetTCPConnection -State Listen -LocalPort 8773 -ErrorAction SilentlyContinue
if ($chemicalListener) {
    $chemicalServer=Get-CimInstance Win32_Process -Filter "ProcessId=$($chemicalListener.OwningProcess)"
    $chemicalSame=$chemicalServer.CommandLine -match 'serve_chemical_snake.py' -and $chemicalServer.CommandLine -match [regex]::Escape($chemicalRunName)
    if (-not $chemicalSame) {
        $chemicalServerParent=Get-CimInstance Win32_Process -Filter "ProcessId=$($chemicalServer.ParentProcessId)"
        $chemicalOwned=$chemicalServer.CommandLine -match [regex]::Escape($projectDirectory) -or $chemicalServerParent.CommandLine -match [regex]::Escape($projectDirectory)
        if ($chemicalServer.CommandLine -match 'scripts[/\\]serve_actor_snake.py' -and $chemicalOwned) {
            Stop-Process -Id $chemicalServer.ProcessId -Force
            $chemicalListener=$null
        } else {throw 'Port 8773 belongs to another service; that service was preserved.'}
    }
}
if (-not $chemicalListener) {
    & (Join-Path $projectDirectory 'Start-NativeWorker.ps1') -Script 'scripts/serve_chemical_snake.py' `
        -WorkerArguments @('--run',$chemicalRunRelative,'--port','8773') -RunDirectory $chemicalRunRelative `
        -LogPrefix ('web-'+(Get-Date -Format 'yyyyMMdd-HHmmss')) | Out-Null
}
if (-not $NoBrowser) {Start-Process 'http://127.0.0.1:8773/'}
Write-Host 'Native chemical Snake: http://127.0.0.1:8773/'
