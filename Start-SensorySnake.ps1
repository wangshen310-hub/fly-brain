param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
$projectDirectory=$PSScriptRoot
$pythonPath=Join-Path $projectDirectory '.venv\Scripts\python.exe'
$runName='native-sensory-basis-controls-20260930-pilot-v1'
$runRelative='runtime/'+$runName
$runDirectory=Join-Path $projectDirectory $runRelative
$statusPath=Join-Path $runDirectory 'status.json'
if (-not (Test-Path -LiteralPath $pythonPath)) {throw 'Project Python is missing.'}
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$matching=Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'scripts[/\\]train_sensory_basis.py' -and
    $_.CommandLine -match [regex]::Escape($runName)
}
if (-not $matching) {
    if (Test-Path -LiteralPath $statusPath) {
        $saved=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        if ($saved.status -eq 'complete') {
            Write-Host 'This finite experiment has completed; showing its saved results.'
        } elseif (Test-Path -LiteralPath (Join-Path $runDirectory 'STOP_AFTER_CYCLE')) {
            Write-Host 'A saved stop request is present; showing memory without resuming.'
        } else {
            $s=$saved.settings
            $trainingArguments=@('-u','scripts/train_sensory_basis.py','--output',$runRelative,
                '--seed',[string]$s.seed,'--size',[string]$s.size,'--stage',[string]$s.stage,
                '--limit',[string]$s.limit,'--cycles',[string]$s.cycles,'--batch-size',[string]$s.batch_size,
                '--evaluation-n',[string]$s.evaluation_n,'--proposal-step',[string]$s.proposal_step,
                '--rank',[string]$s.rank,'--resume')
            $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
            Start-Process -FilePath $pythonPath -ArgumentList $trainingArguments -WorkingDirectory $projectDirectory `
                -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runDirectory "resume-$stamp.out.log") `
                -RedirectStandardError (Join-Path $runDirectory "resume-$stamp.err.log") | Out-Null
            Write-Host 'Resuming from the most recent committed native synaptic memory.'
        }
    } else {
        Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/train_sensory_basis.py','--output',
            $runRelative,'--cycles','8','--batch-size','6','--rank','4','--seed','930501' `
            -WorkingDirectory $projectDirectory -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $runDirectory 'out.log') `
            -RedirectStandardError (Join-Path $runDirectory 'err.log') | Out-Null
    }
} else {Write-Host 'The trainer is already running; no duplicate was started.'}
$listener=Get-NetTCPConnection -State Listen -LocalPort 8773 -ErrorAction SilentlyContinue
if ($listener) {
    $process=Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
    if ($process.CommandLine -notmatch 'serve_sensory_snake.py') {
        throw 'Port 8773 belongs to another program; it was not stopped.'
    }
} else {
    Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/serve_sensory_snake.py','--run',$runRelative,
        '--port','8773' -WorkingDirectory $projectDirectory -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runDirectory 'web.out.log') `
        -RedirectStandardError (Join-Path $runDirectory 'web.err.log') | Out-Null
}
if (-not $NoBrowser) {Start-Process 'http://127.0.0.1:8773/'}
Write-Host 'Native sensory Snake: http://127.0.0.1:8773/'
