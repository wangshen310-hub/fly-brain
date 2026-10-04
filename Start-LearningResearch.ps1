param([switch]$ResumeTraining, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$pythonPath = Join-Path $projectDirectory '.venv\Scripts\python.exe'
$experimentName = 'paired-native-snake-event-v2'
$experimentDirectory = Join-Path $projectDirectory ('runtime\' + $experimentName)
$statusPath = Join-Path $experimentDirectory 'status.json'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project Python environment is missing.' }
$listener = Get-NetTCPConnection -LocalPort 8773 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $listenerProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
    if ($listenerProcess.CommandLine -notmatch 'serve_learning_research.py') {
        throw 'Port 8773 belongs to another program; it was not stopped.'
    }
} else {
    Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/serve_learning_research.py','--port','8773' `
        -WorkingDirectory $projectDirectory -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $projectDirectory 'runtime\research-dashboard.out.log') `
        -RedirectStandardError (Join-Path $projectDirectory 'runtime\research-dashboard.err.log') | Out-Null
}
if ($ResumeTraining) {
    if (-not (Test-Path -LiteralPath $statusPath)) { throw 'No experiment memory exists to resume.' }
    $saved = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($saved.status -eq 'complete') {
        Write-Host 'This finite experiment is complete; its memory and results are preserved.'
    } else {
        $matching = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train_paired_native_snake.py' -and
            $_.CommandLine -match [regex]::Escape($experimentName)
        }
        if ($matching) {
            Write-Host 'The experiment is already running. No duplicate trainer was started.'
        } else {
            if (Test-Path -LiteralPath (Join-Path $experimentDirectory 'STOP_AFTER_GENERATION')) {
                throw 'A stop request is present. Remove that request file before explicitly resuming.'
            }
            $s = $saved.settings
            $trainingArguments = @('-u','scripts/train_paired_native_snake.py','--output',
                ('runtime/' + $experimentName),'--arm',$s.arm,'--seed',[string]$s.seed,
                '--generations',[string]$s.generations,'--pairs',[string]$s.pairs,
                '--batch-size',[string]$s.batch_size,'--size',[string]$s.size,
                '--limit',[string]$s.limit,'--evaluation-n',[string]$s.evaluation_n,'--resume')
            $logStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            Start-Process -FilePath $pythonPath -ArgumentList $trainingArguments `
                -WorkingDirectory $projectDirectory -WindowStyle Hidden `
                -RedirectStandardOutput (Join-Path $experimentDirectory ("resume-$logStamp.out.log")) `
                -RedirectStandardError (Join-Path $experimentDirectory ("resume-$logStamp.err.log")) | Out-Null
            Write-Host 'Resume process started; inspect the dashboard and log for source/checkpoint validation.'
        }
    }
}
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8773/' }
Write-Host 'Research dashboard: http://127.0.0.1:8773/'
