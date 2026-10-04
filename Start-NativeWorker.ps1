param(
    [Parameter(Mandatory=$true)][string]$Script,
    [Parameter(Mandatory=$true)][string[]]$WorkerArguments,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$LogPrefix
)
$ErrorActionPreference='Stop'
$nativeProjectDirectory=[System.IO.Path]::GetFullPath($PSScriptRoot)
$nativeRunDirectory=[System.IO.Path]::GetFullPath((Join-Path $nativeProjectDirectory $RunDirectory))
if (-not $nativeRunDirectory.StartsWith($nativeProjectDirectory+[System.IO.Path]::DirectorySeparatorChar,[System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Worker output directory must stay in this workspace.'
}
if ($Script -notmatch '^scripts[/\\][A-Za-z0-9_]+\.py$' -or $LogPrefix -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'Explicit workspace script and log prefix required.'
}
New-Item -ItemType Directory -Path $nativeRunDirectory -Force | Out-Null
$nativeLaunchPath=Join-Path $nativeRunDirectory ($LogPrefix+'.launch.json')
if (Test-Path -LiteralPath $nativeLaunchPath) {throw 'Preserve previous worker launch; choose a new log prefix.'}
$nativeLaunch=@{script=$Script;arguments=@($WorkerArguments);cwd=$nativeProjectDirectory;
    creation_flags=16777216;
    worker_source_sha256=(Get-FileHash -LiteralPath (Join-Path $nativeProjectDirectory 'scripts\durable_worker.py') -Algorithm SHA256).Hash.ToLower();
    stdout=(Join-Path $nativeRunDirectory ($LogPrefix+'.out.log'));
    stderr=(Join-Path $nativeRunDirectory ($LogPrefix+'.err.log'))}
[System.IO.File]::WriteAllText($nativeLaunchPath,($nativeLaunch | ConvertTo-Json -Depth 8),[System.Text.UTF8Encoding]::new($false))
$nativePython=Join-Path $nativeProjectDirectory '.venv\Scripts\pythonw.exe'
$nativeWorker=Join-Path $nativeProjectDirectory 'scripts\durable_worker.py'
# CreateProcess receives this directly; no command shell interprets arguments.
# Windows filesystem paths cannot contain a double quote.
$nativeCommand='"'+$nativePython+'" -u "'+$nativeWorker+'" --script "'+$Script+'" --launch-data "'+$nativeLaunchPath+'"'
# Win32_Process.Create itself can inherit a process job. Explicit breakaway
# is essential; pythonw prevents a console without unsupported startup flags.
$nativeStartup=New-CimInstance -ClassName Win32_ProcessStartup -Namespace root/cimv2 -ClientOnly -Property @{CreateFlags=[uint32]16777216}
$nativeCreated=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$nativeCommand;CurrentDirectory=$nativeProjectDirectory;ProcessStartupInformation=$nativeStartup}
if ($nativeCreated.ReturnValue -ne 0) {throw "Independent worker creation failed with code $($nativeCreated.ReturnValue)."}
[PSCustomObject]@{ProcessId=$nativeCreated.ProcessId;Script=$Script;LaunchData=$nativeLaunchPath;HostDependency='WMI service, no Codex job inheritance'}
