param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [Parameter(Mandatory = $true)]
    [int]$CurrentSupervisorPid,

    [Parameter(Mandatory = $true)]
    [int]$CurrentRunnerPid
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot
$outputDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\research\alpha_escape_continuation"
$phaseBoundary = Join-Path $outputDirectory "phase_1_breadth_repair_atoms_results.json"
$statusPath = Join-Path $outputDirectory "hybrid_handoff_status.json"
$launcher = Join-Path $RepositoryRoot "backtests\stock\auto\runners\launch_iaric_alpha_escape_continuation_background.ps1"

function Write-Status {
    param([hashtable]$Payload)
    $temporary = "$statusPath.tmp"
    $Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

function Get-Descendants {
    param([int]$RootPid)
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $descendants = @()
    $front = @($RootPid)
    while ($front.Count -gt 0) {
        $next = @()
        foreach ($parentPid in $front) {
            $children = @($all | Where-Object ParentProcessId -eq $parentPid)
            $descendants += $children
            $next += @($children.ProcessId)
        }
        $front = $next
    }
    return @($descendants)
}

Write-Status @{
    status = "waiting_for_phase_1_boundary"
    current_supervisor_pid = $CurrentSupervisorPid
    current_runner_pid = $CurrentRunnerPid
    boundary_artifact = $phaseBoundary
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}

while (-not (Test-Path -LiteralPath $phaseBoundary)) {
    $supervisor = Get-CimInstance Win32_Process -Filter "ProcessId=$CurrentSupervisorPid" -ErrorAction SilentlyContinue
    $runner = Get-CimInstance Win32_Process -Filter "ProcessId=$CurrentRunnerPid" -ErrorAction SilentlyContinue
    if ($null -eq $supervisor -or $supervisor.CommandLine -notmatch 'launch_iaric_alpha_escape_continuation_background') {
        throw "Current alpha-escape supervisor identity check failed"
    }
    if ($null -eq $runner -or $runner.CommandLine -notmatch 'run_iaric_alpha_escape_continuation') {
        throw "Current alpha-escape runner identity check failed"
    }
    Start-Sleep -Seconds 5
}

$completedBoundary = Get-Item -LiteralPath $phaseBoundary
Write-Status @{
    status = "phase_1_complete_stopping_old_runner"
    boundary_completed_at_utc = $completedBoundary.LastWriteTimeUtc.ToString("o")
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}

$supervisor = Get-CimInstance Win32_Process -Filter "ProcessId=$CurrentSupervisorPid"
$runner = Get-CimInstance Win32_Process -Filter "ProcessId=$CurrentRunnerPid"
if ($supervisor.CommandLine -notmatch 'launch_iaric_alpha_escape_continuation_background') {
    throw "Refusing to stop an unexpected supervisor process"
}
if ($runner.CommandLine -notmatch 'run_iaric_alpha_escape_continuation') {
    throw "Refusing to stop an unexpected runner process"
}
Stop-Process -Id $CurrentSupervisorPid -Force
Start-Sleep -Seconds 1
foreach ($processId in @((Get-Descendants -RootPid $CurrentRunnerPid).ProcessId | Sort-Object -Descending)) {
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}
Stop-Process -Id $CurrentRunnerPid -Force -ErrorAction SilentlyContinue

$pwsh = (Get-Command pwsh).Source
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    ('"' + $launcher + '"'),
    "-RepositoryRoot",
    ('"' + $RepositoryRoot + '"'),
    "-MaxRestarts",
    "4"
)
$newSupervisor = Start-Process -FilePath $pwsh -ArgumentList $arguments -WindowStyle Hidden -PassThru
Write-Status @{
    status = "hybrid_runner_started"
    old_supervisor_pid = $CurrentSupervisorPid
    old_runner_pid = $CurrentRunnerPid
    new_supervisor_pid = $newSupervisor.Id
    phase_0_and_phase_1_cache_preserved = $true
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}
