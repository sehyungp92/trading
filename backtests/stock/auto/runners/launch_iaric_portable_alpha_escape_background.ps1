param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [int]$MaxRestarts = 3
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $RepositoryRoot

$outputDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_4\portable_alpha_escape"
$supervisorStatusPath = Join-Path $outputDirectory "supervisor_status.json"
$pollSeconds = 30
$stallTimeoutSeconds = 1800
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

function Write-JsonStatus {
    param([string]$Path, [hashtable]$Payload)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-DescendantProcesses {
    param([int]$RootPid)
    $all = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $descendants = @()
    $front = @($RootPid)
    while ($front.Count -gt 0) {
        $next = @()
        foreach ($parentPid in $front) {
            $children = @($all | Where-Object ParentProcessId -eq $parentPid)
            if ($children.Count -gt 0) {
                $descendants += $children
                $next += @($children.ProcessId)
            }
        }
        $front = $next
    }
    return @($descendants)
}

function Stop-RunnerTree {
    param([int]$RunnerPid)
    $descendants = @(Get-DescendantProcesses -RootPid $RunnerPid)
    foreach ($processId in @($descendants.ProcessId | Sort-Object -Descending)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $RunnerPid -Force -ErrorAction SilentlyContinue
}

function Get-LatestOutputWriteUtc {
    $files = @(Get-ChildItem -LiteralPath $outputDirectory -File -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) { return [DateTime]::UtcNow }
    return ($files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc
}

for ($attempt = 1; $attempt -le $MaxRestarts; $attempt++) {
    $attemptStdout = Join-Path $outputDirectory "attempt_${attempt}_stdout.log"
    $attemptStderr = Join-Path $outputDirectory "attempt_${attempt}_stderr.log"
    $arguments = @(
        "-m", "backtests.stock.auto.runners.run_iaric_portable_alpha_escape",
        "--output-dir", $outputDirectory,
        "--max-workers", "2"
    )
    $runner = Start-Process -FilePath "python" -ArgumentList $arguments -WindowStyle Hidden `
        -RedirectStandardOutput $attemptStdout -RedirectStandardError $attemptStderr -PassThru
    $lastActivityUtc = [DateTime]::UtcNow
    $lastWriteUtc = Get-LatestOutputWriteUtc
    $lastCpuSeconds = 0.0
    $watchdogReason = ""
    while (-not $runner.HasExited) {
        Start-Sleep -Seconds $pollSeconds
        $runner.Refresh()
        if ($runner.HasExited) { break }
        $descendants = @(Get-DescendantProcesses -RootPid $runner.Id)
        $processIds = @($runner.Id) + @($descendants.ProcessId)
        $processes = @($processIds | ForEach-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        $cpuSeconds = [double](($processes | Measure-Object CPU -Sum).Sum)
        $latestWriteUtc = Get-LatestOutputWriteUtc
        if (($cpuSeconds - $lastCpuSeconds) -ge 1.0 -or $latestWriteUtc -gt $lastWriteUtc) { $lastActivityUtc = [DateTime]::UtcNow }
        $lastCpuSeconds = $cpuSeconds
        $lastWriteUtc = $latestWriteUtc
        $idleSeconds = ([DateTime]::UtcNow - $lastActivityUtc).TotalSeconds
        Write-JsonStatus -Path $supervisorStatusPath -Payload @{
            status = "monitoring_targeted_portable_alpha_escape"
            attempt = $attempt
            runner_pid = $runner.Id
            descendant_pids = @($descendants.ProcessId)
            total_cpu_seconds = [Math]::Round($cpuSeconds, 2)
            idle_seconds = [Math]::Round($idleSeconds, 1)
            stall_timeout_seconds = $stallTimeoutSeconds
            last_output_write_utc = $latestWriteUtc.ToString("o")
            updated_at_utc = [DateTime]::UtcNow.ToString("o")
        }
        if ($idleSeconds -ge $stallTimeoutSeconds) {
            $watchdogReason = "no process CPU or output activity for $stallTimeoutSeconds seconds"
            Stop-RunnerTree -RunnerPid $runner.Id
            break
        }
    }
    $runner.Refresh()
    $exitCode = if ($watchdogReason) { 124 } elseif ($runner.HasExited) { $runner.ExitCode } else { 125 }
    if ($exitCode -in @(0, 2)) {
        Write-JsonStatus -Path $supervisorStatusPath -Payload @{
            status = if ($exitCode -eq 0) { "complete_value_verified" } else { "complete_gates_blocked" }
            attempt = $attempt
            exit_code = $exitCode
            updated_at_utc = [DateTime]::UtcNow.ToString("o")
        }
        exit $exitCode
    }
    Write-JsonStatus -Path $supervisorStatusPath -Payload @{
        status = "restart_pending"
        attempt = $attempt
        exit_code = $exitCode
        watchdog_reason = $watchdogReason
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    if ($attempt -lt $MaxRestarts) { Start-Sleep -Seconds ([Math]::Min(15 * $attempt, 60)) }
}

Write-JsonStatus -Path $supervisorStatusPath -Payload @{
    status = "failed_requires_attention"
    attempts = $MaxRestarts
    exit_code = $exitCode
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}
exit $exitCode
