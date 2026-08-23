param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [int]$MaxRestarts = 4,

    [switch]$SkipFoldValidation
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $RepositoryRoot

$outputDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\research\alpha_escape_continuation"
$stdoutPath = Join-Path $outputDirectory "background_stdout.log"
$stderrPath = Join-Path $outputDirectory "background_stderr.log"
$supervisorStatusPath = Join-Path $outputDirectory "supervisor_status.json"
$stallTimeoutSeconds = 1200
$pollSeconds = 30
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$attemptOffset = 0
$existingAttempts = @(
    Get-ChildItem -LiteralPath $outputDirectory -Filter "attempt_*_stdout.log" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.BaseName -match '^attempt_(\d+)_stdout$') {
                [int]$Matches[1]
            }
        }
)
if ($existingAttempts.Count -gt 0) {
    $attemptOffset = [int](($existingAttempts | Measure-Object -Maximum).Maximum)
}

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

function Get-LatestResearchWriteUtc {
    $files = @(Get-ChildItem -LiteralPath $outputDirectory -File -ErrorAction SilentlyContinue)
    if ($files.Count -eq 0) {
        return [DateTime]::UtcNow
    }
    return ($files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc
}

for ($attempt = 1; $attempt -le $MaxRestarts; $attempt++) {
    $attemptNumber = $attemptOffset + $attempt
    $started = [DateTime]::UtcNow
    "[$($started.ToString('o'))] Starting IARIC alpha-escape continuation attempt $attemptNumber." |
        Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
    $attemptStdout = Join-Path $outputDirectory "attempt_${attemptNumber}_stdout.log"
    $attemptStderr = Join-Path $outputDirectory "attempt_${attemptNumber}_stderr.log"
    $arguments = @(
        "-m",
        "backtests.stock.auto.runners.run_iaric_alpha_escape_continuation",
        "--output-dir",
        $outputDirectory,
        "--max-workers",
        "2"
    )
    if ($SkipFoldValidation) {
        $arguments += "--skip-fold-validation"
    }
    $runner = Start-Process -FilePath "python" -ArgumentList $arguments `
        -WindowStyle Hidden -RedirectStandardOutput $attemptStdout `
        -RedirectStandardError $attemptStderr -PassThru

    $lastActivityUtc = [DateTime]::UtcNow
    $lastWriteUtc = Get-LatestResearchWriteUtc
    $lastCpuSeconds = 0.0
    $watchdogReason = ""
    while (-not $runner.HasExited) {
        Start-Sleep -Seconds $pollSeconds
        $runner.Refresh()
        if ($runner.HasExited) {
            break
        }
        $descendants = @(Get-DescendantProcesses -RootPid $runner.Id)
        $processIds = @($runner.Id) + @($descendants.ProcessId)
        $processes = @($processIds | ForEach-Object {
            Get-Process -Id $_ -ErrorAction SilentlyContinue
        })
        $cpuSeconds = [double](($processes | Measure-Object CPU -Sum).Sum)
        $latestWriteUtc = Get-LatestResearchWriteUtc
        if (($cpuSeconds - $lastCpuSeconds) -ge 1.0 -or $latestWriteUtc -gt $lastWriteUtc) {
            $lastActivityUtc = [DateTime]::UtcNow
        }
        $lastCpuSeconds = $cpuSeconds
        $lastWriteUtc = $latestWriteUtc
        $idleSeconds = ([DateTime]::UtcNow - $lastActivityUtc).TotalSeconds
        Write-JsonStatus -Path $supervisorStatusPath -Payload @{
            status = "monitoring"
            attempt = $attemptNumber
            runner_pid = $runner.Id
            descendant_pids = @($descendants.ProcessId)
            total_cpu_seconds = [Math]::Round($cpuSeconds, 2)
            idle_seconds = [Math]::Round($idleSeconds, 1)
            stall_timeout_seconds = $stallTimeoutSeconds
            last_research_write_utc = $latestWriteUtc.ToString("o")
            updated_at_utc = [DateTime]::UtcNow.ToString("o")
        }
        if ($idleSeconds -ge $stallTimeoutSeconds) {
            $watchdogReason = "no process CPU or research-file activity for $stallTimeoutSeconds seconds"
            Stop-RunnerTree -RunnerPid $runner.Id
            break
        }
    }

    $runner.Refresh()
    $exitCode = if ($watchdogReason) { 124 } elseif ($runner.HasExited) { $runner.ExitCode } else { 125 }
    foreach ($pair in @(@($attemptStdout, $stdoutPath), @($attemptStderr, $stderrPath))) {
        if (Test-Path -LiteralPath $pair[0]) {
            Get-Content -LiteralPath $pair[0] |
                Out-File -LiteralPath $pair[1] -Encoding utf8 -Append
        }
    }
    $finished = [DateTime]::UtcNow.ToString("o")
    "[$finished] Alpha-escape attempt $attemptNumber exited with code $exitCode." |
        Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
    if ($exitCode -eq 0) {
        Write-JsonStatus -Path $supervisorStatusPath -Payload @{
            status = "complete"
            attempt = $attemptNumber
            exit_code = 0
            updated_at_utc = $finished
        }
        exit 0
    }
    $restartPayload = @{
        status = "restart_pending"
        attempt = $attemptNumber
        exit_code = $exitCode
        watchdog_reason = $watchdogReason
        updated_at_utc = $finished
    }
    Write-JsonStatus -Path (Join-Path $outputDirectory "queue_status.json") -Payload $restartPayload
    Write-JsonStatus -Path $supervisorStatusPath -Payload $restartPayload
    if ($attempt -lt $MaxRestarts) {
        Start-Sleep -Seconds ([Math]::Min(15 * $attempt, 60))
    }
}

$failure = @{
    status = "failed_requires_attention"
    attempts = $MaxRestarts
    exit_code = $exitCode
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}
Write-JsonStatus -Path (Join-Path $outputDirectory "queue_status.json") -Payload $failure
Write-JsonStatus -Path $supervisorStatusPath -Payload $failure
exit $exitCode
