param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [Parameter(Mandatory = $true)]
    [int]$WaitForSupervisorPid,

    [int]$AdoptRunnerPid = 0
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $RepositoryRoot

$outputDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\structural_challenger"
$stdoutPath = Join-Path $outputDirectory "background_stdout.log"
$stderrPath = Join-Path $outputDirectory "background_stderr.log"
$supervisorStatusPath = Join-Path $outputDirectory "supervisor_status.json"
$handoffRequestPath = Join-Path $outputDirectory "planned_stage_handoff.json"
$stallTimeoutSeconds = 600
$pollSeconds = 30

function Write-JsonStatus {
    param([string]$Path, [hashtable]$Payload)
    $temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temporary -Encoding utf8
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
    $paths = @(
        (Join-Path $outputDirectory "evaluation_cache.json"),
        (Join-Path $outputDirectory "structural_screen_cache.json"),
        (Join-Path $outputDirectory "progress.json"),
        (Join-Path $outputDirectory "queue_status.json")
    )
    $files = @($paths | ForEach-Object {
        Get-Item -LiteralPath $_ -ErrorAction SilentlyContinue
    })
    if ($files.Count -eq 0) {
        return [DateTime]::UtcNow
    }
    return ($files | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1).LastWriteTimeUtc
}

if ($WaitForSupervisorPid -gt 0) {
    Wait-Process -Id $WaitForSupervisorPid -ErrorAction SilentlyContinue
}

for ($attempt = 1; $attempt -le 6; $attempt++) {
    $started = [DateTime]::UtcNow
    "[$($started.ToString('o'))] Starting IARIC structural challenger attempt $attempt." |
        Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

    $attemptStdout = Join-Path $outputDirectory "attempt_${attempt}_stdout.log"
    $attemptStderr = Join-Path $outputDirectory "attempt_${attempt}_stderr.log"
    $arguments = @(
        "-m",
        "backtests.stock.auto.runners.run_iaric_structural_challenger",
        "--output-dir",
        $outputDirectory,
        "--max-workers",
        "2"
    )
    if ($attempt -eq 1 -and $AdoptRunnerPid -gt 0) {
        $runner = Get-Process -Id $AdoptRunnerPid -ErrorAction SilentlyContinue
        if ($null -eq $runner) {
            throw "Cannot adopt missing runner PID $AdoptRunnerPid"
        }
        "[$($started.ToString('o'))] Adopting active runner PID $AdoptRunnerPid." |
            Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
    }
    else {
        $runner = Start-Process -FilePath "python" -ArgumentList $arguments `
            -WindowStyle Hidden -RedirectStandardOutput $attemptStdout `
            -RedirectStandardError $attemptStderr -PassThru
    }

    $lastActivityUtc = [DateTime]::UtcNow
    $lastResearchWriteUtc = Get-LatestResearchWriteUtc
    $lastCpuSeconds = 0.0
    $watchdogReason = ""
    $plannedHandoff = $false

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
        if (($cpuSeconds - $lastCpuSeconds) -ge 1.0 -or $latestWriteUtc -gt $lastResearchWriteUtc) {
            $lastActivityUtc = [DateTime]::UtcNow
        }
        $lastCpuSeconds = $cpuSeconds
        $lastResearchWriteUtc = $latestWriteUtc
        $idleSeconds = ([DateTime]::UtcNow - $lastActivityUtc).TotalSeconds
        Write-JsonStatus -Path $supervisorStatusPath -Payload @{
            status = "monitoring"
            attempt = $attempt
            runner_pid = $runner.Id
            descendant_pids = @($descendants.ProcessId)
            total_cpu_seconds = [Math]::Round($cpuSeconds, 2)
            idle_seconds = [Math]::Round($idleSeconds, 1)
            stall_timeout_seconds = $stallTimeoutSeconds
            last_research_write_utc = $latestWriteUtc.ToString("o")
            updated_at_utc = [DateTime]::UtcNow.ToString("o")
        }
        if (Test-Path -LiteralPath $handoffRequestPath) {
            $handoff = Get-Content -LiteralPath $handoffRequestPath -Raw |
                ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($null -ne $handoff -and $handoff.status -eq "pending") {
                $stageResultPath = Join-Path $outputDirectory ([string]$handoff.stage_result)
                $stageResult = Get-Item -LiteralPath $stageResultPath -ErrorAction SilentlyContinue
                $requestedUtc = [DateTimeOffset]::Parse(
                    [string]$handoff.requested_at_utc
                ).UtcDateTime
                if ($null -ne $stageResult -and $stageResult.LastWriteTimeUtc -ge $requestedUtc) {
                    Write-JsonStatus -Path $handoffRequestPath -Payload @{
                        status = "triggered"
                        stage_result = [string]$handoff.stage_result
                        reason = [string]$handoff.reason
                        runner_pid = $runner.Id
                        stage_completed_at_utc = $stageResult.LastWriteTimeUtc.ToString("o")
                        triggered_at_utc = [DateTime]::UtcNow.ToString("o")
                    }
                    $plannedHandoff = $true
                    $watchdogReason = "planned stage-boundary code handoff"
                    Stop-RunnerTree -RunnerPid $runner.Id
                    break
                }
            }
        }
        if ($idleSeconds -ge $stallTimeoutSeconds) {
            $watchdogReason = "no process CPU or research-file activity for $stallTimeoutSeconds seconds"
            Stop-RunnerTree -RunnerPid $runner.Id
            break
        }
    }

    $runner.Refresh()
    $exitCode = if ($plannedHandoff) { 75 } elseif ($watchdogReason) { 124 } elseif ($runner.HasExited) { $runner.ExitCode } else { 125 }
    foreach ($sourceAndTarget in @(
        @($attemptStdout, $stdoutPath),
        @($attemptStderr, $stderrPath)
    )) {
        if (Test-Path -LiteralPath $sourceAndTarget[0]) {
            Get-Content -LiteralPath $sourceAndTarget[0] |
                Out-File -LiteralPath $sourceAndTarget[1] -Encoding utf8 -Append
        }
    }
    $finished = [DateTime]::UtcNow.ToString("o")
    "[$finished] Structural challenger attempt $attempt exited with code $exitCode." |
        Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
    if ($exitCode -eq 0) {
        Write-JsonStatus -Path $supervisorStatusPath -Payload @{
            status = "complete"
            attempt = $attempt
            exit_code = 0
            updated_at_utc = $finished
        }
        exit 0
    }

    $statusPayload = @{
        status = if ($plannedHandoff) { "planned_restart_pending" } else { "restart_pending" }
        attempt = $attempt
        exit_code = $exitCode
        watchdog_reason = $watchdogReason
        updated_at_utc = $finished
    }
    Write-JsonStatus -Path (Join-Path $outputDirectory "queue_status.json") -Payload $statusPayload
    Write-JsonStatus -Path $supervisorStatusPath -Payload $statusPayload
    if ($attempt -lt 6) {
        Start-Sleep -Seconds ([Math]::Min(15 * $attempt, 60))
    }
}

$failurePayload = @{
    status = "failed_requires_attention"
    attempts = 6
    exit_code = $exitCode
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}
Write-JsonStatus -Path (Join-Path $outputDirectory "queue_status.json") -Payload $failurePayload
Write-JsonStatus -Path $supervisorStatusPath -Payload $failurePayload

exit $exitCode
