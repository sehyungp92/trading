param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [int]$OldRunnerProcessId
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot

$escapeDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\escape_round"
$phaseOnePath = Join-Path $escapeDirectory "phase_1_composition_aperture_results.json"
$cachePath = Join-Path $escapeDirectory "evaluation_cache.json"
$stdoutPath = Join-Path $escapeDirectory "background_stdout.log"
$stderrPath = Join-Path $escapeDirectory "background_stderr.log"
$handoffStarted = [DateTime]::UtcNow

function Get-DescendantProcessIds([int]$ParentId) {
    $all = @(Get-CimInstance Win32_Process)
    $result = [System.Collections.Generic.List[int]]::new()
    $frontier = [System.Collections.Generic.List[int]]::new()
    $frontier.Add($ParentId)
    while ($frontier.Count -gt 0) {
        $next = [System.Collections.Generic.List[int]]::new()
        foreach ($parent in $frontier) {
            foreach ($child in @($all | Where-Object ParentProcessId -eq $parent)) {
                $result.Add([int]$child.ProcessId)
                $next.Add([int]$child.ProcessId)
            }
        }
        $frontier = $next
    }
    return @($result)
}

while ($null -ne (Get-Process -Id $OldRunnerProcessId -ErrorAction SilentlyContinue)) {
    if (Test-Path -LiteralPath $phaseOnePath) {
        $phaseOne = Get-Item -LiteralPath $phaseOnePath
        if ($phaseOne.LastWriteTimeUtc -ge $handoffStarted) {
            break
        }
    }
    Start-Sleep -Seconds 5
}

# Phase 1 may have committed immediately before the parent queues Phase 2.
# Stop that superseded branch, including its exact descendants, before the
# repaired branch starts so the two-worker ceiling is never exceeded.
if ($null -ne (Get-Process -Id $OldRunnerProcessId -ErrorAction SilentlyContinue)) {
    $descendants = @(Get-DescendantProcessIds -ParentId $OldRunnerProcessId)
    foreach ($processId in @($descendants | Sort-Object -Descending)) {
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $OldRunnerProcessId -Force -ErrorAction SilentlyContinue
}

$handoffTime = [DateTime]::UtcNow.ToString("o")
"[$handoffTime] Phase 1 cached; handing off to baseline-relative gate repair." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

& python -m backtests.stock.auto.runners.migrate_iaric_escape_orchestration_cache `
    --cache $cachePath `
    1>> $stdoutPath 2>> $stderrPath
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation-cache migration failed with exit code $LASTEXITCODE"
}

$resumeTime = [DateTime]::UtcNow.ToString("o")
"[$resumeTime] Cache migrated; resuming corrected full phased escape search." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

& python -m backtests.stock.auto.runners.run_iaric_escape_round3 `
    --start-date 2024-03-25 `
    --end-date 2026-03-01 `
    --max-workers 2 `
    --output-dir $escapeDirectory `
    1>> $stdoutPath 2>> $stderrPath

$exitCode = $LASTEXITCODE
$finished = [DateTime]::UtcNow.ToString("o")
"[$finished] Corrected full phased escape search exited with code $exitCode." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
exit $exitCode
