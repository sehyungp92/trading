param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot

$sourceDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v3_robust_breadth"
$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v5_selective_sector_overflow"
$frozenCandidate = Join-Path $sourceDirectory "frozen_selection_candidate.json"

if (-not (Test-Path -LiteralPath $frozenCandidate -PathType Leaf)) {
    throw "Missing frozen Phase-7 candidate: $frozenCandidate"
}
if (Test-Path -LiteralPath $outputDirectory) {
    throw "Phase-8 continuation output already exists; refusing to overwrite $outputDirectory"
}
New-Item -ItemType Directory -Path $outputDirectory | Out-Null

$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonCandidates = @()
if (Test-Path -LiteralPath $bundledPython) {
    $pythonCandidates += $bundledPython
}
$pythonCandidates += (Get-Command python -ErrorAction Stop).Source
$pythonExecutable = $null
foreach ($candidate in ($pythonCandidates | Select-Object -Unique)) {
    & $candidate -c "import pandas, pyarrow" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExecutable = $candidate
        break
    }
}
if ($null -eq $pythonExecutable) {
    throw "No Python interpreter with pandas and pyarrow is available"
}

$stdoutPath = Join-Path $outputDirectory "runner_stdout.log"
$stderrPath = Join-Path $outputDirectory "runner_stderr.log"
$arguments = @(
    "-m",
    "backtests.stock.auto.runners.run_iaric_residual_phase8_continuation",
    "--output-dir",
    $outputDirectory,
    "--source-output",
    $sourceDirectory,
    "--max-workers",
    "2"
)
$startedAtUtc = [DateTime]::UtcNow.ToString("o")
$metadata = @{
    status = "launching_phase_8_continuation"
    round = 2
    experiment_contract = "iaric_round2_phase8_selective_sector_overflow_continuation_v2"
    source_output = $sourceDirectory
    frozen_candidate = $frozenCandidate
    python_executable = $pythonExecutable
    max_workers = 2
    starting_phase = 8
    phases_1_through_7_rerun = $false
    registered_phase_count = 20
    continuation_phase_count = 9
    executable_score_component_ceiling = 7
    optimizer_score_components = 7
    data_contract = "retained_local_research_snapshot"
    selection_end = "2025-07-31"
    locked_validation_end = "2026-03-01"
    holdout_start = "2026-03-02"
    locked_validation_accessed = $false
    holdout_accessed = $false
    started_at_utc = $startedAtUtc
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "background_status.json") -Encoding utf8

$runner = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru

$runnerMetadata = @{
    runner_pid = $runner.Id
    round = 2
    experiment_contract = $metadata.experiment_contract
    source_output = $sourceDirectory
    frozen_candidate = $frozenCandidate
    python_executable = $pythonExecutable
    max_workers = 2
    starting_phase = 8
    phases_1_through_7_rerun = $false
    registered_phase_count = 20
    continuation_phase_count = 9
    executable_score_component_ceiling = 7
    optimizer_score_components = 7
    data_contract = $metadata.data_contract
    locked_validation_accessed = $false
    holdout_accessed = $false
    started_at_utc = $startedAtUtc
}
$runnerMetadata | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "runner_metadata.json") -Encoding utf8
$runner.Id
