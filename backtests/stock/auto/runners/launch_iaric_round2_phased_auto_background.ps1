param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot
$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v1"
$baselineConfig = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_1\optimized_config.json"
if (-not (Test-Path -LiteralPath $baselineConfig -PathType Leaf)) {
    throw "Missing frozen Round 1 baseline: $baselineConfig"
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$metadataPath = Join-Path $outputDirectory "runner_metadata.json"
if (Test-Path -LiteralPath $metadataPath) {
    $priorMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($null -ne $priorMetadata.runner_pid) {
        $priorProcess = Get-Process -Id ([int]$priorMetadata.runner_pid) -ErrorAction SilentlyContinue
        if ($null -ne $priorProcess) {
            throw "IARIC Round 2 phased auto is already running as PID $($priorMetadata.runner_pid)"
        }
    }
}
$stdoutPath = Join-Path $outputDirectory "runner_stdout.log"
$stderrPath = Join-Path $outputDirectory "runner_stderr.log"

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

$arguments = @(
    "-m",
    "backtests.stock.auto.runners.run_iaric_residual_phased_auto",
    "--output-dir",
    $outputDirectory,
    "--baseline-config",
    $baselineConfig,
    "--max-workers",
    "2",
    "--data-contract",
    "retained_local_research_snapshot",
    "--skip-protected-integration"
)
$startedAtUtc = [DateTime]::UtcNow.ToString("o")
@{
    status = "launching"
    round = 2
    experiment_contract = "iaric_round2_non_saturated_exact_v1"
    python_executable = $pythonExecutable
    baseline_config = $baselineConfig
    max_workers = 2
    executable_score_component_ceiling = 7
    optimizer_score_components = 7
    data_contract = "retained_local_research_snapshot"
    selection_end = "2025-07-31"
    locked_validation_end = "2026-03-01"
    holdout_start = "2026-03-02"
    holdout_accessed = $false
    started_at_utc = $startedAtUtc
} | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "background_status.json") -Encoding utf8
$runner = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru
@{
    runner_pid = $runner.Id
    round = 2
    python_executable = $pythonExecutable
    baseline_config = $baselineConfig
    max_workers = 2
    executable_score_component_ceiling = 7
    optimizer_score_components = 7
    data_contract = "retained_local_research_snapshot"
    holdout_accessed = $false
    started_at_utc = $startedAtUtc
} | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath $metadataPath -Encoding utf8
$runner.Id
