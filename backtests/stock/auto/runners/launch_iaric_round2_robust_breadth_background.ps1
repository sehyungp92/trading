param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot

$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v4_post_phase7_alpha_dd_frontier"
$cacheSourceDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v3_robust_breadth\cache"
$baselineConfig = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v3_robust_breadth\frozen_selection_candidate.json"

if (-not (Test-Path -LiteralPath $baselineConfig -PathType Leaf)) {
    throw "Missing latest frozen optimized baseline: $baselineConfig"
}
if (-not (Test-Path -LiteralPath $cacheSourceDirectory -PathType Container)) {
    throw "Missing source-fingerprinted Phase 1 atlas cache: $cacheSourceDirectory"
}

$metadataPath = Join-Path $outputDirectory "runner_metadata.json"
if (Test-Path -LiteralPath $metadataPath) {
    $priorMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($null -ne $priorMetadata.runner_pid) {
        $priorProcess = Get-Process -Id ([int]$priorMetadata.runner_pid) -ErrorAction SilentlyContinue
        if ($null -ne $priorProcess) {
            throw "Corrected IARIC Round 2 phased auto is already running as PID $($priorMetadata.runner_pid)"
        }
    }
    throw "Corrected IARIC Round 2 output already exists; refusing to overwrite $outputDirectory"
}

New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$outputCacheDirectory = Join-Path $outputDirectory "cache"
New-Item -ItemType Directory -Path $outputCacheDirectory -Force | Out-Null
Get-ChildItem -LiteralPath $cacheSourceDirectory -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $outputCacheDirectory
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
$metadata = @{
    status = "launching"
    round = 2
    experiment_contract = "iaric_round2_post_phase7_alpha_dd_aspirational_18_step_v6"
    breadth_contract = "positive_one_sided_95pct_winsorized_r_positive_median_top5_gross_positive_share_lte_50pct"
    python_executable = $pythonExecutable
    baseline_config = $baselineConfig
    max_workers = 2
    phase_count = 18
    executable_score_component_ceiling = 7
    optimizer_score_components = 7
    optimizer_score_contract = "iaric_round2_non_saturated_exact_v2"
    data_contract = "retained_local_research_snapshot"
    selection_end = "2025-07-31"
    locked_validation_end = "2026-03-01"
    holdout_start = "2026-03-02"
    reused_source_fingerprinted_atlas_cache = $true
    atlas_cache_source = $cacheSourceDirectory
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
    breadth_contract = $metadata.breadth_contract
    python_executable = $pythonExecutable
    baseline_config = $baselineConfig
    max_workers = 2
    phase_count = 18
    executable_score_component_ceiling = 7
    optimizer_score_components = 7
    optimizer_score_contract = $metadata.optimizer_score_contract
    data_contract = $metadata.data_contract
    reused_source_fingerprinted_atlas_cache = $true
    atlas_cache_source = $cacheSourceDirectory
    locked_validation_accessed = $false
    holdout_accessed = $false
    started_at_utc = $startedAtUtc
}
$runnerMetadata | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath $metadataPath -Encoding utf8
$runner.Id
