param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [ValidateRange(8, 14)]
    [int]$ResumeAfterPhase = 8
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot

$sourceDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v3_robust_breadth"
$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_2\phased_auto_alpha_v5_selective_sector_overflow"
$artifactNames = @{
    8 = "phase_8_selective_sector_overflow_and_displacement_quality.json"
    9 = "phase_9_quality_aperture_and_discrimination.json"
    10 = "phase_10_risk_and_notional_frontier.json"
    11 = "phase_11_exit_capture_frontier.json"
    12 = "phase_12_final_alpha_frequency_synergy.json"
    13 = "phase_13_path_causal_profit_retention.json"
    14 = "phase_14_capacity_neutral_alpha_recycling.json"
}
$completedArtifact = Join-Path $outputDirectory $artifactNames[$ResumeAfterPhase]
$nextArtifact = if ($artifactNames.ContainsKey($ResumeAfterPhase + 1)) {
    Join-Path $outputDirectory $artifactNames[$ResumeAfterPhase + 1]
} else {
    Join-Path $outputDirectory "phase_15_final_robustness_and_target_assessment.json"
}

if (-not (Test-Path -LiteralPath $completedArtifact -PathType Leaf)) {
    throw "Cannot resume without completed Phase ${ResumeAfterPhase}: $completedArtifact"
}
if (Test-Path -LiteralPath $nextArtifact) {
    throw "The next phase already exists; refusing an ambiguous resume: $nextArtifact"
}

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

$resumeIndex = 1
while (Test-Path -LiteralPath (Join-Path $outputDirectory "runner_resume_${resumeIndex}_stdout.log")) {
    $resumeIndex += 1
}
$stdoutPath = Join-Path $outputDirectory "runner_resume_${resumeIndex}_stdout.log"
$stderrPath = Join-Path $outputDirectory "runner_resume_${resumeIndex}_stderr.log"
$arguments = @(
    "-m",
    "backtests.stock.auto.runners.run_iaric_residual_phase8_continuation",
    "--output-dir",
    $outputDirectory,
    "--source-output",
    $sourceDirectory,
    "--max-workers",
    "2",
    "--resume-after-phase",
    "$ResumeAfterPhase"
)
$startedAtUtc = [DateTime]::UtcNow.ToString("o")
$status = @{
    status = "launching_phase_$($ResumeAfterPhase + 1)_resume"
    round = 2
    experiment_contract = "iaric_round2_audited_frontier_resume_v2"
    source_output = $sourceDirectory
    output_directory = $outputDirectory
    completed_phase = $ResumeAfterPhase
    completed_phase_artifact = $completedArtifact
    completed_phases_rerun = $false
    python_executable = $pythonExecutable
    max_workers = 2
    starting_phase = $ResumeAfterPhase + 1
    registered_phase_count = 20
    continuation_phases_remaining = 16 - $ResumeAfterPhase
    executable_score_component_ceiling = 7
    locked_validation_accessed = $false
    holdout_accessed = $false
    memory_bounding_enabled = $true
    started_at_utc = $startedAtUtc
}
$status | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "background_status.json") -Encoding utf8

$runner = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru

$metadata = @{
    runner_pid = $runner.Id
    resume_index = $resumeIndex
    resume_contract = $status.experiment_contract
    source_output = $sourceDirectory
    output_directory = $outputDirectory
    completed_phase = $ResumeAfterPhase
    completed_phases_rerun = $false
    max_workers = 2
    starting_phase = $ResumeAfterPhase + 1
    memory_bounding_enabled = $true
    locked_validation_accessed = $false
    holdout_accessed = $false
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
    started_at_utc = $startedAtUtc
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "runner_resume_${resumeIndex}_metadata.json") -Encoding utf8
$runner.Id
