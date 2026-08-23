param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [string]$AuthorityManifest = ""
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot

$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $bundledPython) {
    $pythonExecutable = $bundledPython
} elseif ($pythonCommand) {
    $pythonExecutable = $pythonCommand.Source
} else {
    throw "No usable Python runtime was found for the IARIC supervisor"
}

$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_4\phased_auto_price_volume_v1"
$phaseZeroDirectory = Join-Path $outputDirectory "phase_0_price_data_integrity_and_parity"
$preflightSummary = Join-Path $phaseZeroDirectory "atlas_summary.json"
$statusPath = Join-Path $outputDirectory "background_status.json"
$preflightStdout = Join-Path $outputDirectory "preflight_stdout.log"
$preflightStderr = Join-Path $outputDirectory "preflight_stderr.log"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

function Write-JsonStatus {
    param([hashtable]$Payload)
    $temporary = "$statusPath.tmp"
    $Payload | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

$preflightArguments = @(
    "-m",
    "backtests.stock.auto.runners.run_iaric_representative_preflight",
    "--output-dir",
    $phaseZeroDirectory
)
if ($AuthorityManifest) {
    $resolvedManifest = (Resolve-Path -LiteralPath $AuthorityManifest).Path
    $preflightArguments += @("--authority-manifest", $resolvedManifest)
}

Write-JsonStatus -Payload @{
    status = "running_phase_0_price_data_integrity_and_parity"
    supervisor_pid = $PID
    max_workers = 2
    holdout_accessed = $false
    obsolete_price_volume_optimizer_allowed = $false
    started_at_utc = [DateTime]::UtcNow.ToString("o")
}

$preflight = Start-Process -FilePath $pythonExecutable -ArgumentList $preflightArguments `
    -WindowStyle Hidden -RedirectStandardOutput $preflightStdout `
    -RedirectStandardError $preflightStderr -PassThru
$null = $preflight.Handle
$preflight.WaitForExit()
$preflight.Refresh()

if ($preflight.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $preflightSummary)) {
    Write-JsonStatus -Payload @{
        status = "failed_phase_0_price_data_preflight"
        supervisor_pid = $PID
        runner_pid = $preflight.Id
        exit_code = $preflight.ExitCode
        max_workers = 2
        holdout_accessed = $false
        obsolete_price_volume_optimizer_allowed = $false
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    exit 1
}

$preflightPayload = Get-Content -Raw -LiteralPath $preflightSummary | ConvertFrom-Json
if (-not [bool]$preflightPayload.representative_reversion_baseline_eligible) {
    Write-JsonStatus -Payload @{
        status = "blocked_missing_authoritative_price_volume_inputs"
        supervisor_pid = $PID
        runner_pid = $preflight.Id
        max_workers = 2
        input_authority = $preflightPayload.input_authority
        input_authority_attestation = $preflightPayload.input_authority_attestation
        ready_reversion_sleeves = @($preflightPayload.ready_reversion_sleeves)
        disabled_sleeves = @($preflightPayload.disabled_sleeves)
        blockers = @($preflightPayload.representative_reversion_baseline_blockers)
        holdout_accessed = $false
        obsolete_price_volume_optimizer_allowed = $false
        optimizer_started = $false
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
    }
    exit 2
}

# Phase 0 only proves price/volume authority. The broad daily residual atlas
# must next qualify the core sleeve before optimization. Secondary five-minute
# sleeves are optional and news/quotes are never prerequisites.
Write-JsonStatus -Payload @{
    status = "blocked_pending_mechanism_pure_opportunity_atlas"
    supervisor_pid = $PID
    runner_pid = $preflight.Id
    max_workers = 2
    ready_reversion_sleeves = @($preflightPayload.ready_reversion_sleeves)
    blockers = @(
        "price/volume authority passed, but the causal residual Phase 1 atlas and candidate registry have not yet been produced from the certified adapters"
    )
    required_next_phase = "phase_1_residual_opportunity_atlas"
    holdout_accessed = $false
    obsolete_price_volume_optimizer_allowed = $false
    optimizer_started = $false
    updated_at_utc = [DateTime]::UtcNow.ToString("o")
}
exit 2
