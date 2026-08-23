param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot
$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_4\phased_auto_residual_v5_frozen_model"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$metadataPath = Join-Path $outputDirectory "runner_metadata.json"
if (Test-Path -LiteralPath $metadataPath) {
    $priorMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    if ($null -ne $priorMetadata.runner_pid) {
        $priorProcess = Get-Process -Id ([int]$priorMetadata.runner_pid) -ErrorAction SilentlyContinue
        if ($null -ne $priorProcess) {
            throw "IARIC residual phased auto is already running as PID $($priorMetadata.runner_pid)"
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
    "--max-workers",
    "2",
    "--data-contract",
    "retained_local_research_snapshot"
)
$startedAtUtc = [DateTime]::UtcNow.ToString("o")
@{
    status = "launching"
    python_executable = $pythonExecutable
    max_workers = 2
    tradable_execution_symbols = 98
    data_contract = "retained_local_research_snapshot"
    data_authority = "project_official_local_snapshot"
    acquisition_receipts_required = $false
    broker_connection_required = $false
    locked_validation_accessed = $false
    holdout_accessed = $false
    started_at_utc = $startedAtUtc
} | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "background_status.json") -Encoding utf8
$runner = Start-Process -FilePath $pythonExecutable -ArgumentList $arguments `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru
@{
    runner_pid = $runner.Id
    python_executable = $pythonExecutable
    max_workers = 2
    tradable_execution_symbols = 98
    data_contract = "retained_local_research_snapshot"
    data_authority = "project_official_local_snapshot"
    acquisition_receipts_required = $false
    broker_connection_required = $false
    started_at_utc = $startedAtUtc
} | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath $metadataPath -Encoding utf8
$runner.Id
