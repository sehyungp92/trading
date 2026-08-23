param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot
$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_4\daily_residual_discovery_v3"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$stdoutPath = Join-Path $outputDirectory "runner_stdout.log"
$stderrPath = Join-Path $outputDirectory "runner_stderr.log"
$pythonCommand = Get-Command python -ErrorAction Stop
$arguments = @(
    "-m",
    "backtests.stock.auto.runners.run_iaric_daily_residual_discovery",
    "--output-dir",
    $outputDirectory,
    "--max-workers",
    "2"
)
$runner = Start-Process -FilePath $pythonCommand.Source -ArgumentList $arguments `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru
@{
    status = "starting_non_promotable_residual_discovery"
    runner_pid = $runner.Id
    max_workers = 2
    optimizer_class = "non_promotable_residual_discovery"
    representative_reversion_baseline_eligible = $false
    promotion_eligible = $false
    locked_validation_accessed = $false
    holdout_accessed = $false
    output_directory = $outputDirectory
    started_at_utc = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json -Depth 5 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "background_status.json") -Encoding utf8
$runner.Id
