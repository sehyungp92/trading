param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot

$escapeDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\escape_round"
$preflightDirectory = Join-Path $escapeDirectory "preflight"
$stdoutPath = Join-Path $escapeDirectory "background_stdout.log"
$stderrPath = Join-Path $escapeDirectory "background_stderr.log"

New-Item -ItemType Directory -Path $escapeDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $preflightDirectory -Force | Out-Null

$started = (Get-Date).ToUniversalTime().ToString("o")
"[$started] Starting corrected latest-fold anchor-isolation preflight." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

& python -m backtests.stock.auto.runners.run_iaric_escape_round3 `
    --smoke `
    --start-date 2025-08-01 `
    --end-date 2026-03-01 `
    --max-workers 2 `
    --output-dir $preflightDirectory `
    1>> $stdoutPath 2>> $stderrPath

if ($LASTEXITCODE -ne 0) {
    $failed = (Get-Date).ToUniversalTime().ToString("o")
    "[$failed] Preflight failed (exit $LASTEXITCODE); full search was not started." | Out-File -LiteralPath $stderrPath -Encoding utf8 -Append
    exit $LASTEXITCODE
}

$preflightPassed = (Get-Date).ToUniversalTime().ToString("o")
"[$preflightPassed] Preflight passed; starting full phased escape search." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

& python -m backtests.stock.auto.runners.run_iaric_escape_round3 `
    --start-date 2024-03-25 `
    --end-date 2026-03-01 `
    --max-workers 2 `
    --output-dir $escapeDirectory `
    1>> $stdoutPath 2>> $stderrPath

$exitCode = $LASTEXITCODE
$finished = (Get-Date).ToUniversalTime().ToString("o")
"[$finished] Full phased escape search exited with code $exitCode." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
exit $exitCode
