param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot

$escapeDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\escape_round"
$stdoutPath = Join-Path $escapeDirectory "background_stdout.log"
$stderrPath = Join-Path $escapeDirectory "background_stderr.log"
$started = [DateTime]::UtcNow.ToString("o")
"[$started] Resuming metadata- and drawdown-gate-corrected escape search from migrated cache." | Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

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
