param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [Parameter(Mandatory = $true)]
    [int]$WaitForPid
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $RepositoryRoot

$escapeDirectory = Join-Path $RepositoryRoot "backtests\output\stock\iaric\round_3\escape_round"
$stdoutPath = Join-Path $escapeDirectory "course_continuation_stdout.log"
$stderrPath = Join-Path $escapeDirectory "course_continuation_stderr.log"

for ($attempt = 1; $attempt -le 3; $attempt++) {
    $started = [DateTime]::UtcNow.ToString("o")
    "[$started] Starting IARIC escape course continuation supervisor attempt $attempt." |
        Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append

    $pidArgument = if ($attempt -eq 1) { $WaitForPid } else { 0 }
    & python -m backtests.stock.auto.runners.run_iaric_escape_course_continuation `
        --output-dir $escapeDirectory `
        --wait-for-pid $pidArgument `
        --max-workers 2 `
        --max-restarts 3 `
        1>> $stdoutPath 2>> $stderrPath

    $exitCode = $LASTEXITCODE
    $finished = [DateTime]::UtcNow.ToString("o")
    "[$finished] Course continuation attempt $attempt exited with code $exitCode." |
        Out-File -LiteralPath $stdoutPath -Encoding utf8 -Append
    if ($exitCode -eq 0) {
        exit 0
    }
    if ($attempt -lt 3) {
        Start-Sleep -Seconds (15 * $attempt)
    }
}

exit $exitCode
