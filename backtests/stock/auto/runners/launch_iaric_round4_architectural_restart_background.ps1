param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [string]$AuthorityManifest = ""
)

$ErrorActionPreference = "Stop"
$resolvedRepositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
Set-Location -LiteralPath $resolvedRepositoryRoot
$outputDirectory = Join-Path $resolvedRepositoryRoot "backtests\output\stock\iaric\round_4\phased_auto_price_volume_v1"
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$stdoutPath = Join-Path $outputDirectory "architectural_supervisor_stdout.log"
$stderrPath = Join-Path $outputDirectory "architectural_supervisor_stderr.log"
$supervisorPath = Join-Path $resolvedRepositoryRoot "backtests\stock\auto\runners\run_iaric_round4_architectural_restart_supervisor.ps1"
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $supervisorPath,
    "-RepositoryRoot",
    $resolvedRepositoryRoot
)
if ($AuthorityManifest) {
    $resolvedManifest = (Resolve-Path -LiteralPath $AuthorityManifest).Path
    $arguments += @("-AuthorityManifest", $resolvedManifest)
}
$runner = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments `
    -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath -PassThru
@{
    status = "starting_price_volume_v1_preflight"
    supervisor_pid = $runner.Id
    max_workers = 2
    holdout_accessed = $false
    optimizer_started = $false
    obsolete_price_volume_optimizer_allowed = $false
    output_directory = $outputDirectory
    started_at_utc = [DateTime]::UtcNow.ToString("o")
} | ConvertTo-Json -Depth 4 | Set-Content `
    -LiteralPath (Join-Path $outputDirectory "background_status.json") -Encoding utf8
$runner.Id
