[CmdletBinding()]
param(
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$ExpectedAccount = 'DESKTOP-CMOIPGE\lab'
$Repository = 'C:\Users\lab\Counterfactual-XAI-Verifier'
$Worktree = Join-Path $Repository '.k'
$Preparation = Join-Path $Worktree 'final_kci_direct_yolo_shap_v1\independent_shap_variable_validation_v1_preparation'
$Output = Join-Path $Worktree 'final_kci_direct_yolo_shap_v1\independent_shap_variable_validation_run_001_lab'
$Python = Join-Path $Repository '.venv\Scripts\python.exe'
$Supervisor = Join-Path $Preparation 'run_independent_validation_managed.py'

$ActualAccount = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
if (-not $ActualAccount.Equals($ExpectedAccount, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Interactive account gate failed: expected $ExpectedAccount, actual $ActualAccount"
}

if ([Environment]::GetEnvironmentVariable('MATLAB_PREFDIR', 'Process')) {
    throw 'MATLAB_PREFDIR is defined in this process; the approved default MATLAB preferences would not be inherited.'
}

foreach ($RequiredPath in @($Python, $Supervisor, $Worktree, $Preparation)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required path is missing: $RequiredPath"
    }
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath $Output -PathType Container)) {
        throw "Resume was requested but the locked output folder does not exist: $Output"
    }
}
elseif (Test-Path -LiteralPath $Output) {
    throw "The locked result folder already exists; no overwrite or renumbering is allowed: $Output"
}

$ExistingMatlab = @(Get-CimInstance Win32_Process -Filter "Name='MATLAB.exe'" -ErrorAction Stop)
if ($ExistingMatlab.Count -gt 0) {
    $ProcessSummary = ($ExistingMatlab | ForEach-Object { "PID=$($_.ProcessId) Created=$($_.CreationDate)" }) -join '; '
    throw "MATLAB residual-process gate failed. No process was terminated. $ProcessSummary"
}

$Arguments = @($Supervisor)
if ($Resume) {
    $Arguments += '--resume'
}

Push-Location -LiteralPath $Worktree
try {
    & $Python @Arguments
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($ExitCode -ne 0) {
    throw "Independent-validation supervisor returned exit code $ExitCode"
}

Write-Host '[complete] direct exact-Shapley independent variable-selection validation finished; inspect the locked gate result before any later work.'
