#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [switch]$WithCuda,
    [switch]$SkipDownload,
    [switch]$ReuseEngines
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command $Python -ErrorAction Stop).Source
Push-Location $repo
try {
    & "$PSScriptRoot/Build-Trt10.ps1"
    if (-not $SkipDownload) {
        & $pythonExe "$PSScriptRoot/prepare_trt10_models.py"
        if ($LASTEXITCODE -ne 0) { throw 'Model preparation failed' }
    }
    $reuse = if ($ReuseEngines) { @('--reuse-engines') } else { @() }
    & $pythonExe "$PSScriptRoot/test_trt10_contract.py"
    if ($LASTEXITCODE -ne 0) { throw 'TensorRT contract tests failed' }
    & $pythonExe "$PSScriptRoot/validate_trt10.py" @reuse
    if ($LASTEXITCODE -ne 0) { throw 'Model inference validation failed; see cache/trt10-validation' }
    if ($WithCuda) {
        & "$PSScriptRoot/Build-Trt10.ps1" -Preset trt10-cuda-release
        & $pythonExe "$PSScriptRoot/validate_trt10.py" --build build/trt10-cuda-release --compute cuda --reuse-engines
        if ($LASTEXITCODE -ne 0) { throw 'CUDA preprocessing/postprocessing validation failed' }
    }
    & $pythonExe "$PSScriptRoot/compare_trt10_outputs.py"
    if ($LASTEXITCODE -ne 0) { throw 'Numerical parity checks failed' }
}
finally { Pop-Location }
