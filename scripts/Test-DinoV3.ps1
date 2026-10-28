#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$Python,
    [switch]$Synthetic,
    [ValidateSet('fp32', 'fp16')][string]$Precision = 'fp32',
    [string]$Checkpoint = 'facebook/dinov3-vits16-pretrain-lvd1689m',
    [string]$Image,
    [int]$ImageSize = 224
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$pythonExe = (Get-Command $Python -ErrorAction Stop).Source
if (-not $Synthetic -and -not $Image) { throw 'Provide -Image for pretrained validation, or use -Synthetic for operator tests.' }
Push-Location $repo
try {
    & "$PSScriptRoot/Build-Trt10.ps1"
    if ($Synthetic) {
        & $pythonExe "$PSScriptRoot/test_dinov3_export.py"
        if ($LASTEXITCODE -ne 0) { throw 'DINOv3 operator export failed' }
        $engineDir = if ($Precision -eq 'fp32') { 'engine' } else { 'engine-fp16' }
        foreach ($kind in @('vit', 'convnext')) {
            & $pythonExe "$PSScriptRoot/build_dinov3_engine.py" --onnx "cache/dinov3-tiny/$kind/onnx" `
                --output "cache/dinov3-tiny/$kind/$engineDir" --precision $Precision
            if ($LASTEXITCODE -ne 0) { throw "DINOv3 $kind engine build failed" }
        }
        & $pythonExe "$PSScriptRoot/test_dinov3_native.py" --precision $Precision
        if ($LASTEXITCODE -ne 0) { throw 'DINOv3 synthetic native validation failed' }
    }
    else {
        & $pythonExe "$PSScriptRoot/export_dinov3.py" --checkpoint $Checkpoint --image-size $ImageSize
        if ($LASTEXITCODE -ne 0) { throw 'DINOv3 pretrained export failed; check checkpoint access and export logs' }
        & $pythonExe "$PSScriptRoot/build_dinov3_engine.py" --precision $Precision
        if ($LASTEXITCODE -ne 0) { throw 'DINOv3 engine build failed' }
        & $pythonExe "$PSScriptRoot/validate_dinov3_real.py" --image $Image
        if ($LASTEXITCODE -ne 0) { throw 'DINOv3 pretrained validation failed' }
    }
}
finally { Pop-Location }
