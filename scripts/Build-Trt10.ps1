#Requires -Version 5.1
param(
    [ValidateSet('trt10-release', 'trt10-cuda-release')]
    [string]$Preset = 'trt10-release',
    [int]$Jobs = 4,
    [string]$CudaArchitectures = '89',
    [string]$BuilderArchitectures = '',
    [switch]$IncludeOptionalLibraries
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio C++ Build Tools were not found.' }
& (Join-Path $vs 'Common7/Tools/Launch-VsDevShell.ps1') -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$env:PATH = (Join-Path $vs 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja') + ';' + $env:PATH
Push-Location $repo
try {
    if (-not $BuilderArchitectures) { $BuilderArchitectures = $CudaArchitectures }
    $optionalLibraries = if ($IncludeOptionalLibraries) { 'ON' } else { 'OFF' }
    & cmake --preset $Preset "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures" `
        "-DCVINFER_TRT_BUILDER_ARCHITECTURES=$BuilderArchitectures" "-DCVINFER_TRT_INCLUDE_OPTIONAL_LIBRARIES=$optionalLibraries"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }
    & cmake --build --preset "build-$Preset" --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed: $LASTEXITCODE" }
}
finally { Pop-Location }
