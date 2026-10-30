#Requires -Version 5.1
param(
    [string]$OutputDirectory = 'package/TensorRT10',
    [string]$Python = '',
    [switch]$SkipNativeBuild,
    [switch]$SkipRuntimeArchive,
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$output = [IO.Path]::GetFullPath($OutputDirectory)
$logs = Join-Path $output 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Push-Location $repo
try {
    if (-not $SkipNativeBuild) {
        & "$PSScriptRoot/Build-Trt10.ps1" *> (Join-Path $logs 'native-build.log')
        if ($LASTEXITCODE -ne 0) { throw 'TensorRT 10 native build failed; see native-build.log' }
    }
    foreach ($project in @('alanxinn.CvinferSharp', 'alanxinn.TensorRtOnnxBuilder')) {
        Write-Host "Packing $project.TensorRT10"
        & dotnet pack "csharp/$project/$project.csproj" -c Release -o $output *> (Join-Path $logs "$project-pack.log")
        if ($LASTEXITCODE -ne 0) { throw "NuGet pack failed: $project; see $logs" }
    }
    $archive = Join-Path $output 'tensorrt10.16.1.11-cuda12.9-cudnn9.27-runtime-win-x64.zip'
    if (-not $SkipRuntimeArchive) {
        & 'csharp/alanxinn.TensorRtOnnxBuilder/tools/Pack-TensorRtRuntime.ps1' `
            -SourceDirectory 'build/trt10-release/builder_runtime' -OutputZip $archive *> (Join-Path $logs 'runtime-pack.log')
    }
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw "Runtime ZIP missing: $archive" }
    $artifacts = @(Get-ChildItem -LiteralPath $output -File | Where-Object { $_.Extension -in @('.nupkg', '.zip') } | ForEach-Object {
        [ordered]@{ file = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    [ordered]@{ tensorRT = '10.16.1.11'; cuda = '12.9'; cudnn = '9.27'; platform = 'win-x64'; artifacts = $artifacts } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'artifacts.json') -Encoding UTF8
    if (-not $SkipTests) {
        if (-not $Python) {
            $cachedPython = Join-Path $repo 'cache/trt10-python/Scripts/python.exe'
            $Python = if (Test-Path -LiteralPath $cachedPython) { $cachedPython } else { (Get-Command python -ErrorAction Stop).Source }
        }
        & $Python "$PSScriptRoot/test_trt10_nuget.py" --packages $output *> (Join-Path $logs 'nuget-tests.log')
        if ($LASTEXITCODE -ne 0) { throw "NuGet model tests failed; see $logs/nuget-tests.log and cache/nuget-trt10-validation" }
    }
    Write-Host "TensorRT10 NuGet/runtime artifacts: $output"
}
finally { Pop-Location }
