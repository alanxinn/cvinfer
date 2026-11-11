#Requires -Version 5.1
param(
    [string]$OutputDirectory = 'package/TensorRT10',
    [string]$Python = '',
    [string[]]$CudaArchitectures = @('86', '89', '120'),
    [switch]$AllArchitectures,
    [switch]$IncludeOptionalLibraries,
    [switch]$SkipNativeBuild,
    [switch]$SkipRuntimeArchive,
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$output = [IO.Path]::GetFullPath($OutputDirectory)
$logs = Join-Path $output 'logs'
$projectMetadata = [xml](Get-Content -LiteralPath (Join-Path $repo 'csharp/alanxinn.CvinferSharp/alanxinn.CvinferSharp.csproj') -Raw)
$version = @($projectMetadata.Project.PropertyGroup.Version | Where-Object { $_ })[0]
$runtimeMetadata = [xml](Get-Content -LiteralPath (Join-Path $repo 'csharp/TensorRtRuntime.props') -Raw)
$runtimeVersion = $runtimeMetadata.SelectSingleNode('/Project/PropertyGroup/TensorRtRuntimePackageVersion').InnerText
New-Item -ItemType Directory -Force -Path $logs | Out-Null
Push-Location $repo
try {
    if (@('86', '89', '120') | Where-Object { $_ -notin $CudaArchitectures }) {
        throw 'Public TensorRT10 NuGets must cover SM86, SM89 and SM120.'
    }
    if (-not $Python) {
        $cachedPython = Join-Path $repo 'cache/trt10-python/Scripts/python.exe'
        $Python = if (Test-Path -LiteralPath $cachedPython) { $cachedPython } else { (Get-Command python -ErrorAction Stop).Source }
    }
    if (-not $SkipNativeBuild) {
        $builderArchitectures = if ($AllArchitectures) { 'ALL' } else { $CudaArchitectures -join ';' }
        & "$PSScriptRoot/Build-Trt10.ps1" -CudaArchitectures ($CudaArchitectures -join ';') `
            -BuilderArchitectures $builderArchitectures -IncludeOptionalLibraries:$IncludeOptionalLibraries *> (Join-Path $logs 'native-build.log')
        if ($LASTEXITCODE -ne 0) { throw 'TensorRT 10 native build failed; see native-build.log' }
    }
    & $Python "$PSScriptRoot/pack_trt10_runtime_nugets.py" --output $output --version $runtimeVersion *> (Join-Path $logs 'runtime-nugets.log')
    if ($LASTEXITCODE -ne 0) { throw "Runtime dependency packing failed; see $logs/runtime-nugets.log" }
    foreach ($project in @('alanxinn.CvinferSharp', 'alanxinn.TensorRtOnnxBuilder')) {
        Write-Host "Packing $project.TensorRT10"
        & dotnet pack "csharp/$project/$project.csproj" -c Release -o $output `
            -p:BundleTensorRtRuntime=true "-p:RestoreAdditionalProjectSources=$output" `
            "-p:RestorePackagesPath=$repo/cache/trt10-main-packages/$version-runtime-$runtimeVersion" `
            -p:RestoreForceEvaluate=true *> (Join-Path $logs "$project-pack.log")
        if ($LASTEXITCODE -ne 0) { throw "NuGet pack failed: $project; see $logs" }
    }
    $profile = if ($AllArchitectures) { 'all-sm' } else { ($CudaArchitectures | ForEach-Object { "sm$_" }) -join '-' }
    if ($IncludeOptionalLibraries) { $profile += '-optional' }
    $archive = Join-Path $output "tensorrt10.16.1.11-cuda12.9-$profile-runtime-win-x64.zip"
    if (-not $SkipRuntimeArchive) {
        & 'csharp/alanxinn.TensorRtOnnxBuilder/tools/Pack-TensorRtRuntime.ps1' `
            -SourceDirectory 'build/trt10-release/builder_runtime' -OutputZip $archive `
            -CudaArchitectures $CudaArchitectures -AllArchitectures:$AllArchitectures `
            -IncludeOptionalLibraries:$IncludeOptionalLibraries *> (Join-Path $logs 'runtime-pack.log')
    }
    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw "Runtime ZIP missing: $archive" }
    # Do not list older full SDK archives left in this output directory as new artifacts.
    $packageNames = @("alanxinn.CvinferSharp.TensorRT10.$version.nupkg", "alanxinn.TensorRtOnnxBuilder.TensorRT10.$version.nupkg")
    foreach ($component in @('Core', 'SM86', 'SM89', 'PTX.Part1', 'PTX.Part2', 'PTX.Part3', 'SM120.Part1', 'SM120.Part2')) {
        $packageNames += "alanxinn.TensorRtRuntime.$component.TensorRT10.$runtimeVersion.nupkg"
    }
    $currentPackages = @($packageNames | ForEach-Object { Get-Item -LiteralPath (Join-Path $output $_) })
    if ($currentPackages | Where-Object { $_.Length -gt 250000000 }) { throw 'A NuGet package exceeds the 250MB publishing limit.' }
    $artifacts = @($currentPackages + @(Get-Item -LiteralPath $archive) | ForEach-Object {
        [ordered]@{ file = $_.Name; bytes = $_.Length; sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    [ordered]@{ packageVersion = $version; runtimePackageVersion = $runtimeVersion;
        tensorRT = '10.16.1.11'; cuda = '12.9'; optionalLibraries = [bool]$IncludeOptionalLibraries;
        architectures = $CudaArchitectures; platform = 'win-x64'; artifacts = $artifacts } |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'artifacts.json') -Encoding UTF8
    $localTestScript = Join-Path $PSScriptRoot 'check_trt10_nuget.py'
    if (-not $SkipTests -and -not (Test-Path -LiteralPath $localTestScript -PathType Leaf)) {
        Write-Host 'Local Python validation script is not present; model checks were skipped.'
    }
    if (-not $SkipTests -and (Test-Path -LiteralPath $localTestScript -PathType Leaf)) {
        if (-not $Python) {
            $cachedPython = Join-Path $repo 'cache/trt10-python/Scripts/python.exe'
            $Python = if (Test-Path -LiteralPath $cachedPython) { $cachedPython } else { (Get-Command python -ErrorAction Stop).Source }
        }
        & $Python $localTestScript --packages $output --version $version *> (Join-Path $logs 'nuget-tests.log')
        if ($LASTEXITCODE -ne 0) { throw "NuGet model tests failed; see $logs/nuget-tests.log and cache/nuget-trt10-validation" }
    }
    Write-Host "TensorRT10 NuGet/runtime artifacts: $output"
}
finally { Pop-Location }
