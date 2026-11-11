#Requires -Version 5.1
param(
    [Parameter(Mandatory = $true)][string]$Project,
    [string]$Version = '*',
    [string[]]$Source = @(),
    [switch]$IncludeBuilder
)
$ErrorActionPreference = 'Stop'
$projectPath = (Resolve-Path -LiteralPath $Project).Path
if ([IO.Path]::GetExtension($projectPath) -ne '.csproj') { throw '-Project must name a .csproj file.' }
[xml]$metadata = Get-Content -LiteralPath $projectPath -Raw
$hasBuilder = $null -ne $metadata.SelectSingleNode("//*[local-name()='PackageReference' and (@Include='alanxinn.TensorRtOnnxBuilder.TensorRT10' or @Update='alanxinn.TensorRtOnnxBuilder.TensorRT10')]")

# Floating versions must be direct project references. A .nupkg dependency
# cannot force highest-version resolution on a consuming project.
$references = [ordered]@{ 'alanxinn.CvinferSharp.TensorRT10' = $Version }
if ($IncludeBuilder) { $references['alanxinn.TensorRtOnnxBuilder.TensorRT10'] = $Version }
$components = @('Core')
if ($IncludeBuilder -or $hasBuilder) {
    $components += @('SM86', 'SM89', 'PTX.Part1', 'PTX.Part2', 'PTX.Part3', 'SM120.Part1', 'SM120.Part2')
}
foreach ($component in $components) { $references["alanxinn.TensorRtRuntime.$component.TensorRT10"] = '*' }
foreach ($entry in $references.GetEnumerator()) {
    & dotnet add $projectPath package $entry.Key --version $entry.Value --no-restore
    if ($LASTEXITCODE -ne 0) { throw "Unable to add/update PackageReference: $($entry.Key)" }
}
$restoreArguments = @('restore', $projectPath, '--force-evaluate', '--no-cache')
foreach ($item in $Source) { $restoreArguments += @('--source', $item) }
& dotnet @restoreArguments
if ($LASTEXITCODE -ne 0) { throw 'NuGet restore failed; see the dependency-resolution errors above.' }
Write-Host 'Cvinfer references updated; runtime packages use independent latest-stable floating versions.'
