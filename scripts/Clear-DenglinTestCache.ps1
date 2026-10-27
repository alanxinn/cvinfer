#Requires -Version 5.1
[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent)).TrimEnd('\')
$runtimeCache = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'alanxinn\DenglinOnnxBuilder'))
$packageCache = [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.nuget\packages\alanxinn.denglinonnxbuilder'))
$temporaryRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\')
$targets = @((Join-Path $workspaceRoot 'cache'), (Join-Path $workspaceRoot 'build'), $runtimeCache, $packageCache)

foreach ($project in Get-ChildItem -LiteralPath (Join-Path $workspaceRoot 'csharp') -Directory) {
    foreach ($name in @('bin', 'obj')) {
        $targets += Join-Path $project.FullName $name
    }
}
$targets += Join-Path $workspaceRoot 'cvinfer_private\denglin_backend\tests\__pycache__'
$targets += @(Get-ChildItem -LiteralPath $temporaryRoot -Directory -Filter 'denglin-runtime-*' |
    Where-Object { $_.Name -match '^denglin-runtime-[a-fA-F0-9]{32}$' } |
    ForEach-Object { $_.FullName })
$targets = @($targets | Sort-Object -Unique | Where-Object { Test-Path -LiteralPath $_ })

# Validate every absolute target before any recursive removal.
foreach ($target in $targets) {
    $absolute = [IO.Path]::GetFullPath($target).TrimEnd('\')
    $workspaceTarget = $absolute.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)
    $knownCache = $absolute -eq $runtimeCache -or $absolute -eq $packageCache
    $knownStaging = (Split-Path -Parent $absolute) -eq $temporaryRoot -and
        (Split-Path -Leaf $absolute) -match '^denglin-runtime-[a-fA-F0-9]{32}$'
    if (-not ($workspaceTarget -or $knownCache -or $knownStaging)) {
        throw "Refusing unchecked target: $absolute"
    }
    Write-Host "Checked: $absolute"
}

# Keep upload ZIPs under artifacts intact, and verify their hashes afterward.
$zipHashes = @{}
$artifactRoot = Join-Path $workspaceRoot 'artifacts'
if (Test-Path -LiteralPath $artifactRoot) {
    foreach ($zip in Get-ChildItem -LiteralPath $artifactRoot -File -Filter '*.zip') {
        $zipHashes[$zip.FullName] = (Get-FileHash -LiteralPath $zip.FullName -Algorithm SHA256).Hash
    }
}

function Remove-CacheLinks([string]$Directory) {
    # Enumerate one level at a time, never traversing junctions into an installed SDK.
    foreach ($entry in Get-ChildItem -LiteralPath $Directory -Force) {
        if ($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            if ($entry.PSIsContainer) {
                # Nonrecursive Directory.Delete removes the link itself on Windows.
                [IO.Directory]::Delete($entry.FullName)
            } else {
                Remove-Item -LiteralPath $entry.FullName -Force
            }
        } elseif ($entry.PSIsContainer) {
            Remove-CacheLinks $entry.FullName
        }
    }
}

$removed = 0
foreach ($target in $targets) {
    if (-not $PSCmdlet.ShouldProcess($target, 'Delete test/build/runtime cache; preserve installed SDK and artifacts ZIPs')) {
        continue
    }
    $item = Get-Item -LiteralPath $target -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        [IO.Directory]::Delete($item.FullName)
    } else {
        Remove-CacheLinks $item.FullName
        Remove-Item -LiteralPath $item.FullName -Recurse -Force
    }
    if (Test-Path -LiteralPath $target) { throw "Cleanup incomplete: $target" }
    $removed++
    Write-Host "Deleted: $target"
}
foreach ($path in $zipHashes.Keys) {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $zipHashes[$path]) {
        throw "Upload ZIP changed: $path"
    }
}
Write-Host "Removed $removed cache directories; $($zipHashes.Count) upload ZIP(s) preserved."
