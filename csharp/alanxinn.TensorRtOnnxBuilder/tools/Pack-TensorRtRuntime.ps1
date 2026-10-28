param(
    [Parameter(Mandatory = $true)]
    [string] $SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string] $OutputZip
)

$ErrorActionPreference = 'Stop'
$requiredFiles = @(
    'nvinfer_10.dll',
    'nvinfer_plugin_10.dll',
    'nvonnxparser_10.dll',
    'nvinfer_builder_resource_ptx_10.dll',
    'cudart64_12.dll'
)

$source = (Resolve-Path -LiteralPath $SourceDirectory).Path
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ('tensorrt-runtime-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging | Out-Null
try {
    foreach ($fileName in $requiredFiles) {
        $sourceFile = Join-Path $source $fileName
        if (-not (Test-Path -LiteralPath $sourceFile -PathType Leaf)) {
            throw "Missing required TensorRT runtime DLL: $sourceFile"
        }
        Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $staging $fileName)
    }

    # Architecture-specific builder resources and cuDNN 9 component DLLs must
    # travel together. SourceDirectory can be CMake's builder_runtime directory.
    foreach ($pattern in @('nvinfer_builder_resource*_10.dll', 'cublas*64_12.dll', 'cudnn*64_9.dll', 'nvrtc*.dll')) {
        Get-ChildItem -LiteralPath $source -Filter $pattern -File | Copy-Item -Destination $staging -Force
    }
    $outputDirectory = Split-Path -Parent $OutputZip
    if ($outputDirectory) { New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null }
    if (Test-Path -LiteralPath $OutputZip) { Remove-Item -LiteralPath $OutputZip -Force }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $OutputZip -CompressionLevel Optimal
    Write-Host "TensorRT runtime ZIP created: $OutputZip"
    Write-Host "Included DLLs: $($requiredFiles -join ';')"
    $hash = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash
    Write-Host "SHA256: $hash"
}
finally {
    $resolvedStaging = [IO.Path]::GetFullPath($staging)
    $tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolvedStaging.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove staging directory outside TEMP: $resolvedStaging"
    }
    if (Test-Path -LiteralPath $resolvedStaging) { Remove-Item -LiteralPath $resolvedStaging -Recurse -Force }
}
