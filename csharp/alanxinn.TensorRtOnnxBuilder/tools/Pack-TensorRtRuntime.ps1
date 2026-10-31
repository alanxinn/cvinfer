param(
    [Parameter(Mandatory = $true)]
    [string] $SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string] $OutputZip,
    [string[]] $CudaArchitectures = @('89'),
    [switch] $AllArchitectures,
    [switch] $IncludeOptionalLibraries
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
if ($AllArchitectures) {
    $resources = @(Get-ChildItem -LiteralPath $source -Filter 'nvinfer_builder_resource_sm*_10.dll' -File | Select-Object -ExpandProperty Name)
    if ($resources.Count -eq 0) { throw 'No architecture-specific builder resources found' }
    $requiredFiles += $resources
} else {
    if ($CudaArchitectures.Count -eq 0) { throw 'Specify at least one builder SM architecture' }
    foreach ($architecture in $CudaArchitectures) {
        if ($architecture -notmatch '^\d+$') { throw "Invalid SM architecture: $architecture" }
        $requiredFiles += "nvinfer_builder_resource_sm${architecture}_10.dll"
    }
}
$requiredFiles = @($requiredFiles | Select-Object -Unique)
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

    # These libraries are dynamically needed by specific plugins/legacy layers,
    # not by the project's standard YOLO/RF-DETR/SAM3/DINOv3 paths.
    if ($IncludeOptionalLibraries) {
        foreach ($fileName in @('cudnn64_9.dll', 'cudnn_graph64_9.dll', 'cublas64_12.dll', 'cublasLt64_12.dll', 'nvrtc64_120_0.dll', 'nvrtc-builtins64_129.dll')) {
            if (-not (Test-Path -LiteralPath (Join-Path $source $fileName) -PathType Leaf)) {
                throw "Optional library missing: $fileName. Stage with -IncludeOptionalLibraries first."
            }
        }
        foreach ($pattern in @('cublas*64_12.dll', 'cudnn*64_9.dll', 'nvrtc*.dll')) {
            Get-ChildItem -LiteralPath $source -Filter $pattern -File | Copy-Item -Destination $staging -Force
        }
    }
    $selectedArchitectures = @($requiredFiles | ForEach-Object { if ($_ -match '^nvinfer_builder_resource_sm(\d+)_10\.dll$') { $Matches[1] } })
    [ordered]@{ format = 'cvinfer-trt10-runtime-v2'; architectures = $selectedArchitectures; allArchitectures = [bool]$AllArchitectures;
        optionalLibraries = [bool]$IncludeOptionalLibraries; dlls = @(Get-ChildItem -LiteralPath $staging -Filter '*.dll' | Select-Object -ExpandProperty Name) } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $staging 'runtime-manifest.json') -Encoding UTF8
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
