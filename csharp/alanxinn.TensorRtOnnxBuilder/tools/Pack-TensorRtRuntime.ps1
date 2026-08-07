param(
    [Parameter(Mandatory = $true)]
    [string] $SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string] $OutputZip
)

$ErrorActionPreference = 'Stop'
$requiredFiles = @(
    'nvinfer.dll',
    'nvinfer_plugin.dll',
    'nvonnxparser.dll',
    'cudart64_110.dll',
    'cublas64_11.dll',
    'cublasLt64_11.dll',
    'cudnn64_8.dll',
    'zlibwapi.dll'
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
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
