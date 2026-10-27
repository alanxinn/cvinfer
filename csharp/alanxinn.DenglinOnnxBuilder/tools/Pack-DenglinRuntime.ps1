param(
    [Parameter(Mandatory = $true)]
    [string] $SdkRoot,

    [Parameter(Mandatory = $true)]
    [string] $PythonRoot,

    [Parameter(Mandatory = $true)]
    [string] $OutputZip,

    [string] $NativeDirectory,
    [string] $Version = '0.1.4',

    [switch] $IncludeOptionalPythonPackages
)

$ErrorActionPreference = 'Stop'

$sdk = (Resolve-Path -LiteralPath $SdkRoot).Path
$python = (Resolve-Path -LiteralPath $PythonRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $sdk 'python\tvm\dl\python\dl\__init__.py'))) {
    throw "Denglin SDK Python module was not found under $sdk"
}
if (-not (Test-Path -LiteralPath (Join-Path $sdk 'python\tvm\dl\python\dl\topi\cuda\__init__.py'))) {
    throw "Denglin SDK dl.topi.cuda module was not found under $sdk"
}
if (-not (Test-Path -LiteralPath (Join-Path $sdk 'bin\nnexec.exe'))) {
    throw "Denglin SDK nnexec.exe was not found under $sdk"
}
if (-not (Test-Path -LiteralPath (Join-Path $python 'python.exe'))) {
    throw "python.exe was not found under $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $python 'Lib\site-packages\onnx\__init__.py'))) {
    throw "ONNX Python package was not found under $python"
}
foreach ($package in @('numpy', 'scipy', 'setuptools')) {
    if (-not (Test-Path -LiteralPath (Join-Path $python ("Lib\site-packages\$package")))) {
        throw "$package Python package was not found under $python"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $python 'Lib\site-packages\dataclasses.py')) -and
    -not (Test-Path -LiteralPath (Join-Path $python 'Lib\dataclasses.py'))) {
    throw "dataclasses Python module was not found under $python"
}

$staging = Join-Path ([System.IO.Path]::GetTempPath()) ('denglin-runtime-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging | Out-Null
$script:DenglinSdkFileLinks = [Collections.Generic.List[string]]::new()

function Test-SkippedName([System.IO.FileSystemInfo] $item, [bool] $isPythonRoot) {
    if ($item.Name -in @('__pycache__', '.git', '.vs', 'obj', 'build', 'tests', 'test', 'samples')) { return $true }
    if ($item.Extension -in @('.pdb', '.ilk', '.obj', '.pyc')) { return $true }
    if ($isPythonRoot -and -not $IncludeOptionalPythonPackages -and
        ($item.Name -like 'tensorflow*' -or
         $item.Name -like 'tensorboard*' -or
         $item.Name -like 'keras*' -or
         $item.Name -like 'torch*' -or
         $item.Name -like 'torchvision*' -or
         $item.Name -like 'paddle*' -or
         $item.Name -like 'jax*' -or
         $item.Name -eq 'nvidia' -or
         $item.Name -eq 'cuda' -or
         $item.Name -eq 'cudnn')) { return $true }
    return $false
}

function Copy-FilteredTree([string] $source, [string] $destination,
    [string] $archiveRoot, [bool] $isPythonRoot = $false) {
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    foreach ($item in Get-ChildItem -LiteralPath $source -Force) {
        if (Test-SkippedName $item $isPythonRoot) { continue }
        $target = Join-Path $destination $item.Name
        if ($item.PSIsContainer) {
            Copy-FilteredTree $item.FullName $target $archiveRoot $isPythonRoot
        } elseif ($archiveRoot -eq 'sdk' -and $item.LinkType -eq 'SymbolicLink') {
            $sdkPrefix = $sdk.TrimEnd('\') + '\'
            $linkTarget = [string]$item.Target
            $resolvedTarget = if ([IO.Path]::IsPathRooted($linkTarget)) {
                [IO.Path]::GetFullPath($linkTarget)
            } else {
                [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $item.FullName) $linkTarget))
            }
            if (-not $resolvedTarget.StartsWith($sdkPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Denglin SDK symlink target is outside the SDK root: $($item.FullName) -> $resolvedTarget"
            }

            $relativeLink = $item.FullName.Substring($sdkPrefix.Length).Replace('\', '/')
            $relativeTarget = $resolvedTarget.Substring($sdkPrefix.Length).Replace('\', '/')
            $script:DenglinSdkFileLinks.Add("$relativeLink|$relativeTarget")
            Write-Host "Recording SDK hard link: $relativeLink -> $relativeTarget"
        } else {
            Copy-Item -LiteralPath $item.FullName -Destination $target -Force
        }
    }
}

try {
    Copy-FilteredTree $sdk (Join-Path $staging 'sdk') 'sdk' $false
    Copy-FilteredTree $python (Join-Path $staging 'python') 'python' $true
    if ($script:DenglinSdkFileLinks.Count -gt 0) {
        $linkManifest = Join-Path $staging 'sdk\denglin-file-links.txt'
        $script:DenglinSdkFileLinks | Set-Content -LiteralPath $linkManifest -Encoding UTF8
    }
    if ($NativeDirectory) {
        $native = (Resolve-Path -LiteralPath $NativeDirectory).Path
        Copy-FilteredTree $native (Join-Path $staging 'native') 'native'
    }

    $manifest = [ordered]@{
        id = 'alanxinn-denglin-runtime'
        version = $Version
        rid = 'win-x64'
        sdk = 'sdk'
        python = 'python/python.exe'
        native = if ($NativeDirectory) { 'native' } else { $null }
    }
    $manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $staging 'manifest.json') -Encoding UTF8

    $outputDirectory = Split-Path -Parent $OutputZip
    if ($outputDirectory) { New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null }
    if (Test-Path -LiteralPath $OutputZip) { Remove-Item -LiteralPath $OutputZip -Force }
    Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $OutputZip -CompressionLevel Optimal
    $hash = (Get-FileHash -LiteralPath $OutputZip -Algorithm SHA256).Hash
    Write-Host "Denglin runtime ZIP created: $OutputZip"
    Write-Host "SHA256: $hash"
    Write-Host "Layout: sdk/ python/$(if ($NativeDirectory) { ' native/' })"
    if (-not $IncludeOptionalPythonPackages) {
        Write-Host "Optional training packages excluded: TensorFlow, PyTorch, Paddle, JAX and CUDA Python packages"
    }
}
finally {
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
