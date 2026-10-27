#Requires -Version 5.1
param([Parameter(Mandatory = $true)][string]$Path)
$ErrorActionPreference = 'Stop'

# Inspect both regular and delay-load PE imports. A public wrapper must not
# require a private implementation even when its DLL is omitted from the ZIP.
$image = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Path).Path)
if ($image.Length -lt 64) { throw 'Native runtime is not a valid PE image' }
$pe = [BitConverter]::ToInt32($image, 0x3c)
if ($pe -lt 64 -or $pe + 24 -gt $image.Length -or [BitConverter]::ToUInt32($image, $pe) -ne 0x4550) {
    throw 'Native runtime has an invalid PE header'
}
$optional = $pe + 24
$optionalSize = [BitConverter]::ToUInt16($image, $pe + 20)
if ($optional + $optionalSize -gt $image.Length -or $optionalSize -lt 224 -or
    [BitConverter]::ToUInt16($image, $optional) -ne 0x20b) {
    throw 'Public native runtime must be a Windows x64 PE image'
}
$sections = @()
$sectionCount = [BitConverter]::ToUInt16($image, $pe + 6)
for ($i = 0; $i -lt $sectionCount; $i++) {
    $offset = $optional + $optionalSize + 40 * $i
    if ($offset + 40 -gt $image.Length) { throw 'Invalid PE section table' }
    $sections += [pscustomobject]@{
        Rva = [BitConverter]::ToUInt32($image, $offset + 12)
        Size = [Math]::Max([BitConverter]::ToUInt32($image, $offset + 8), [BitConverter]::ToUInt32($image, $offset + 16))
        Raw = [BitConverter]::ToUInt32($image, $offset + 20)
    }
}
function Get-ImageOffset([long]$Rva) {
    foreach ($section in $sections) {
        if ($Rva -ge $section.Rva -and $Rva -lt $section.Rva + $section.Size) {
            return [int]($section.Raw + $Rva - $section.Rva)
        }
    }
    throw "Invalid PE RVA: $Rva"
}
function Get-ImportName([long]$Rva) {
    $offset = Get-ImageOffset $Rva
    $end = $offset
    while ($end -lt $image.Length -and $image[$end] -ne 0 -and $end - $offset -lt 512) { $end++ }
    if ($end -ge $image.Length -or $image[$end] -ne 0) { throw 'Invalid PE import name' }
    return [Text.Encoding]::ASCII.GetString($image, $offset, $end - $offset)
}
$imports = @()
foreach ($kind in @('regular', 'delay')) {
    $directoryIndex = if ($kind -eq 'regular') { 1 } else { 13 }
    $rva = [BitConverter]::ToUInt32($image, $optional + 112 + 8 * $directoryIndex)
    if ($rva -eq 0) { continue }
    $offset = Get-ImageOffset $rva
    $stride = if ($kind -eq 'regular') { 20 } else { 32 }
    while ($true) {
        if ($offset + $stride -gt $image.Length) { throw 'Invalid PE import descriptor' }
        $nameOffset = if ($kind -eq 'regular') { 12 } else { 4 }
        $nameRva = [BitConverter]::ToUInt32($image, $offset + $nameOffset)
        if ($nameRva -eq 0) { break }
        if ($kind -eq 'delay' -and ([BitConverter]::ToUInt32($image, $offset) -band 1) -eq 0) {
            throw 'Unsupported VA-based delay import; refusing public package'
        }
        $imports += Get-ImportName $nameRva
        $offset += $stride
    }
}
$privateImports = @($imports | Where-Object { $_ -match '^alanxinn_cvinfer_(cuda|denglin_accel)\.dll$' })
if ($privateImports.Count -gt 0) {
    throw "Private accelerator dependency cannot be published: $($privateImports -join ', '). Rebuild with BUILD_CUDA_BACKEND=OFF and BUILD_DENGLIN_ACCELERATOR=OFF."
}
Write-Host "Public native runtime verified (including delay imports): $Path"
