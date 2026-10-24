param(
    [string]$BuilderExe = 'csharp\alanxinn.DenglinOnnxBuilderDemo\bin\Release\net8.0-windows\alanxinn.DenglinOnnxBuilderDemo.exe',
    [string]$DemoExe = 'csharp\alanxinn.CvinferDemo\bin\Debug\net8.0-windows\alanxinn.CvinferDemo.exe',
    [string]$OutputRoot = 'cache\denglin-auto',
    [int]$Iterations = 1,
    [switch]$BuildOnly,
    [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'
$repo = (Get-Location).Path
$builder = (Resolve-Path -LiteralPath $BuilderExe).Path
$demo = if (-not $BuildOnly) { (Resolve-Path -LiteralPath $DemoExe).Path } else { $null }
$outputRootPath = [IO.Path]::GetFullPath((Join-Path $repo $OutputRoot))
New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null

$models = @(
    @{ Name='yolo11n'; Type='yolo11-det'; Onnx='test\yolo11n.onnx' },
    @{ Name='yolo11n-seg'; Type='yolo11-seg'; Onnx='test\yolo11n-seg.onnx' },
    @{ Name='yolo11s-pose'; Type='yolo11-pose'; Onnx='test\yolo11s-pose.onnx' },
    @{ Name='rfdetr-det'; Type='rf-detr-det'; Onnx='rf_detr\weights\onnx\det-n\rfdetr-nano.onnx' },
    @{ Name='rfdetr-seg'; Type='rf-detr-seg'; Onnx='rf_detr\weights\onnx\seg-n\rfdetr-seg-nano.onnx' }
)

$results = [Collections.Generic.List[object]]::new()
foreach ($model in $models) {
    $onnx = [IO.Path]::GetFullPath((Join-Path $repo $model.Onnx))
    $engine = Join-Path $outputRootPath "$($model.Name).denglin.raw.engine"
    $buildExit = -1
    $inferExit = $null
    $status = 'FAIL'
    $message = ''

    try {
        if (-not (Test-Path -LiteralPath $onnx)) { throw "ONNX file not found: $onnx" }
        Write-Host "[BUILD] $($model.Name)"
        & $builder --onnx $onnx --engine $engine
        $buildExit = $LASTEXITCODE
        if ($buildExit -ne 0) { throw "builder exit code $buildExit" }
        if (-not (Test-Path -LiteralPath $engine) -or (Get-Item -LiteralPath $engine).Length -le 0) {
            throw 'builder returned success but did not create a non-empty engine'
        }

        if (-not $BuildOnly) {
            Write-Host "[INFER] $($model.Name)"
            & $demo --type $model.Type --model $engine --img (Join-Path $repo 'test\bus.jpg') `
                --warmup 1 --iterations $Iterations --timing --backend denglin
            $inferExit = $LASTEXITCODE
            if ($inferExit -ne 0) { throw "inference exit code $inferExit" }
        }
        $status = 'PASS'
        $message = if ($BuildOnly) { 'engine created' } else { 'engine created and inference passed' }
    }
    catch {
        $message = $_.Exception.Message
        Write-Warning "$($model.Name): $message"
        if (-not $ContinueOnError) {
            # Keep the report useful even when stopping at the first failure.
            $results.Add([pscustomobject]@{ Model=$model.Name; BuildExit=$buildExit; InferenceExit=$inferExit; Status=$status; Message=$message; Engine=$engine })
            break
        }
    }
    $results.Add([pscustomobject]@{ Model=$model.Name; BuildExit=$buildExit; InferenceExit=$inferExit; Status=$status; Message=$message; Engine=$engine })
}

$reportPath = Join-Path $outputRootPath 'validation.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding UTF8
$results | Format-Table -AutoSize
Write-Host "Validation report: $reportPath"
if (@($results | Where-Object Status -ne 'PASS').Count -gt 0) { exit 1 }
exit 0
