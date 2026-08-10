# TensorRT ONNX 转换 Demo

通过 `alanxinn.TensorRtOnnxBuilder` 将 ONNX 转为 TensorRT engine，
不调用 `trtexec.exe`。

## 准备

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
dotnet pack .\csharp\alanxinn.TensorRtOnnxBuilder\alanxinn.TensorRtOnnxBuilder.csproj -c Release
```

## 固定输入

```powershell
dotnet run --project .\csharp\alanxinn.TensorRtOnnxBuilderDemo `
  --onnx .\models\model.onnx `
  --engine .\engines\model.engine `
  --precision fp32 `
  --workspace-mib 4096
```

## 动态输入

```powershell
dotnet run --project .\csharp\alanxinn.TensorRtOnnxBuilderDemo `
  --onnx .\models\dynamic.onnx `
  --engine .\engines\dynamic.engine `
  --precision fp16 `
  --input-name images `
  --min-shape 1x3x320x320 `
  --opt-shape 1x3x640x640 `
  --max-shape 1x3x1280x1280
```

参数：

```text
--precision fp32|fp16|int8
--workspace-mib N
--device N
--no-tf32
--sparse
--strict
```

输出包含 engine 路径、大小和耗时。
