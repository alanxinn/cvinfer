# TensorRT ONNX 转 Engine Demo

该控制台程序通过 NuGet 包 `alanxinn.TensorRtOnnxBuilder` 调用 TensorRT C++ DLL，
直接将 ONNX 模型转换为 TensorRT serialized engine，不启动 `trtexec.exe`。

## 使用

以下命令从仓库根目录执行。先构建 native builder，并打包版本 `0.1.1` 的本地 NuGet
（Demo 会从 `csharp\alanxinn.TensorRtOnnxBuilder\bin\Release` 作为本地源还原）：

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
dotnet pack .\csharp\alanxinn.TensorRtOnnxBuilder\alanxinn.TensorRtOnnxBuilder.csproj -c Release
```

打包时控制台只会显示 NuGet 收集的 `alanxinn_tensorrt_builder.dll`。大型 TensorRT/CUDA
runtime 不再进入 NuGet；Demo 首次转换时会从 GitHub Release 自动下载并缓存。Demo 和
NuGet 均要求 Windows x64；不要设置 `RuntimeIdentifier=win-x64`。

默认下载地址为：

```text
https://github.com/alanxinn/cvinfer/releases/download/tensorrt8.1.6-cuda11.8-runtime-win-x64-0.1.1.zip/tensorrt8.1.6-cuda11.8-runtime-win-x64-0.1.1.zip
```

也可以在程序启动时设置 `TensorRtRuntime.DownloadUrl`、`ExpectedSha256` 和
`CacheDirectory`，或提前将八个 runtime DLL 放入 `CacheDirectory` 实现离线运行。

执行固定输入模型转换：

```powershell
dotnet run --project .\csharp\alanxinn.TensorRtOnnxBuilderDemo `
  --onnx .\models\model.onnx `
  --engine .\engines\model.fp16.engine `
  --precision fp16 `
  --workspace-mib 4096
```

动态输入模型需要提供 profile：

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

`--precision` 支持 `fp32`、`fp16`、`int8`；`--no-tf32`、`--sparse`、`--strict`
分别用于关闭 TF32、启用稀疏权重和启用 strict types。程序会输出 engine 路径、大小
和转换耗时。NuGet 只在
`bin\Debug\net8.0\runtimes\win-x64\native`（或对应配置目录）放置 builder DLL，
不会复制到 `net8.0` 输出根目录；下载的大型 runtime 默认位于用户本地缓存目录。
