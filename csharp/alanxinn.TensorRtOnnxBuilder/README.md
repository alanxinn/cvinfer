# alanxinn.TensorRtOnnxBuilder

C# 调用 TensorRT native builder，将 ONNX 转为 serialized engine。
不启动 `trtexec.exe`。

## 安装

本分支版本为 `0.2.0-trt10`，使用本地项目引用或自行 pack；旧的已发布包不适用于 TRT 10。

```powershell
dotnet add reference path/to/cvinfer/csharp/alanxinn.TensorRtOnnxBuilder/alanxinn.TensorRtOnnxBuilder.csproj
```

要求 Windows x64、NVIDIA 驱动和匹配的 TensorRT/CUDA。
NuGet 仅包含 C# 程序集和 builder DLL。

## C# 示例

```csharp
using alanxinn.TensorRt;

var result = TensorRtOnnxBuilder.Build(
    "model.onnx",
    "model.engine",
    new TensorRtBuildOptions
    {
        Precision = TensorRtPrecision.Fp32,
        WorkspaceMiB = 4096,
        EnableTf32 = true
    });
```

## 动态输入

```csharp
var profile = new TensorRtOptimizationProfile
{
    MinShapes = new() { ["images"] = new TensorRtShape(1, 3, 320, 320) },
    OptShapes = new() { ["images"] = new TensorRtShape(1, 3, 640, 640) },
    MaxShapes = new() { ["images"] = new TensorRtShape(1, 3, 1280, 1280) }
};
```

动态输入的每个输入都必须同时出现在 min/opt/max 中。
INT8 适用于包含 Q/DQ 节点的 ONNX。

## Runtime

本分支优先发现 `ALANXINN_TENSORRT_RUNTIME_DIR`、本地 CMake 的 `builder_runtime`，
或按 `TENSORRT_ROOT` / `CUDA_PATH` / `CUDNN_ROOT` 从已安装 SDK 生成缓存。
网络下载必须显式配置匹配 TRT 10 的 ZIP URL；没有默认已发布的 TRT 10 下载包。缓存位于：

```text
%LOCALAPPDATA%\alanxinn\TensorRtOnnxBuilder\0.2.0-trt10\win-x64
```

可配置：

```csharp
TensorRtRuntime.DownloadUrl = "...";
TensorRtRuntime.CacheDirectory = @"D:\TensorRT\runtime";
TensorRtRuntime.ExpectedSha256 = "...";
TensorRtRuntime.DisableAutoDownload = true;
```

## 构建

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
dotnet pack .\csharp\alanxinn.TensorRtOnnxBuilder\alanxinn.TensorRtOnnxBuilder.csproj -c Release
```

详细转换命令见同目录 Demo README。
