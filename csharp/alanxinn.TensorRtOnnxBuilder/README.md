# alanxinn.TensorRtOnnxBuilder

C# 调用 TensorRT native builder，将 ONNX 转为 serialized engine。
不启动 `trtexec.exe`。

## 安装

```powershell
dotnet add package alanxinn.TensorRtOnnxBuilder --version 0.1.3
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

TensorRT 8.6 解析原始 YOLO26 ONNX 可能在 `ReduceMax/TopK` 失败；
请先使用修正版 ONNX，或升级 TensorRT。

## Runtime

缺少 runtime 时默认下载并缓存到：

```text
%LOCALAPPDATA%\alanxinn\TensorRtOnnxBuilder\0.1.3\win-x64
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
