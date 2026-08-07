# alanxinn.TensorRtOnnxBuilder

独立的 C# 类库和 NuGet 包，用于调用 TensorRT C++ builder 将 ONNX 转换为序列化
TensorRT engine。类库不启动 `trtexec.exe`，而是通过 P/Invoke 调用包内的
`alanxinn_tensorrt_builder.dll`。由于 TensorRT/CUDA DLL 体积超过 nuget.org 单包限制，
大型依赖改为首次使用时从 GitHub Release 下载并缓存。

## 安装

```powershell
dotnet add package alanxinn.TensorRtOnnxBuilder --version 0.1.1
```

使用条件：Windows x64、兼容的 NVIDIA 驱动和与 engine 目标环境匹配的 TensorRT。
NuGet 包只包含 C# 程序集和小型 native builder DLL；不包含大型 TensorRT/CUDA DLL、
NVIDIA 开发头文件和导入库。
engine 仍然与 TensorRT 版本、CUDA 版本和 GPU 架构相关，不能保证跨环境通用。

## Runtime 自动下载

首次调用 `TensorRtOnnxBuilder.Build` 时，库会按以下顺序查找 runtime：应用的
`runtimes/win-x64/native`、应用目录、本地缓存目录。找不到完整 runtime 时，默认下载：

```text
https://github.com/alanxinn/cvinfer/releases/download/v0.1.1/tensorrt8.1.6-cuda11.8-runtime-win-x64-0.1.1.zip
```

缓存目录默认为 `%LOCALAPPDATA%\alanxinn\TensorRtOnnxBuilder\0.1.1\win-x64`。
也可以在启动时显式配置镜像、缓存和 SHA256：

```csharp
using alanxinn.TensorRt;

TensorRtRuntime.DownloadUrl = "https://your-server/tensorrt8.1.6-cuda11.8-runtime-win-x64-0.1.1.zip";
TensorRtRuntime.ExpectedSha256 = "ZIP 文件 SHA256";
TensorRtRuntime.CacheDirectory = @"D:\TensorRT\runtime";
TensorRtRuntime.Initialize();
```

离线部署时将八个 runtime DLL 手动放入 `CacheDirectory`，或设置
`TensorRtRuntime.DisableAutoDownload = true`。GitHub Release ZIP 必须包含：
`nvinfer.dll`、`nvinfer_plugin.dll`、`nvonnxparser.dll`、`cudart64_110.dll`、
`cublas64_11.dll`、`cublasLt64_11.dll`、`cudnn64_8.dll`、`zlibwapi.dll`。

## 固定输入模型

```csharp
using alanxinn.TensorRt;

var result = TensorRtOnnxBuilder.Build(
    @"models\model.onnx",
    @"engines\model.fp16.engine",
    new TensorRtBuildOptions
    {
        Precision = TensorRtPrecision.Fp16,
        WorkspaceMiB = 4096,
        EnableTf32 = true
    });

Console.WriteLine($"{result.EnginePath}: {result.EngineSizeBytes} bytes");
```

## 动态输入

```csharp
using System.Collections.Generic;
using alanxinn.TensorRt;

var profile = new TensorRtOptimizationProfile
{
    MinShapes = new Dictionary<string, TensorRtShape>
    {
        ["images"] = new TensorRtShape(1, 3, 320, 320)
    },
    OptShapes = new Dictionary<string, TensorRtShape>
    {
        ["images"] = new TensorRtShape(1, 3, 640, 640)
    },
    MaxShapes = new Dictionary<string, TensorRtShape>
    {
        ["images"] = new TensorRtShape(1, 3, 1280, 1280)
    }
};

TensorRtOnnxBuilder.Build(
    "dynamic.onnx",
    "dynamic.engine",
    new TensorRtBuildOptions
    {
        Precision = TensorRtPrecision.Fp16,
        OptimizationProfiles = new[] { profile }
    });
```

动态 ONNX 的每个输入都必须在 min/opt/max 三个字典中出现；`opt` shape 是 TensorRT
主要优化的尺寸。INT8 适用于已经包含 Q/DQ 量化节点的 ONNX；需要自定义 calibrator
时应扩展 native C++ builder API，而不是把 TensorRT C++ ABI 直接暴露给 C#。

## 构建与打包

先用 CMake 构建 native builder（会将 builder 和 TensorRT/CUDA 依赖复制到独立的
`build/<preset>/tensorrt_builder` 目录，不会与常规 cvinfer 的 `bin` 混用）：

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
dotnet pack .\csharp\alanxinn.TensorRtOnnxBuilder\alanxinn.TensorRtOnnxBuilder.csproj -c Release
```

打包项目会从 `NativeRuntimeSource`（默认指向上述 `tensorrt_builder` 目录）查找
`alanxinn_tensorrt_builder.dll` 并放入：

```text
runtimes/win-x64/native/
└─ alanxinn_tensorrt_builder.dll
```

NativeRuntimeSource 目录仍可保留完整 DLL 集合，供本地运行和制作 GitHub Release ZIP；
NuGet 不再收集大型 TensorRT、CUDA 或 cuDNN DLL。

使用仓库内脚本制作 Release ZIP：

```powershell
.\tools\Pack-TensorRtRuntime.ps1 `
  -SourceDirectory ..\..\build\ninja-msvc-release\tensorrt_builder `
  -OutputZip .\tensorrt8.1.6-cuda11.8-runtime-win-x64-0.1.1.zip
```

`dotnet pack` 时会在控制台输出 `TensorRT native DLLs staged` 和
`TensorRT NuGet native DLLs`，可核对实际复制和打包的文件列表。

也可以在 `dotnet pack` 时通过 `-p:NativeRuntimeSource=<目录>` 指定自定义 CMake
输出目录。TensorRT 开发环境必须在 native builder 编译时存在；使用方安装 NuGet 包、
兼容显卡驱动并允许首次运行下载 runtime 即可调用转换接口。应用应使用 x64 平台。
