# alanxinn.TensorRtOnnxBuilder.TensorRT10

C# 调用 TensorRT native builder，将 ONNX 转为 serialized engine。
不启动 `trtexec.exe`。

混合 FP16 张量和 FP32 归一化的 ONNX（如 WeVisOne-VL）应设置
`StronglyTyped = true, Precision = TensorRtPrecision.Fp32, EnableTf32 = false`。
此时张量精度来自 ONNX，`Fp32` 表示不添加自动 FP16／INT8 降精度提示；不能同时启用 `StrictTypes`。
对应 Demo 参数为 `--strongly-typed --precision fp32 --no-tf32`。

## 安装

TensorRT 10 包名为 `alanxinn.TensorRtOnnxBuilder.TensorRT10`，版本为 `0.2.1`。

运行库版本独立，当前八个环境包保留 `0.2.0`；主包打包时选择最新稳定环境包，依赖允许更高版本。
需要在消费项目中始终使用最新稳定运行库时，运行 `scripts/Install-Cvinfer.ps1 -Project <项目.csproj> -IncludeBuilder`，
并用 `-Source` 指定包源。脚本会为八个运行库添加浮动直接引用；普通传递依赖默认取最低满足版本。
程序集名和 `using alanxinn.TensorRt` 保持兼容；旧的无后缀已发布包不适用于 TRT 10。

```powershell
dotnet add package alanxinn.TensorRtOnnxBuilder.TensorRT10 --version 0.2.1 --source E:/gitproject/DualAIVison/cvinfer/package/TensorRT10
```

要求 Windows x64、NVIDIA 驱动和匹配的 TensorRT/CUDA。
新 NuGet 自动还原 RTX 30/40/50 的 Core、SM86/SM89/SM120 和 PTX 资源依赖，
无需开发 SDK 或额外 ZIP；首次使用自动合并资源并校验 SHA256。
0.2.1 新增强类型构建支持，运行库由 NuGet 自动部署；尚未远程发布。实机验证仅覆盖 RTX 4060。
Windows x64 的派生 RID（如 .NET Framework 的 `win7-x64`）统一部署到 `runtimes/win-x64/native`。

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

本分支优先使用 NuGet 自动还原的资源；源码项目引用也可发现 `ALANXINN_TENSORRT_RUNTIME_DIR`、本地 CMake 的 `builder_runtime`，
或按 `TENSORRT_ROOT` / `CUDA_PATH` 从已安装 SDK 生成缓存，默认不复制 cuDNN/cuBLAS/NVRTC。
特定插件确实需要时设置 `TensorRtRuntime.IncludeOptionalLibraries = true`，此时才使用 `CUDNN_ROOT`。
NuGet 安装不需要配置 URL；仅源码/旧部署流程的外部 ZIP 下载需要显式 URL。缓存位于：

```text
%LOCALAPPDATA%\alanxinn\TensorRtOnnxBuilder\0.2.0\minimal\win-x64
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
./scripts/Build-Trt10.ps1
./scripts/Pack-Trt10.ps1 -SkipNativeBuild
```

详细转换命令见同目录 Demo README。
发布请运行 `scripts/Pack-Trt10.ps1`，同时生成依赖子包；直接 pack 会阻止生成缺少依赖的包。
默认覆盖 SM86/SM89/SM120，见 [DLL 清单](../../docs/TRT10_DLL_DEPENDENCIES.md)。
