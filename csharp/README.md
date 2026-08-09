# C# 类库与调用 Demo

目录包含四个项目：

```text
alanxinn.CvinferSharp/  C# 类库，封装 alanxinn_cvinfer.dll 的 P/Invoke 和结果对象
alanxinn.CvinferDemo/   控制台 Demo，通过 alanxinn.CvinferSharp 调用模型，使用 OpenCvSharp `Mat` 读取、传递和绘制图像
alanxinn.TensorRtOnnxBuilder/  独立的 ONNX 转 TensorRT engine 类库和 NuGet 包
alanxinn.TensorRtOnnxBuilderDemo/  使用上述 NuGet 包转换 ONNX 到 TensorRT engine 的控制台 Demo
```

ONNX 转 TensorRT engine 使用独立的 `alanxinn.TensorRtOnnxBuilder` 类库和 NuGet
包；它通过 `alanxinn_tensorrt_builder.dll` 调用 TensorRT C++ API，不启动
`trtexec.exe`。实现结论和 native 依赖清单见
[alanxinn.TensorRtOnnxBuilder/README.md](alanxinn.TensorRtOnnxBuilder/README.md) 和
[../tensorrt_builder/README.md](../tensorrt_builder/README.md)。

```powershell
dotnet add package alanxinn.TensorRtOnnxBuilder --version 0.1.2
```

ONNX 转 engine 的可运行示例见
[alanxinn.TensorRtOnnxBuilderDemo/README.md](alanxinn.TensorRtOnnxBuilderDemo/README.md)。
该包只在应用输出的 `runtimes/win-x64/native` 目录放置小型 builder DLL，不复制到输出根目录。
大型 TensorRT/CUDA runtime 首次调用时从 GitHub Release 下载到用户本地缓存，也可以提前
手动部署用于离线运行。

## 使用前准备

推荐通过 `alanxinn.CvinferSharp` NuGet 包部署 native runtime。包内只包含构建时
找到的以下运行时文件：

```text
runtimes/win-x64/native/
├─ alanxinn_cvinfer.dll
├─ alanxinn_cvinfer_cuda.dll（可选）
├─ nvinfer.dll
├─ cudart64_*.dll
├─ zlibwapi.dll
└─ opencv_world*.dll
```

使用方项目设置 `PlatformTarget=x64`，不要设置 `RuntimeIdentifier=win-x64`，
这样可以保留 NuGet 的 `runtimes/win-x64/native` 目录结构。TensorRT、CUDA、
OpenCV 运行库版本必须与 C++ DLL 和 engine 匹配。CPU 构建包可以不包含
`alanxinn_cvinfer_cuda.dll`；只有使用私有 CUDA 构建产物打包时才会包含它。

## 命令行

推荐使用命名参数：

```powershell
alanxinn.CvinferDemo.exe --type det --model yolo11n.engine --img bus.jpg --output result.jpg --classes 80 --preprocess auto --postprocess auto --warmup 100 --iterations 1000
alanxinn.CvinferDemo.exe --type seg --model yolo11n-seg.engine --img bus.jpg --output result.jpg --preprocess auto --postprocess auto --warmup 100 --iterations 1000

# RF-DETR 检测
alanxinn.CvinferDemo.exe --type rf-detr --model rf-detr.engine --img bus.jpg --output result.jpg --classes 80 --preprocess auto --postprocess auto --warmup 100 --iterations 1000
```

支持参数：`--type`、`--model`、`--img`、`--output`、`--classes`、`--preprocess`、`--postprocess`、`--warmup`、`--iterations`。输入宽高不再作为命令行参数传入：类库会直接读取 TensorRT engine 的输入 shape。固定输入 engine 使用固定 shape；动态输入 engine 使用 optimization profile 0 的 `OPT` shape。`warmup` 不计入统计，`iterations` 为正式测量次数，输出平均/最小/最大耗时。backend 支持 `auto`、`cpu`、`cuda`。不再兼容旧的位置参数形式。

RF-DETR 当前前处理和后处理实现为 CPU；选择 `auto` 或 `cpu` 均可，选择 `cuda` 会明确提示该模型暂不提供 CUDA 专用前/后处理实现。TensorRT 推理仍在 GPU 上执行。

`--input-width` 和 `--input-height` 已移除，继续传入会直接提示错误。

分割模式会生成带类别颜色的 Mask 叠加结果，并将每个原始 Mask 保存为
`<output>_mask_<index>.pgm`；检测框和分割框也按类别使用不同颜色。
启用 `--warmup` 或 `--iterations` 后进入 benchmark 模式，只保存最终结果图，
不再保存独立的 PGM Mask 文件。

## 类库调用流程

```csharp
using alanxinn;
using var model = new CvinferModel(enginePath, CvinferModelType.YoloDetection, config);
using var result = model.InferBgr(bgrBytes, width, height, 3);
var detection = result.GetDetection(0);
```

分割模型使用 `GetSegmentation()` 和 `CopyMask()` 获取 mask。类库负责句柄释放、UTF-8 路径转换、错误码转换和非托管缓冲区交互。

`alanxinn.CvinferDemo` 使用 OpenCvSharp `Mat` 直接读取 BGR 图像，并通过 Mat 的原生地址调用推理接口，不再经过 `Bitmap` 转换。

`alanxinn.CvinferDemo` 通过本地 NuGet 包引用 `alanxinn.CvinferSharp`。native DLL 作为 NuGet runtime native 资产使用，不由项目脚本复制到应用输出根目录：

```text
csharp/alanxinn.CvinferSharp/bin/Release/alanxinn.CvinferSharp.0.1.1.nupkg
```

使用 Demo 前先生成类库包：

```powershell
dotnet pack .\alanxinn.CvinferSharp\alanxinn.CvinferSharp.csproj -c Release
dotnet run --project .\alanxinn.CvinferDemo
```

Demo 项目会将类库包所在的 `bin/Release` 目录作为本地 NuGet 源。相同
`PackageId + Version` 的 NuGet 包会优先使用全局缓存；如果重新生成了同版本
包，必须清理缓存后重新还原：

```powershell
dotnet nuget locals global-packages --clear
dotnet nuget locals http-cache --clear
Remove-Item .\alanxinn.CvinferDemo\bin, .\alanxinn.CvinferDemo\obj -Recurse -Force
dotnet restore .\alanxinn.CvinferDemo\alanxinn.CvinferDemo.csproj
```

否则可能出现 `.nupkg` 中存在 `alanxinn_cvinfer_cuda.dll`，但
`runtimes/win-x64/native` 输出目录没有该 DLL 的情况。

## 生成 NuGet 包

`alanxinn.CvinferSharp` 已配置 NuGet 元数据和文件版本：

```text
PackageId: alanxinn.CvinferSharp
Version: 0.1.1
AssemblyVersion: 0.1.1.0
FileVersion: 0.1.1.0
TargetFrameworks: net472; net6.0; net8.0
PlatformTarget: x64
```

在 `csharp/alanxinn.CvinferSharp` 目录执行：

```powershell
dotnet pack -c Release
```

生成的 `.nupkg` 默认位于 `bin/Release`。项目会优先从
`build/ninja-msvc-private-cuda-release/bin`，否则从
`build/ninja-msvc-release/bin` 复制 `alanxinn_cvinfer.dll`、可选的
`alanxinn_cvinfer_cuda.dll`、`nvinfer.dll`、`cudart64_*.dll` 和 Release 版
`opencv_world*.dll` 到包内 `runtimes/win-x64/native`。不会通过自定义 MSBuild
target 将这些 DLL 复制到使用方输出根目录。详细说明见
[alanxinn.CvinferSharp/README.md](alanxinn.CvinferSharp/README.md)。

如果需要 CUDA NuGet 包，先使用 `ninja-msvc-private-cuda-release` 构建预设；
如果使用普通 `ninja-msvc-release`，生成的是 CPU/Auto 包，CUDA DLL 会被跳过。
