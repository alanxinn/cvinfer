# alanxinn_cvinfer

TensorRT/Denglin 双后端推理项目，包含：

- `core/tensorrt`：TensorRT engine、输入输出 tensor、CUDA stream。
- `core/denglin`：可选的登临（Denglin）dlNNE 独立 Core；通过公共 Session
  接口接入 YOLO、RF-DETR 和 C ABI。
- `yolo`：YOLO11 检测、分割、pose 前后处理。
- `rf_detr`：RF-DETR 检测/分割前后处理。
- `api`：稳定 C ABI 动态库。
- `tensorrt_builder`：ONNX 转 TensorRT engine。
- `csharp`：C# P/Invoke、NuGet 和 Demo。

## 运行环境

当前构建配置使用：

| 组件 | 版本 |
| --- | --- |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.9.3.28（`cudnn64_8.dll`） |
| TensorRT | 8.6.1.6（CUDA 11.8） |
| Denglin SDK | `E:/denglin-hamming/sdk`（启用 Denglin 时） |

需使用与 CUDA、TensorRT 和 GPU 架构匹配的驱动及运行库。

推理 NuGet 包包含对应的 native API 和运行时 DLL，应用只需安装匹配的
NVIDIA/Denglin 驱动。`alanxinn.TensorRtOnnxBuilder` 的完整 TensorRT parser、
plugin、cuBLAS 和 cuDNN 运行库由 `TensorRtRuntime` 单独配置或按需下载，
不会放入 `tensorrt_builder` 输出目录。

## NuGet Packages

| Package | Version | Description |
| --- | --- | --- |
| [alanxinn.CvinferSharp](https://www.nuget.org/packages/alanxinn.CvinferSharp) | 0.1.4 | C# TensorRT 推理封装及 TensorRT/CUDA/OpenCV runtime |
| alanxinn.CvinferSharp.Denglin | 0.1.4 | C# Denglin 推理封装及 Denglin SDK runtime |
| [alanxinn.TensorRtOnnxBuilder](https://www.nuget.org/packages/alanxinn.TensorRtOnnxBuilder) | 0.1.3 | C# ONNX 转 TensorRT engine，支持 FP32/FP16/INT8 |

## 模型类型

C# Demo 的 `--type` 只支持：

```text
yolo11-det
yolo11-seg
yolo11-pose
rf-detr-det
rf-detr-seg
```

示例：

```powershell
alanxinn.CvinferDemo.exe `
  --type rf-detr-seg `
  --model rfdetr-seg-nano.engine `
  --img bus.jpg `
  --warmup 100 `
  --iterations 1000 `
  --preprocess auto `
  --postprocess auto `
  --timing
```

输入尺寸和类别数默认从 engine 输出自动读取；`--classes` 仅用于显式限制类别数。
`warmup` 不计入统计，`iterations` 为正式统计次数。
计时默认关闭，需要输出耗时统计时添加 `--timing`。

RF-DETR 当前前处理、后处理使用 CPU；TensorRT 推理仍在 GPU 上执行。
显式选择 RF-DETR 的 `cuda` 前后处理会返回不支持错误。

## 构建

标准双后端构建（该 preset 继承的 base 已启用 Denglin）：

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
```

私有 CUDA backend：

```powershell
cmake --preset ninja-msvc-private-cuda-release
cmake --build --preset build-ninja-msvc-private-cuda-release
```

需要配置 `CUDA_ROOT`、`TENSORRT_ROOT`、`OPENCV_ROOT`。
上述两个 preset 还会使用 `DENGLIN_SDK_ROOT` 构建 Denglin；如果只需要
TensorRT，可在自定义配置时显式设置 `-DBUILD_DENGLIN_BACKEND=OFF`。
构建产物位于：

- `build/<preset>/bin/alanxinn_cvinfer_trt.dll`：TensorRT native API；
- `build/<preset>/bin/alanxinn_cvinfer_denglin.dll`：Denglin native API（启用 Denglin 时生成）；
- `build/<preset>/bintrt`：TensorRT/CUDA/OpenCV 及可选 CUDA backend runtime；
- `build/<preset>/bindl`：Denglin SDK runtime（含 `hc2`/`hc3` loader）；
- `build/<preset>/tensorrt_builder`：ONNX builder DLL。

启用 Denglin Core 时额外配置：

```powershell
cmake -S . -B build-denglin `
  -DBUILD_DENGLIN_BACKEND=ON `
  -DDENGLIN_SDK_ROOT=E:/denglin-hamming/sdk `
  -DCUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" `
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6_cuda-11.8" `
  -DOPENCV_ROOT=E:/opencv/build
```

不使用 preset 时，必须同时提供上述 CUDA、TensorRT、OpenCV 和 Denglin SDK
路径；这些路径可按本机安装位置调整。

Denglin Core 当前支持 ONNX 构建、异步 `Enqueue` 推理、原始 engine
序列化，以及带模型/SDK/设备元数据校验的 engine 缓存；详见
`core/denglin/README.md`。

C# 创建模型时可通过 `CvinferInferenceBackend.TensorRt` 或
`CvinferInferenceBackend.Denglin` 选择推理后端。启用 Denglin 构建时，
`--backend denglin` 可加载 ONNX 或 Denglin engine/cache；未启用时返回明确的
unsupported 错误。

## ONNX 转换

```powershell
dotnet add package alanxinn.TensorRtOnnxBuilder --version 0.1.3
```

RF-DETR 在 TensorRT 8.6 下使用 FP32 构建；builder 会自动避免错误的 FP16 输出。

## 约定

- engine 与 TensorRT、CUDA、GPU 架构相关。
- 模型权重位于 `**/weights/`，不提交 Git。
- 单个模型句柄默认串行使用；并发推理请创建独立句柄。

详细说明见各模块 README。

全部阶段变更及与原始实现的差异见
[IMPLEMENTATION_CHANGELOG.md](IMPLEMENTATION_CHANGELOG.md)。
