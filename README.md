# alanxinn_cvinfer

TensorRT/Denglin 双后端推理项目，包含：

- `core/tensorrt`：TensorRT engine、输入输出 tensor、CUDA stream。
- `core/denglin`：可选的登临（Denglin）dlNNE 独立 Core；通过公共 Session
  接口接入 YOLO、RF-DETR 和 C ABI。
- `yolo`：YOLO11 检测、分割、pose 前后处理。
- `rf_detr`：RF-DETR 检测/分割前后处理。
- `sam3`：SAM3 框与正负点交互、文本概念分割；已使用本地 SAM3.1 真实权重完成 C ABI/C# 推理对照验证。
- `dinov3`：DINOv3 ViT/ConvNeXt 图像特征推理，输出全局向量、patch 特征与特征图。
- `api`：稳定 C ABI 动态库。
- `tensorrt_builder`：ONNX 转 TensorRT engine。
- `csharp`：C# P/Invoke、NuGet 和 Demo。

## 运行环境

TRT10 构建及模型验证见 [适配说明](docs/TRT10.md) 和 [本机验证记录](docs/TRT10_VALIDATION.md)。
SAM3 的导出、接口、测试范围和权重要求见 [SAM3 说明](sam3/README.md)。
DINOv3 的图像推理、特征输出和验证范围见 [DINOv3 说明](dinov3/README.md)。

当前构建配置使用：

| 组件 | 版本 |
| --- | --- |
| CUDA Toolkit | 12.9.86 |
| cuDNN | 9.27.0（`cudnn64_9.dll` 及其组件） |
| TensorRT | 10.16.1.11（CUDA 12） |
| Denglin SDK | `E:/denglin-hamming/sdk`（启用 Denglin 时） |

需使用与 CUDA、TensorRT 和 GPU 架构匹配的驱动及运行库。

推理 NuGet 包包含对应的 native API 和运行时 DLL，应用只需安装匹配的
NVIDIA/Denglin 驱动。`alanxinn.TensorRtOnnxBuilder` 的完整 TensorRT parser、
plugin、cuBLAS 和 cuDNN 运行库由 `TensorRtRuntime` 单独配置或按需下载，
保存在独立的 `builder_runtime` 目录。TRT 10 本身不依赖外部 cuDNN；
为本项目指定的环境同时部署 cuDNN 9.27 组件。详细步骤见 [TRT10 适配说明](docs/TRT10.md)。

## NuGet Packages

本分支的 TensorRT 包版本为 `0.2.0-trt10`，应从源码构建；下表保留已发布版本信息，
已发布的 `0.1.x` TensorRT 包对应旧 SDK，不能与本分支的 native DLL 混用。

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
yolo26-det
yolo26-seg
yolo26-pose
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

## 构建

本分支默认构建 TensorRT 10 后端。自动初始化 MSVC 和 Ninja：

```powershell
.\scripts\Build-Trt10.ps1
# 启用已有的 private/cuda_backend（本机存在时）
.\scripts\Build-Trt10.ps1 -Preset trt10-cuda-release
```

preset 已配置本机 `CUDA_ROOT`、`TENSORRT_ROOT`、`CUDNN_ROOT`、`OpenCV_DIR`。
默认 CUDA 架构为 RTX 4060 的 SM89；其他 GPU 使用 `-CudaArchitectures` 覆盖。
Denglin 为可选后端，需要显式启用 `BUILD_DENGLIN_BACKEND`。
构建产物位于：

- `build/<preset>/bin/alanxinn_cvinfer_trt.dll`：TensorRT native API；
- `build/<preset>/bin/alanxinn_cvinfer_denglin.dll`：Denglin native API（启用 Denglin 时生成）；
- `build/<preset>/bintrt`：TensorRT/CUDA/OpenCV runtime；
- `build/<preset>/bindl`：Denglin SDK runtime（含 `hc2`/`hc3` loader）；
- `build/<preset>/tensorrt_builder`：ONNX builder DLL；
- `build/<preset>/builder_runtime`：TRT 10 parser、plugin、builder resources、CUDA 12 和 cuDNN 9 组件。

启用 Denglin Core 时额外配置：

```powershell
cmake -S . -B build-denglin `
  -DBUILD_DENGLIN_BACKEND=ON `
  -DDENGLIN_SDK_ROOT=E:/denglin-hamming/sdk `
  -DCUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" `
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.16.1.11-cuda-12" `
  -DCUDNN_ROOT="C:/Program Files/NVIDIA/CUDNN/v9.27" `
  -DOpenCV_DIR=E:/opencv/build/x64/vc16/lib
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
dotnet build .\csharp\alanxinn.TensorRtOnnxBuilderDemo -c Release
```

RF-DETR 默认保持 FP32，避免 transformer 分类 logits 在 FP16 下的数值误差；验证时还关闭 TF32。
TensorRT 8 engine 不能直接用于本分支，必须从 ONNX 重新构建。

## 约定

- engine 与 TensorRT、CUDA、GPU 架构相关。
- 模型权重位于 `**/weights/`，不提交 Git。
- 单个模型句柄默认串行使用；并发推理请创建独立句柄。

详细说明见各模块 README。

全部阶段变更及与原始实现的差异见
[IMPLEMENTATION_CHANGELOG.md](IMPLEMENTATION_CHANGELOG.md)。
