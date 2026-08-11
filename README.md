# alanxinn_cvinfer

TensorRT 推理项目，包含：

- `core`：TensorRT engine、输入输出 tensor、CUDA stream。
- `yolo`：YOLO11/YOLO26 检测、分割、Pose 前后处理。
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

需使用与 CUDA、TensorRT 和 GPU 架构匹配的驱动及运行库。

下载 NuGet Package，自带对应运行环境，需手动安装 nvidia 驱动

## NuGet Packages

| Package | Version | Description |
| --- | --- | --- |
| [alanxinn.CvinferSharp](https://www.nuget.org/packages/alanxinn.CvinferSharp) | 0.1.2 | C# TensorRT 推理封装，支持 YOLO11/YOLO26/RF-DETR 检测、分割和 Pose |
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
  --postprocess auto
```

输入尺寸和类别数默认从 engine 输出自动读取；`--classes` 仅用于显式限制类别数。
`warmup` 不计入统计，`iterations` 为正式统计次数。

RF-DETR 当前前处理、后处理使用 CPU；TensorRT 推理仍在 GPU 上执行。
显式选择 RF-DETR 的 `cuda` 前后处理会返回不支持错误。

## 构建

普通构建：

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
构建产物位于 `build/<preset>/bin`，ONNX builder 位于
`build/<preset>/tensorrt_builder`。

## ONNX 转换

```powershell
dotnet add package alanxinn.TensorRtOnnxBuilder --version 0.1.3
```

RF-DETR 在 TensorRT 8.6 下使用 FP32 构建；builder 会自动避免错误的 FP16 输出。
YOLO26 end2end engine 的后处理使用 `cpu`/`auto`；Pose 后处理使用 CPU。
TensorRT 8.6 解析原始 YOLO26 ONNX 可能在 `ReduceMax/TopK` 失败，
需使用已修正版 ONNX 或升级 TensorRT。

## 约定

- engine 与 TensorRT、CUDA、GPU 架构相关。
- 模型权重位于 `**/weights/`，不提交 Git。
- 单个模型句柄默认串行使用；并发推理请创建独立句柄。

详细说明见各模块 README。
