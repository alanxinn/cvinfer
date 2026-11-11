# Cvinfer · TensorRT 10

面向 C++ 和 C#／.NET 的 GPU 图像推理与 ONNX 模型转换工具。
当前 `trt10` 分支以 **TensorRT 10.16.1.11 + CUDA 12.9** 为主要环境，
提供 C++、C ABI、C# 封装、Python 导出／推理脚本和统一 C# Demo；另保留可选的登临（Denglin）后端。

项目地址：[alanxinn/cvinfer](https://github.com/alanxinn/cvinfer)。

## 功能概览

| 模块 | 已实现功能 | 说明 |
| --- | --- | --- |
| YOLO11 / YOLO26 | 目标检测、实例分割、人体姿态关键点 | 支持预处理、后处理和耗时统计；[模块说明](yolo/README.md) |
| RF-DETR | 目标检测、实例分割 | 默认使用 FP32 保持数值一致性 |
| SAM2.1 | 图片框选、正负点、纯点提示、连续掩码修正 | 支持删除点、清空提示和缓存图像特征；支持 tiny / small / base-plus / large，默认下载 tiny；[模块说明](sam2/README.md) |
| SAM3 / SAM3.1 | 框与正负点交互分割、文本短语概念分割 | 累积点提示；文本返回多个实例的框、分数和掩码；支持本地 SAM3.1 multiplex 权重转换；[模块说明](sam3/README.md) |
| DINOv3 | ViT / ConvNeXt 单图特征提取 | 输出全局向量、归一化向量、patch 特征和 NCHW 特征图，提供余弦相似度接口；[模块说明](dinov3/README.md) |
| MediaPipe Hands | 手掌检测、多手 21 点推理 | 输出左右手、原图关键点和世界坐标，支持旋转裁剪、模型自动下载；[模块说明](mediapipe/README.md) |
| WeVisOne-VL | 检测、指代、点、计数、OCR、分割、抠图等 11 个 task | 2B 真实图片的 11 个 task 已通过 C++／C ABI／C# 对照；4B 未测；[模块说明](wevisone/README.md)、[验证记录](wevisone/VALIDATION.md) |
| ONNX Builder | ONNX → TensorRT engine | C# 调用原生 builder，无需 trtexec；支持 FP32 / FP16 / INT8 配置、动态输入 profile、workspace 和设备选择；[模块说明](tensorrt_builder/README.md) |

SAM2.1 当前仅支持图片交互，语言描述和视频留待后续。
SAM3 不包含视频传播、记忆及多对象 multiplex；原生文本接口支持英文 ASCII 短语，
其他文本可通过官方 tokenizer 编码接口传入，具体语言效果需另行验证。
MediaPipe 逐图片检测，不包含跨帧跟踪。DINOv3 输出图像特征，分类／检测／分割任务头需另行接入。

## 运行环境

| 项目 | 当前配置 |
| --- | --- |
| 平台 | Windows x64，应用使用 x64 进程 |
| TensorRT | 10.16.1.11（CUDA 12） |
| CUDA Toolkit | 12.9.86 |
| cuDNN | 9.27.0，可选，默认不打包 |
| .NET 类库 | .NET Framework 4.7.2、.NET 6、.NET 8 |
| C# Demo | .NET 8 Windows |
| GPU 资源 | SM86、SM89、SM120，面向 RTX 30／40／50 系列 |

NuGet 部署提供相应 TensorRT／CUDA 运行库，无需另装完整 SDK；机器仍需安装兼容的 NVIDIA 驱动。
模型和 engine 不随 NuGet 分发。engine 应在目标显卡与匹配的 TensorRT 环境中构建，
旧 TensorRT 8 engine 需要从 ONNX 重新转换。

默认不打包 cuDNN、cuBLAS、NVRTC；特定插件需要时显式启用可选依赖。
源码构建需要本机 CUDA、TensorRT、OpenCV、MSVC 和 Ninja。
详细清单见 [DLL 依赖](docs/TRT10_DLL_DEPENDENCIES.md) 和 [TRT10 适配说明](docs/TRT10.md)。

## NuGet 包

当前版本为 **0.2.1**，已生成本地包，尚未远程发布。产物目录：`package/TensorRT10`。
本分支包名在原包名末尾增加 `.TensorRT10`，C# 命名空间仍为 `alanxinn`。
旧版无此后缀的 TensorRT 包不能与本分支原生 DLL 混用。

| 包名 | 用途 | 自动依赖 |
| --- | --- | --- |
| `alanxinn.CvinferSharp.TensorRT10` | YOLO、RF-DETR、SAM2.1、SAM3、DINOv3、MediaPipe 推理接口及原生库 | Core 运行库包 |
| `alanxinn.TensorRtOnnxBuilder.TensorRT10` | ONNX 转 TensorRT engine | 下列全部 8 个运行库包 |

运行库独立打包，由 NuGet 自动还原，无需逐个安装：

主包与运行库版本独立：当前主包为 `0.2.1`，未变化的运行库保留 `0.2.0`。
打包主包时选择源中的最新稳定运行库，生成允许升级的最低版本依赖，不再要求运行库与主包版本相等。

| 运行库包名 | 内容 |
| --- | --- |
| `alanxinn.TensorRtRuntime.Core.TensorRT10` | TensorRT 核心、插件、CUDA runtime 和 VC++ 运行库 |
| `alanxinn.TensorRtRuntime.SM86.TensorRT10` | SM86 builder 资源 |
| `alanxinn.TensorRtRuntime.SM89.TensorRT10` | SM89 builder 资源 |
| `alanxinn.TensorRtRuntime.PTX.Part1.TensorRT10` | PTX 资源分片 1 |
| `alanxinn.TensorRtRuntime.PTX.Part2.TensorRT10` | PTX 资源分片 2 |
| `alanxinn.TensorRtRuntime.PTX.Part3.TensorRT10` | PTX 资源分片 3 |
| `alanxinn.TensorRtRuntime.SM120.Part1.TensorRT10` | SM120 资源分片 1 |
| `alanxinn.TensorRtRuntime.SM120.Part2.TensorRT10` | SM120 资源分片 2 |

转换包还原全部架构资源，不会根据当前显卡选择性下载 NuGet 包。
首次转换自动合并 PTX／SM120 分片并校验哈希，正常使用无需去 GitHub 下载 runtime ZIP。
打包生成的 `tensorrt10.16.1.11-cuda12.9-sm86-sm89-sm120-runtime-win-x64.zip` 用于离线备用。
远程发布时需同时提供两个主包和 8 个依赖包。

从当前仓库根目录向已有 C# 项目安装本地包（替换项目路径）：

```powershell
.\scripts\Install-Cvinfer.ps1 -Project .\YourApp\YourApp.csproj -Version 0.2.1 -Source "$PWD/package/TensorRT10"
# 同时需要 ONNX 转换时添加 -IncludeBuilder
```

安装脚本在消费项目中为运行库添加 `Version="*"` 的直接引用，并重新评估还原，选择最新稳定版本。
普通 NuGet 安装同样支持不同版本的主包／环境包，但其传递依赖默认选择最低满足版本；包自身不能强制消费项目追新。

后续还原需在 NuGet 源中保留该本地目录；更多部署说明见 [C# 推理包说明](csharp/alanxinn.CvinferSharp/README.md)。

## 快速使用

### 图片检测、分割和姿态

统一 Demo 默认引用本地 TRT10 源码项目，需要先完成下方源码构建。以 RF-DETR 分割为例：

```powershell
dotnet run --project .\csharp\alanxinn.CvinferDemo -c Release -- `
  --type rf-detr-seg `
  --model rfdetr-seg-nano.engine `
  --img bus.jpg `
  --warmup 10 --iterations 100 --timing
```

`--type` 支持 `yolo11-det`、`yolo11-seg`、`yolo11-pose`、`yolo26-det`、
`yolo26-seg`、`yolo26-pose`、`rf-detr-det`、`rf-detr-seg` 和 `mediapipe-hands`。
输入尺寸和类别数默认从 engine 读取，`--classes` 可显式限制类别数。
`warmup` 不计入统计；计时默认关闭，添加 `--timing` 开启。
手部模式还需 `--palm-model palm.engine`，用法见 [MediaPipe Hands](mediapipe/README.md)。

### SAM2.1 图片交互分割

准备好包含编码器、解码器和 `bundle.json` 的 engine 目录后：

```csharp
using alanxinn;

using var sam = new Sam2Segmentor("cache/sam2.1/engines");
sam.SetImage(bgrBytes, width, height); // 连续 BGR uint8 数据
sam.SetBox(50, 390, 245, 910);         // 原图 XYXY 坐标
var initial = sam.Predict();
sam.AddPoint(150, 600, positive: true);
sam.AddPoint(240, 850, positive: false);
var refined = sam.Predict(usePreviousMask: true);
```

`Predict(true)` 需要已有成功预测的掩码，输出为原图大小的 0/255 掩码。
权重下载、四种规格导出、删除点和纯点提示见 [SAM2.1 说明](sam2/README.md)。
SAM3 使用 `Sam3Segmentor`，DINOv3 使用 `DinoV3FeatureExtractor`，
分别参见 [SAM3 示例](sam3/README.md) 和 [DINOv3 示例](dinov3/README.md)。
WeVisOne-VL 使用 `WeVisOneModel`，统一 Demo 入口为 `--type wevisone --vl-task <task>`，
模型准备、固定画布及当前验证状态见 [WeVisOne-VL](wevisone/README.md)。WeVisOne-VL 接口与 DLL 使用本次源码生成的 0.2.1 包，旧 0.2.0 包不包含该模块。
这些接口不属于上面 Demo 的普通 `--type` 参数。

### ONNX 转 engine

转换 Demo 调用与 NuGet 转换包相同的 `TensorRtOnnxBuilder` 接口：

```powershell
dotnet run --project .\csharp\alanxinn.TensorRtOnnxBuilderDemo -c Release -- `
  --onnx .\models\model.onnx `
  --engine .\engines\model.engine `
  --precision fp32 --no-tf32 --workspace-mib 2048
```

动态输入配置和 API 见 [转换 Demo](csharp/alanxinn.TensorRtOnnxBuilderDemo/README.md)。
RF-DETR 默认保持 FP32，验证时关闭 TF32；FP16／INT8 的可用性和精度需按具体模型验证。
模型下载／导出依赖各模块的 Python 环境，部分权重需要官方授权；权重不会随安装 NuGet 自动下载。

## 验证状态

已记录的真实 GPU 验证环境为 **RTX 4060 8GB（SM89）、Windows x64**。
SM86／SM120 运行资源已打包，尚无 RTX 30／50 系列实机验证记录。

| 范围 | 已完成验证 |
| --- | --- |
| YOLO11n / YOLO26n | 检测、分割、姿态共 6 个真实模型的 NuGet 推理 |
| RF-DETR nano | 检测和分割 2 个真实模型的 NuGet 推理 |
| SAM2.1 tiny | 官方权重经实际 NuGet builder 转换；9 组提示对照官方 PyTorch，mask IoU 均为 1.0；C# 掩码一致 |
| SAM3.1 | 本地真实权重、1008×1008 FP32，框／正负点／文本与 PyTorch 对照；C# 与 C ABI 掩码一致 |
| DINOv3 | ViT-S/16、ViT-S+/16、ConvNeXt-Tiny 预训练权重经实际 NuGet builder 转换；21 组真实图片对照通过，C# 与 C ABI 输出逐字节一致；已生成相似度示例与 PCA 特征图 |
| MediaPipe Hands | 真实模型 FP32 / FP16，对照 ONNX Runtime，并核对 C++ / C ABI / C# 输出 |
| NuGet 部署 | 两个主包曾在 net472 / net6.0 / net8.0 独立还原并实际运行；验证自动依赖、分片合并和原生 DLL 来源 |

最近一次本地 0.2.0 包消费验证共 14 项通过，包含两项 DINOv3 合成算子检查，
报告位于 `cache/nuget-trt10-validation/validation.json`。转换验证还覆盖同进程
builder → YOLO 推理 → builder 的连续调用，禁止回退到 SDK／外部 ZIP。
上述结果说明本次转换和接口对照通过，不代表所有模型规格、图片或显卡均已验证。

详细记录：[TRT10 模型验证](docs/TRT10_VALIDATION.md)、
[NuGet 独立消费验证](docs/TRT10_NUGET_0.2.0_VALIDATION.md)、
[SAM2.1](sam2/VALIDATION.md)、[SAM3.1](sam3/VALIDATION.md)、
[DINOv3](dinov3/VALIDATION.md)、[MediaPipe](mediapipe/README.md)。
历史记录保留当时的包大小和用例数量，当前产物哈希以 `package/TensorRT10/artifacts.json` 为准。

## 源码构建与打包

```powershell
# 自动初始化 MSVC 与 Ninja，默认构建 TRT10
.\scripts\Build-Trt10.ps1
# 生成两个主包、8 个运行库包和离线 ZIP
.\scripts\Pack-Trt10.ps1
```

Preset 中的 `CUDA_ROOT`、`TENSORRT_ROOT`、`CUDNN_ROOT`、`OpenCV_DIR` 需按本机安装位置调整。
默认 CUDA 架构为 `86;89;120`。存在私有 CUDA 前后处理模块时可使用
`Build-Trt10.ps1 -Preset trt10-cuda-release`；公开 NuGet 不依赖私有 CUDA DLL。

打包脚本在本地验证脚本存在时自动执行模型消费测试；Python 测试／验证脚本按仓库约定不提交，
全新克隆不会自带这些本地脚本，缺失时会明确跳过。统一 C# 接口测试入口和参数见
[C# 测试说明](csharp/README.md#统一接口测试入口)。

| 路径 | 内容 |
| --- | --- |
| `core/tensorrt`、`api` | TensorRT Session、tensor、CUDA stream 和 C ABI |
| `yolo`、`rf_detr` | 检测、实例分割、姿态前后处理 |
| `sam2`、`sam3`、`dinov3`、`mediapipe` | 独立模型接口 |
| `tensorrt_builder` | 原生 ONNX builder |
| `csharp` | .NET 封装、NuGet 规则和 Demo |
| `build/trt10-release/bin` | 原生 API DLL 和 Demo |
| `build/trt10-release/bintrt` | 推理运行库 |
| `build/trt10-release/tensorrt_builder`、`builder_runtime` | 转换 DLL 和运行资源 |
| `package/TensorRT10` | NuGet、离线 ZIP、产物哈希和打包日志 |

## 可选 Denglin 后端

保留 `core/denglin` 和公共 Session 接口，可接入 YOLO、RF-DETR 和 C ABI。
需显式启用 `BUILD_DENGLIN_BACKEND`，配置本机 `DENGLIN_SDK_ROOT` 和相应运行环境。
C# 使用 `CvinferInferenceBackend.Denglin`，Demo 使用 `--backend denglin`。
SAM2.1、SAM3、DINOv3、MediaPipe 是本分支 TensorRT 专用接口，不包含在 Denglin 推理包中。
构建、ONNX／engine 缓存和设备元数据校验见 [Denglin Core](core/denglin/README.md)。

## 使用约定

- 模型权重、ONNX、engine 和本地验证产物不提交 Git，模型许可遵循上游要求。
- 每个模型实例默认串行调用；并发推理应创建独立实例。
- 更换 TensorRT 版本或目标 GPU 后重新构建 engine。
