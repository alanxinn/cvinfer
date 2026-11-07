# C# 项目

目录：

```text
alanxinn.CvinferSharp/              C# P/Invoke 封装
alanxinn.CvinferDemo/               推理 Demo
alanxinn.TensorRtOnnxBuilder/       ONNX 转 engine 类库
alanxinn.TensorRtOnnxBuilderDemo/   转换 Demo
```

## 运行环境

- Windows x64
- NVIDIA 驱动
- TensorRT backend 使用与 engine 匹配的 TensorRT/CUDA/OpenCV；Denglin backend
  使用匹配的 Denglin SDK/device runtime
- 应用使用 x64 平台

## 推理 Demo

```powershell
alanxinn.CvinferDemo.exe `
  --type yolo11-det `
  --model yolo11n.engine `
  --img bus.jpg `
  --warmup 100 `
  --iterations 1000 `
  --timing
```

`--type` 只支持：

```text
yolo11-det
yolo11-seg
yolo11-pose
yolo26-det
yolo26-seg
yolo26-pose
rf-detr-det
rf-detr-seg
mediapipe-hands
```

其他参数：

```text
--model PATH
--img PATH
--output PATH
--classes N                 可选；默认按 engine 输出自动推断
--backend tensorrt|denglin  推理引擎后端（Denglin 需启用 native backend）
--preprocess auto|cpu|cuda
--postprocess auto|cpu|cuda
--warmup N
--iterations N
--timing                    可选；开启预处理、推理和后处理计时
```

输入尺寸和类别数默认从 engine 输出读取；分割模式输出叠加图，Pose 模式输出关键点。非 benchmark 模式还保存
`<output>_mask_<index>.pgm`。

RF-DETR 分割 engine 输出为 `dets`、`labels`、`masks`。

MediaPipe 手部接口 `MediaPipeHandDetector` 已包含在 `alanxinn.CvinferSharp` 中，
通过同一个 Demo 使用（仅 TensorRT）：

```powershell
dotnet run --project csharp/alanxinn.CvinferDemo -c Release -- `
  --type mediapipe-hands `
  --palm-model cache/mediapipe-hands/palm-fp32.engine `
  --model cache/mediapipe-hands/hand-fp32.engine `
  --img cache/mediapipe-hands/woman_hands.jpg `
  --output cache/mediapipe-hands/demo/hands.jpg
```

默认同时保存 `hands.json`，包含手框、置信度、左右手、21 个像素关键点和世界坐标。
可用 `--json PATH` 指定 JSON 文件，`--palm-threshold` / `--hand-threshold` 设置阈值，
`--nms` 设置 NMS 阈值，`--max-hands` 限制手数；支持 `--warmup`、`--iterations`
和 `--timing`（整次手部推理耗时）。下载模型、生成 engine 和 Demo 测试见
[MediaPipe 说明](../mediapipe/README.md)。

计时默认关闭，添加 `--timing` 后才输出耗时统计。

## 统一接口测试入口

接口集成测试统一使用 `alanxinn.CvinferDemo`，无需独立测试项目。
TensorRT 配置支持以下测试模式；参数错误返回 2，运行失败返回 1，全部通过返回 0。

```powershell
dotnet build csharp/alanxinn.CvinferDemo -c Release
# 同一进程 ONNX builder → YOLO → builder；可同时指定两个随机 tiny 模型做接口检查
dotnet csharp/alanxinn.CvinferDemo/bin/Release/net8.0-windows/alanxinn.CvinferDemo.dll `
  --test integration --fixtures cache/trt10-contract `
  --model cache/trt10-validation/yolo11n.engine `
  --sam3-model cache/sam3-tiny/engines `
  --dinov3-model cache/dinov3-tiny/vit/engine/model.engine `
  --json cache/trt10-contract/validation-cvinferdemo.json

# SAM3.1 真实图片与 native mask 逐字节对照
dotnet csharp/alanxinn.CvinferDemo/bin/Release/net8.0-windows/alanxinn.CvinferDemo.dll `
  --test sam3-real --model cache/sam31/engines --img cache/trt10-models/bus.jpg `
  --reference cache/sam31/real-validation-images --json cache/sam31/validation-cvinferdemo.json
```

`--fixtures` 目录需包含 `int64.onnx` 和 810×1080 的原始 BGR `bus.bgr`。
可选的 `--sam3-model` / `--dinov3-model` 是现有随机 tiny 测试模型，验证接口、
提示状态、归一化、结果所有权和释放行为，不代表真实模型质量。
真实 SAM3.1 的模型和参考 mask 准备流程见 [SAM3 验证说明](../sam3/VALIDATION.md)。
MediaPipe 仍使用上面的 `--type mediapipe-hands` 图片入口，通过
`scripts/check_mediapipe_hands.py --demo ...` 自动调用 Demo 进行数值对照。

NuGet 验证同样调用 `CvinferDemo --test nuget --reference <测试请求.json>`。
运行 `scripts/Pack-Trt10.ps1` 会打包并自动执行
`scripts/check_trt10_nuget.py`：按包哈希隔离还原目录，将实际 Demo 切换为纯
PackageReference 模式，检查包内 DLL、运行时依赖及 YOLO/RF-DETR、SAM2/SAM3、
DINOv3 和 MediaPipe FP32/FP16。包测试禁止 ProjectReference 和手工复制 native DLL。
可单独复测：

```powershell
python scripts/check_trt10_nuget.py --packages package/TensorRT10
```

## C# 调用

```csharp
using alanxinn;

var config = new CvinferConfig
{
    ClassCount = 0,
    InferenceBackend = CvinferInferenceBackend.TensorRt,
    PreprocessBackend = CvinferComputeBackend.Auto,
    PostprocessBackend = CvinferComputeBackend.Auto
};

using var model = new CvinferModel(
    "rfdetr-seg-nano.engine",
    CvinferModelType.RfDetrSegmentation,
    config);

using var result = model.InferBgr(bgrBytes, width, height, 3);
var segmentation = result.GetSegmentation(0);
var mask = result.CopyMask(0);
```

将 `InferenceBackend` 设置为 `CvinferInferenceBackend.Denglin` 后，native
层会通过公共 Session 工厂加载 Denglin ONNX/engine。未启用 Denglin 或缺少
运行时 DLL 时，模型创建会返回可读错误。

## 构建和打包

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
.\csharp\alanxinn.CvinferSharp\pack-both.ps1 -Configuration Release
```

会生成两个包：

- `alanxinn.CvinferSharp`：TensorRT/CUDA/OpenCV runtime；
- `alanxinn.CvinferSharp.Denglin`：Denglin SDK runtime。

包输出位置：

```text
alanxinn.CvinferSharp/bin/TensorRT/Release/
alanxinn.CvinferSharp/bin/Denglin/
```

两个包的 native DLL 都位于 `runtimes/win-x64/native/`。应用只需引用
对应的 NuGet 包，不需要手工复制 DLL。Denglin 包另外通过包内构建目标
自动部署 `runtimes/win-x64/hc2`、`runtimes/win-x64/hc3`，供 SDK 的
`hcrt.dll` 相对路径加载。

TensorRT 包加载 `alanxinn_cvinfer_trt.dll`，Denglin 包加载
`alanxinn_cvinfer_denglin.dll`。两个包可以同时引用，运行时通过
`CvinferInferenceBackend` 选择后端；两套 native API DLL 不能互相替代，
也不能改回旧的无后缀 `alanxinn_cvinfer.dll`。

当前 Demo 已同时引用两个 0.1.4 包，直接通过 `--backend` 选择后端：

```powershell
dotnet run --project .\csharp\alanxinn.CvinferDemo `
  -- --type yolo11-det --backend tensorrt --model yolo11n.engine --img bus.jpg

dotnet run --project .\csharp\alanxinn.CvinferDemo `
  -- --type yolo11-det --backend denglin --model yolo11n.denglin.cache.raw.engine --img bus.jpg
```

修改同版本包后，清理 NuGet 缓存再还原。

ONNX 转换见：

- [alanxinn.TensorRtOnnxBuilder](alanxinn.TensorRtOnnxBuilder/README.md)
- [转换 Demo](alanxinn.TensorRtOnnxBuilderDemo/README.md)

SAM3 使用独立的 `Sam3Segmentor`，参见 [SAM3 接口与验证范围](../sam3/README.md)。
DINOv3 使用 `DinoV3FeatureExtractor` 提取全局／patch／NCHW 特征，支持图像余弦相似度，
参见 [DINOv3 图像推理](../dinov3/README.md)。
本分支 Demo 默认引用本地 TRT10 项目。
