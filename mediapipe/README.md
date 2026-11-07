# MediaPipe 手部推理（TensorRT 10）

`mediapipe` 提供完整的图片手部推理：手掌检测、2016 anchors 解码、NMS、
按手腕/中指根旋转裁剪、21 点推理，以及原图坐标/世界坐标还原。支持多手，
默认最多两手；每张图片独立检测，不包含跨帧跟踪。

模型采用 [OpenCV Zoo 的 MediaPipe ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/handpose_estimation_mediapipe)：

| 阶段 | 模型 | FP32 NHWC 输入 |
| --- | --- | --- |
| 手掌检测 | `palm_detection_mediapipe_2023feb.onnx` | `[1,192,192,3]` |
| 手部关键点 | `handpose_estimation_mediapipe_2023feb.onnx` | `[1,224,224,3]` |

下载后校验固定 SHA256，统一 tensor 名称，使用项目现有 native builder 自动生成
engine，成功后记录 ONNX/engine 哈希；缓存匹配时复用 engine。文件保存在
`cache/mediapipe-hands`，不加入 Git。模型的使用和分发许可见上游模型目录。

## 自动下载、转换与测试

本流程需要现有 `build/trt10-release/tensorrt_builder` 和 `bintrt` 运行库。
engine 必须在目标显卡和匹配的 TensorRT 版本上生成；更换显卡或 SDK 时使用新的
`--root`/`--output` 目录或移除对应 engine 后重新转换。

```powershell
python -m pip install -r scripts/requirements-mediapipe-hands.txt
python scripts/prepare_mediapipe_hands.py
python scripts/check_mediapipe_hands.py
# 可选 FP16（输入输出仍为 FP32）
python scripts/check_mediapipe_hands.py --precision fp16
```

`check_mediapipe_hands.py` 自动下载 MediaPipe 官方 `woman_hands.jpg`，
通过现有 `trt_raw_outputs.exe` 执行新 engine，与 ONNX Runtime CPU 对照。
覆盖原图、90° 旋转、镜像、非方形画布、空图、非法输入、max_hands 和仿射变换还原。
报告为 `validation-fp32.json`/`validation-fp16.json`，预览为 `checked-*.jpg`。
默认测试无需编译，但仅验证 engine、现有 native core 和 Python 前后处理。
新 C++/C ABI 的验证需先生成 `alanxinn_cvinfer_mediapipe.dll`，再添加 `--native`，
该模式还检查缓冲区不足、错误恢复和实例释放。

## C++ / C# 接口验证记录

2026-11-05 在 RTX 4060、TensorRT 10.16.1.11、CUDA 12.9 环境完成：

| 检查 | 结果 |
| --- | --- |
| native DLL、C++ contract 编译 | 通过 |
| C# net472 / net6.0 / net8.0 编译 | 通过，0 警告、0 错误 |
| FP32 / FP16 native 与 Python/ONNX Runtime 对照 | 通过 |
| C++ / C ABI：原图、90° 旋转、镜像、宽画布、空图 | 通过，两种精度均覆盖 |
| C++ / C ABI：移动、容量查询、短缓冲区无部分写入、非法参数、错误恢复、max_hands | 通过 |
| C# CvinferDemo：5 类图片、全部 133 个字段对照、图片/JSON 保存 | 两种精度均通过，最大绝对误差为 0 |
| C# CvinferDemo：warmup / iterations / timing、非法参数、native 异常、maxHands | 通过 |

复现时先运行 `scripts/Build-Trt10.ps1`，再执行以下命令（已有 engine 时添加
`--skip-prepare` 可跳过下载和转换）：

```powershell
python scripts/check_mediapipe_hands.py --native --skip-prepare --precision fp32
python scripts/check_mediapipe_hands.py --native --skip-prepare --precision fp16

# 复用上面导出的原始 BGR 和 native 结果，逐字段验证 C++/C ABI 与 C#
$env:PATH = (Resolve-Path build/trt10-release/bintrt).Path + ';' + $env:PATH
build/trt10-release/bin/mediapipe_hands_contract.exe cache/mediapipe-hands fp32
build/trt10-release/bin/mediapipe_hands_contract.exe cache/mediapipe-hands fp16
dotnet build csharp/alanxinn.CvinferDemo -c Release
python scripts/check_mediapipe_hands.py --skip-prepare --precision fp32 --demo csharp/alanxinn.CvinferDemo/bin/Release/net8.0-windows/alanxinn.CvinferDemo.dll
python scripts/check_mediapipe_hands.py --skip-prepare --precision fp16 --demo csharp/alanxinn.CvinferDemo/bin/Release/net8.0-windows/alanxinn.CvinferDemo.dll
```

原始对照数据位于 `cache/mediapipe-hands/interface-fixtures`；native 报告为
`validation-fp32.json` / `validation-fp16.json`，C# 报告为
`validation-csharp-demo-fp32.json` / `validation-csharp-demo-fp16.json`。
C# 运行时验证使用 .NET 8.0.2；net472 / net6.0 仅完成编译验证。

## 图片推理

```powershell
# 不编译新 DLL：使用已有 native core 执行，适合验证，每阶段会启动子进程
python scripts/mediapipe_hands_infer.py --image path/to/hands.jpg --backend raw
# 新 DLL 可用时：常驻 engine 的 native 前后处理，适合实际使用
python scripts/mediapipe_hands_infer.py --image path/to/hands.jpg
```

CLI 自动补齐模型及 engine，默认输出 `cache/mediapipe-hands/result/hands.json`
和 `hands.jpg`。可设置 `--palm-threshold`、`--hand-threshold`、`--nms`、
`--max-hands`、`--precision`、`--root` 和 `--build`。

C# 手部封装属于 `alanxinn.CvinferSharp`，示例统一在 `alanxinn.CvinferDemo` 中运行。
先用准备脚本生成两个 engine，再执行：

```powershell
dotnet run --project csharp/alanxinn.CvinferDemo -c Release -- `
  --type mediapipe-hands `
  --palm-model cache/mediapipe-hands/palm-fp32.engine `
  --model cache/mediapipe-hands/hand-fp32.engine `
  --img cache/mediapipe-hands/woman_hands.jpg `
  --output cache/mediapipe-hands/demo/hands.jpg `
  --max-hands 2 --warmup 5 --iterations 10 --timing
```

Demo 自动部署手部 native DLL，输出关键点/手框图片和同名 JSON；`--json PATH`
可单独指定 JSON 文件。JSON 中 `landmarks` / `worldLandmarks` 各为 63 个浮点数，
按 21 个 xyz 排列。阈值参数为 `--palm-threshold`、`--hand-threshold` 和 `--nms`；
计时为完整手部推理的总耗时。Denglin 配置不包含此 TensorRT 专用功能。

## C++ / C ABI / C#

CMake 默认 `BUILD_MEDIAPIPE_HANDS=ON`，新增目标 `alanxinn_cvinfer_mediapipe`。
C++ 头文件为 `mediapipe/hands.h`，C 头文件为 `mediapipe/hands_api.h`。
这些接口加载两个已生成 engine；模型下载和转换由上面的准备脚本完成。

```cpp
#include "mediapipe/hands.h"
cvinfer::mediapipe::HandDetector detector("palm-fp32.engine", "hand-fp32.engine");
auto hands = detector.infer_bgr(bgr, width, height);
```

```csharp
using var detector = new alanxinn.MediaPipeHandDetector(
    "cache/mediapipe-hands/palm-fp32.engine",
    "cache/mediapipe-hands/hand-fp32.engine", maxHands: 2);
var hands = detector.InferBgr(bgrBytes, width, height);
foreach (var hand in hands)
    Console.WriteLine($"{hand.Label}: {hand.Confidence:F3}, wrist=({hand.Landmarks[0]}, {hand.Landmarks[1]})");
```

C# TensorRT 包已加入新 DLL 的打包规则，Denglin 包排除该 TensorRT 专用封装。
运行时部署 `alanxinn_cvinfer_mediapipe.dll` 与现有 TensorRT/CUDA/OpenCV DLL。
可通过 `ALANXINN_MEDIAPIPE_NATIVE_DIR` 指定 native DLL 所在目录。

输入为连续的 3 通道 uint8 BGR，不自动镜像。结果数组由调用方拥有；每个并发流
使用独立 detector。`box` 为原图 xyxy，`landmarks` 为 21 个 xyz（xy 像素，
z 为手腕相对深度、采用 x 的像素尺度）；`world_landmarks` 为以手部为中心的米制
坐标，xy 已转回原图方向。`handedness` 保留模型原始输出，0 为 Left、1 为 Right；
标签遵循 MediaPipe 模型的镜像输入约定，非镜像相机画面的解剖学左右手需应用自行
交换标签。关键点顺序为手腕 0，拇指 1–4、食指 5–8、中指 9–12、无名指 13–16、
小指 17–20。
