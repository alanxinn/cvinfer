# YOLO11

YOLO11 检测、分割和 Pose 适配层，负责：

- BGR→RGB、letterbox、归一化和 CHW；
- YOLO11 检测输出解码和 NMS；
- YOLO11 prototype 与 mask 系数合成；
- Pose keypoint 坐标还原；
- 坐标还原和 Mask 裁剪；
- CPU/CUDA backend 和耗时统计。

接口：

```text
yolo/include/yolo/yolo.h
yolo/include/yolo/yolo_cuda.h
yolo/src/yolo.cpp
```

底层推理由 `core::create_inference_session()` 创建公共 Session，
通过 `Config::inference_backend` 选择 TensorRT 或 Denglin。

## 配置

`yolo::Config` 支持输入尺寸、类别数（默认按输出 shape 自动推断）、输出 tensor 名称、
置信度/NMS/Mask 阈值、class-agnostic NMS、预处理/后处理 backend，以及
`InferenceBackend::TensorRT`/`InferenceBackend::Denglin` 推理后端。

输入尺寸默认从 session 读取；Denglin ONNX 未显式提供时使用 `1x3x640x640`。
动态输入使用 profile 0 的 `OPT` shape。
未构建 CUDA backend 时，`auto`/`cpu` 可用，显式 `cuda` 会报错。
Pose 后处理当前使用 CPU；可继续使用 CUDA 前处理。
计时默认关闭，需要统计时将 `Config.enable_timing` 设为 `true`。

## C++ Demo

```text
alanxinn_yolo11_det_core_demo
alanxinn_yolo11_seg_core_demo
```

```powershell
alanxinn_yolo11_det_core_demo yolo11n.engine bus.jpg
alanxinn_yolo11_seg_core_demo yolo11n-seg.engine bus.jpg
```

完整位置参数：

```text
<engine> <image> [input_w] [input_h] [classes]
[preprocessed_output] [result_output] [warmup] [iterations]
[preprocess_backend] [postprocess_backend]
```

C++ Demo 使用位置参数；C# Demo 使用命名参数。
