# yolo

YOLO 检测和分割适配层，负责：

- BGR→RGB、letterbox、归一化和 CHW；
- 检测输出解码和 NMS；
- prototype 与 mask 系数合成；
- 坐标还原和 Mask 裁剪；
- CPU/CUDA backend 和耗时统计。

接口：

```text
yolo/include/yolo/yolo.h
yolo/include/yolo/yolo_cuda.h
yolo/src/yolo.cpp
```

底层 TensorRT 推理由 `core::TrtSession` 完成。

## 配置

`yolo::Config` 支持输入尺寸、类别数、输出 tensor 名称、
置信度/NMS/Mask 阈值、class-agnostic NMS 和 backend。

输入尺寸默认从 engine 读取；动态输入使用 profile 0 的 `OPT` shape。
未构建 CUDA backend 时，`auto`/`cpu` 可用，显式 `cuda` 会报错。

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
