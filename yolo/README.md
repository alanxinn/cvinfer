# yolo：YOLO 模型适配库

## 职责

`yolo` 将 YOLO11 检测和分割模型统一到同一套适配层中：

- BGR 图像预处理；
- letterbox、RGB 转换、归一化和 CHW 排布；
- 检测输出解码；
- 分割检测分支和 prototype 分支解析；
- 类别 NMS 和 class-agnostic NMS；
- Mask 系数与 prototype 合成；
- 坐标逆变换、Mask 裁剪和结果输出；
- CPU/CUDA 前处理与后处理选择；
- 推理耗时统计。

源码位置：

```text
yolo/include/yolo/yolo.h
yolo/src/yolo.cpp
yolo/src/yolo_cuda_loader.cpp
yolo/include/yolo/yolo_cuda.h
```

## 调用关系

```text
Detector / Segmentor
        ↓
preprocess → core::TrtSession → decode/NMS/Mask
```

`Detector` 和 `Segmentor` 不创建 TensorRT runtime 或 context，所有底层推理由 `core` 完成。
公开 YOLO 适配层只包含 CPU 前后处理和 CUDA backend 接口。CUDA 实现位于私有
backend，通过 `ALANXINN_CUDA_BACKEND_DIR` 接入。

## 配置

`yolo::Config` 支持：

- 输入宽高和类别数量；
- 输入、检测输出、prototype tensor 名称；
- 置信度、NMS、Mask 阈值；
- class-agnostic NMS；
- `Auto`、`Cpu`、`Cuda` 前处理和后处理 backend。

`Auto` 保持默认兼容行为；需要 GPU 前处理或后处理时显式设置 `Cuda`。
未启用私有 backend 时，`Auto` 和 `Cpu` 可用，显式选择 `Cuda` 会返回错误。

输入宽高默认从 engine 的输入 tensor 自动读取。固定 shape engine 使用固定尺寸；
动态 shape engine 使用 TensorRT optimization profile 0 的 `OPT` 尺寸。只有在
确实需要覆盖 engine 默认尺寸时，才通过 C++ `Config` 显式设置宽高。

## Demo

构建目标：

```text
alanxinn_yolo11_det_core_demo
alanxinn_yolo11_seg_core_demo
```

示例：

```powershell
alanxinn_yolo11_det_core_demo yolo11n.engine bus.jpg 640 640 80 pre.jpg result.jpg
alanxinn_yolo11_seg_core_demo yolo11n-seg.engine bus.jpg 810 1080 80 pre.jpg result.jpg
```

参数顺序为：

```text
<engine> <image> [input_w] [input_h] [classes]
[preprocessed_output] [result_output] [warmup] [iterations]
[preprocess_backend] [postprocess_backend]
```

backend 支持 `auto`、`cpu` 和 `cuda`。C++ Demo 使用位置参数；C# Demo 使用命名参数，两者参数格式不兼容。

追加 warmup 和 iterations 后进入 benchmark 模式，只保存一次最终结果图：

```powershell
alanxinn_yolo11_seg_core_demo yolo11n-seg.engine bus.jpg 810 1080 80 pre.jpg result.jpg 100 1000
```

benchmark 模式会跳过中间预处理图保存，只统计 warmup 之后的正式 iterations。输出包括前处理、TensorRT 推理、后处理和总耗时的平均值、最小值和最大值。

非 benchmark 模式会保存预处理图和最终结果图；benchmark 模式只保存最终结果图。

## 构建产物

```text
alanxinn_cvinfer_yolo.lib
alanxinn_cvinfer_cuda.dll
```

`alanxinn_cvinfer_cuda.dll` 是可选的 CUDA 前后处理加速库。不存在时，CPU/Auto
路径不受影响；显式选择 CUDA backend 会报错，不会静默回退。启用私有 backend
构建后，DLL 输出到构建目录 `bin/`；NuGet 部署时，该 DLL 与主 DLL 一起位于
`runtimes/win-x64/native`。
