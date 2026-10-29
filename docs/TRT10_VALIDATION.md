# TRT10 本机验证记录

日期：2026-10-28。GPU：RTX 4060 8GB（SM89），驱动 616.92。
CUDA runtime API 返回 12090；cuDNN runtime API 返回 92700。

| 模型 | 精度 | 对象数 | 非空掩码 | 有效关键点 | CPU/CUDA/对照检查 |
| --- | --- | ---: | ---: | ---: | --- |
| yolo11n | fp16 | 5 | 0 | 0 | PASS |
| yolo11n-seg | fp16 | 5 | 5 | 0 | PASS |
| yolo11n-pose | fp16 | 4 | 0 | 50 | PASS |
| yolo26n | fp16 | 5 | 0 | 0 | PASS |
| yolo26n-seg | fp16 | 4 | 4 | 0 | PASS |
| yolo26n-pose | fp16 | 4 | 0 | 47 | PASS |
| rfdetr-nano | fp32 | 7 | 0 | 0 | PASS |
| rfdetr-seg-nano | fp32 | 8 | 8 | 0 | PASS |

CPU/CUDA 指前后处理模式；全部模型的引擎均在 NVIDIA GPU 上执行。
YOLO26 和 pose 的 NVIDIA CUDA 模式使用 CUDA 前处理与 CPU 后处理；YOLO11 det/seg 和 RF-DETR 使用 CUDA 前后处理。

YOLO11 原始 tensor 使用相对 RMS < 1%、置信度绝对误差 < 0.05，并检查高置信度分类一致。
YOLO26 TopK 输出是无序检测集合，按类别和 IoU 匹配；本次最小匹配 IoU > 0.996，分类一致。
RF-DETR 使用 FP32、关闭 TF32；与 ONNX Runtime 的 logits 最大绝对误差 < 0.00024。
RF-DETR 分割 mask logits 最大绝对误差 0.00661；完整误差指标见 JSON。

底层动态 shape、host/device 输入、输出回拷开关、缓冲复用、非法 profile 和 INT64 测试均通过。
C# 封装 .NET 4.7.2/6/8 编译通过；YOLO11/YOLO26/RF-DETR C# 实际推理，以及同一进程 builder→infer→builder 通过。

原始日志、图片、模型和引擎不进入 Git，位于 `cache/trt10-models`、`cache/trt10-validation`、`cache/trt10-contract`。
复现命令见 [TRT10.md](TRT10.md)。

2026-10-29：SAM3 六个微型随机权重算子图的 PyTorch/ONNX/TensorRT、C++ 提示状态和 C# 调用测试通过。
后续已取得本地 SAM3.1 权重：完整 1008×1008 FP32 的框、正负点、文本与 C# 实测通过；
详见 [SAM3.1 真实模型验证](../sam3/VALIDATION.md)。此前微型测试仍保留为算子／接口回归用例。
