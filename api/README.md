# api

`api` 将 `core`、YOLO11 和 RF-DETR 封装为稳定 C ABI，供 C# 或其他语言调用。

模型配置的 `inference_backend` 用于选择推理引擎后端：
`CVINFER_INFERENCE_BACKEND_TENSORRT`（默认）或
`CVINFER_INFERENCE_BACKEND_DENGLIN`。启用 `BUILD_DENGLIN_BACKEND` 后，
YOLO/RF-DETR 会通过公共 Session 工厂加载对应后端；未启用 Denglin 时选择
该后端返回 `CVINFER_STATUS_UNSUPPORTED`。

C# 侧对应 `CvinferInferenceBackend.TensorRt` 和
`CvinferInferenceBackend.Denglin`，配置会原样传入
`cvinfer_model_config_v2_t.inference_backend`。
TensorRT 后端加载 TensorRT engine；Denglin 后端加载 ONNX 或 Denglin 自有
engine/cache，二者文件格式不可混用。

## 主要接口

```cpp
cvinfer_model_config_v2_init(&config);
cvinfer_model_create_v2(model_path, model_type, &config, &model);
cvinfer_model_infer_bgr(model, bgr, width, height, channels, &result);
cvinfer_result_count(result);
cvinfer_result_get_detection(result, index, &detection);
cvinfer_result_get_segmentation(result, index, &segmentation);
cvinfer_result_copy_mask(result, index, buffer, bytes, &copied);
cvinfer_result_get_pose(result, index, &pose);
cvinfer_result_copy_keypoints(result, index, buffer, bytes, &copied);
cvinfer_result_destroy(result);
cvinfer_model_destroy(model);
```

失败时返回状态码，并通过 `cvinfer_last_error()` 获取错误信息。

类别数默认从 engine 输出 shape 自动推断；只有需要限制类别范围时才设置
`class_count`。

模型类型：

```text
CVINFER_MODEL_YOLO11_DET
CVINFER_MODEL_YOLO11_SEG
CVINFER_MODEL_YOLO11_POSE
CVINFER_MODEL_RF_DETR_DET
CVINFER_MODEL_RF_DETR_SEG
```

## 构建产物

```text
alanxinn_cvinfer_trt.dll
alanxinn_cvinfer_trt.lib
alanxinn_cvinfer_denglin.dll       # BUILD_DENGLIN_BACKEND=ON 时生成
alanxinn_cvinfer_denglin.lib
```

C# 按 `CvinferInferenceBackend` 动态加载对应 native API；C/C++ 调用方应按
模型后端显式链接匹配的 DLL。两个 C# NuGet 包可以同时引用，但每个模型句柄
仍必须使用与其 `inference_backend` 对应的 native API。

```csharp
// TensorRT: alanxinn_cvinfer_trt.dll
// Denglin:  alanxinn_cvinfer_denglin.dll
```

TensorRT runtime 输出到 `bintrt`，Denglin runtime 输出到 `bindl`。两套 native
DLL 不再共享 CUDA/设备 runtime，避免 TensorRT 的 `cudart` 与 Denglin 的
`curt` 在同一 native 模块中冲突。

CUDA backend 为可选 DLL。未部署时 `auto`、`cpu` 可用，显式 `cuda` 会报错。

计时默认关闭；模型创建后可调用
`cvinfer_model_set_timing_enabled(model, 1)` 开启计时，结果中的
`cvinfer_timing_t` 才会填充耗时数据。

详细 C# 用法见 [csharp/README.md](../csharp/README.md)。
