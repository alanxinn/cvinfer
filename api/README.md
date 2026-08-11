# api

`api` 将 `core`、YOLO11 和 RF-DETR 封装为稳定 C ABI，供 C# 或其他语言调用。

## 主要接口

```cpp
cvinfer_model_config_v2_init(&config);
cvinfer_model_create_v2(engine, model_type, &config, &model);
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
alanxinn_cvinfer.dll
alanxinn_cvinfer.lib
```

C# 通过以下方式加载：

```csharp
[DllImport("alanxinn_cvinfer.dll", CallingConvention = CallingConvention.Cdecl)]
```

CUDA backend 为可选 DLL。未部署时 `auto`、`cpu` 可用，显式 `cuda` 会报错。

详细 C# 用法见 [csharp/README.md](../csharp/README.md)。
