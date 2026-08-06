# api：C ABI 动态库

## 职责

`api` 将 C++ 的 `core + yolo` 能力封装为稳定的 C ABI，供 C#、其他语言和外部 C/C++ 应用调用。

源码位置：

```text
api/include/cvinfer/cvinfer_api.h
api/src/cvinfer_api.cpp
```

## 依赖关系

```text
alanxinn_cvinfer.dll
        ↓
alanxinn_cvinfer_yolo
        ↓
alanxinn_cvinfer_core
```

启用私有 CUDA backend 时，`alanxinn_cvinfer_yolo` 通过延迟加载连接：

```text
alanxinn_cvinfer_yolo
        ↓
alanxinn_cvinfer_cuda.dll
```

API 层负责：

- 模型句柄创建和销毁；
- 检测/分割模型类型选择；
- C 结构体与 C++ 结果转换；
- 错误码和线程局部错误信息；
- Mask 缓冲区复制；
- 前处理、推理、后处理耗时返回；
- C ABI 下的 UTF-8 路径和固定宽度字段。

## 主要接口

```cpp
cvinfer_model_config_v2_init(&config);
cvinfer_model_create_v2(engine, model_type, &config, &model);
cvinfer_model_infer_bgr(model, bgr, width, height, channels, &result);
cvinfer_result_count(result);
cvinfer_result_get_detection(result, index, &detection);
cvinfer_result_get_segmentation(result, index, &segmentation);
cvinfer_result_copy_mask(result, index, buffer, bytes, &copied);
cvinfer_result_destroy(result);
cvinfer_model_destroy(model);
```

所有导出函数使用 `extern "C"` 和 `__cdecl`。失败时返回状态码，并通过 `cvinfer_last_error()` 获取错误文本。

## 构建产物

```text
alanxinn_cvinfer.dll
alanxinn_cvinfer.lib
```

`alanxinn_cvinfer_cuda.dll` 由私有 backend 单独生成，不属于普通 API 构建的必需产物。

C API 导出函数名仍使用 `cvinfer_*`，用于保持接口字段和既有调用约定稳定；动态库和导入库统一使用 `alanxinn` 前缀。

## C# 调用

C# 类库通过以下名称加载动态库：

```csharp
[DllImport("alanxinn_cvinfer.dll", CallingConvention = CallingConvention.Cdecl)]
```

当配置 `Cuda` 前处理或后处理时，程序会动态加载可选的
`alanxinn_cvinfer_cuda.dll`。Windows 下优先从
`alanxinn_cvinfer.dll` 自身所在目录加载，因此 NuGet 的
`runtimes/win-x64/native` 目录可以直接使用，不要求复制到应用程序根目录。
主库必须在构建时启用私有 CUDA backend 才能调用其中的 CUDA 符号；
该 DLL 不存在或主库未包含 backend 时，CPU/Auto 路径仍可使用，但显式
CUDA 配置会返回错误。

具体 C# 配置、Demo 参数和 benchmark 方式请参阅 [csharp/README.md](../csharp/README.md)。
