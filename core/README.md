# core

`core` 按推理后端拆分为两个目录：

- `tensorrt`：当前默认的 TensorRT engine 推理实现。
- `denglin`：登临（Denglin）dlNNE 后端，默认关闭，可通过
  `BUILD_DENGLIN_BACKEND` 和 `DENGLIN_SDK_ROOT` 启用；根目录的 Ninja preset
  已为双后端构建预设为开启。

TensorRT 后端提供与模型无关的推理能力：

- engine 加载和反序列化；
- 输入输出名称、shape、数据类型；
- 动态 shape；
- Host/Device 输入；
- Host/Device 输出；
- CUDA stream、显存和推理执行。

接口：

```text
core/tensorrt/include/core/trt_core.h
core/tensorrt/src/trt_core.cpp
```

模型适配层使用统一 Session 接口：

```text
core/include/core/inference_session.h
core/src/inference_session.cpp
```

通过 `InferenceSessionOptions::backend` 选择 TensorRT 或 Denglin，
`create_inference_session()` 会创建对应实现。Denglin 未启用时选择该后端会
返回明确的运行时错误。

共享 Tensor 类型位于 `core/include/core/tensor.h`；Denglin Core 位于：

```text
core/denglin/include/core/denglin_core.h
core/denglin/src/denglin_core.cpp
```

## 示例

统一 Session（模型适配层使用）：

```cpp
cvinfer::core::InferenceSessionOptions options;
options.backend = cvinfer::core::InferenceBackend::Denglin;
options.input_name = "images";
options.input_shape = {1, 3, 640, 640};
auto session = cvinfer::core::create_inference_session("model.onnx", options);
```

直接使用 TensorRT Core：

```cpp
auto session = cvinfer::core::TrtSession::load(engine_path);
const auto input = session.input_names().front();
const auto shape = session.tensor_shape(input);
session.set_shape(input, shape);
session.set_input(input, host_data, host_bytes);
session.infer();
auto outputs = session.outputs();
```

设备输入输出：

```cpp
session.set_input_device(input, device_data, device_bytes);
session.set_output_host_copy(false);
session.infer();
const void* output = session.device_output(output_name);
```

动态输入默认使用 TensorRT optimization profile 0 的 `OPT` shape。
`core` 不包含 YOLO11、RF-DETR 或 OpenCV 业务逻辑。

Denglin 的 `infer()` 在内部依次把 H2D、`ExecutionContext::Enqueue` 和
D2H 复制提交到同一个 CUDA stream，最后同步该 stream 后返回。这样保留
现有 Session 的同步调用契约，同时不再使用同步 `ExecuteV2` 或同步输出复制。
Denglin Core 还提供 `serialize_engine()` 原始 engine 导出，以及带 SDK、设备、
模型 hash 和构建参数校验的 `DenglinCacheMode::ReadWrite/ReadOnly` 缓存。

Denglin 阶段三增加动态输入 profile、`set_shape()` 运行时 shape 切换、
`set_input_device()`/`set_output_device()` device I/O，以及
`infer_async()`/`enqueue()` 和 `synchronize()`。支持 tensor-address API 的
engine 使用 `SetTensorAddress` + `EnqueueV3`，旧 engine 自动使用 binding
指针和 `EnqueueV2`/`Enqueue`。异步调用返回后，调用方必须等
`synchronize()` 再复用输入或读取 host output。

构建产物：

```text
alanxinn_cvinfer_core_tensorrt.lib
alanxinn_cvinfer_core_denglin.lib  # BUILD_DENGLIN_BACKEND=ON 时生成
alanxinn_cvinfer_core_session_tensorrt.lib
alanxinn_cvinfer_core_session_denglin.lib
```

旧目标名 `alanxinn_cvinfer_core` 和 `alanxinn_cvinfer_core_session` 保留为
TensorRT CMake alias，以兼容已有调用方；新代码应按后端链接带后缀的 target。
