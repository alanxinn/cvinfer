# core

`core` 提供与模型无关的 TensorRT 推理能力：

- engine 加载和反序列化；
- 输入输出名称、shape、数据类型；
- 动态 shape；
- Host/Device 输入；
- Host/Device 输出；
- CUDA stream、显存和推理执行。

接口：

```text
core/include/core/trt_core.h
core/src/trt_core.cpp
```

## 示例

```cpp
auto session = cvinfer::core::TrtSession::load(engine_path);
const auto input = session.input_names().front();
const auto shape = session.tensor_shape(input);
session.set_shape(input, shape);
session.set_input(input, host_data, host_bytes);
session.infer();
auto outputs = session.outputs();
```

CUDA 后处理：

```cpp
session.set_input_device(input, device_data, device_bytes);
session.set_output_host_copy(false);
session.infer();
const void* output = session.device_output(output_name);
```

动态输入默认使用 TensorRT optimization profile 0 的 `OPT` shape。
`core` 不包含 YOLO11、RF-DETR 或 OpenCV 业务逻辑。

构建产物：

```text
alanxinn_cvinfer_core.lib
```
