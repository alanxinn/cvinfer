# core：TensorRT 核心库

## 职责

`core` 提供与具体模型无关的 TensorRT 能力：

- engine 文件读取和反序列化；
- `IRuntime`、`ICudaEngine`、`IExecutionContext` 生命周期管理；
- 输入输出 tensor 名称、shape 和数据类型查询；
- 动态输入 shape 设置；
- Host/Device 输入；
- Device 输出访问和 Host 输出复制；
- CUDA stream、显存缓冲区和推理执行；
- TensorRT 异常转换为统一 C++ 异常。

核心接口位于：

```text
core/include/core/trt_core.h
core/src/trt_core.cpp
```

## 依赖边界

```text
alanxinn_cvinfer_core
        ↓
TensorRT + CUDA
```

`core` 不包含 YOLO 的 letterbox、NMS、类别、Mask 或图像绘制逻辑，也不依赖 OpenCV 图像对象。

`core` 只负责 TensorRT 推理和 CUDA 设备资源管理；前处理、后处理是否使用
CPU 或 CUDA 由上层模型适配库决定。当前 Windows 构建产物为
`alanxinn_cvinfer_core.lib`，不会单独生成 DLL。

对于输入 tensor，`tensor_shape()` 在 engine 提供固定 shape 时返回固定尺寸；
如果输入是动态 shape，则自动读取 optimization profile 0 的 `OPT` 尺寸。
上层模型可以据此完成预处理和 `set_shape`，无需额外传入输入宽高。

## 基本调用

```cpp
auto session = cvinfer::core::TrtSession::load(engine_path);
const auto input = session.input_names().front();
const auto shape = session.tensor_shape(input);

session.set_shape(input, shape);
session.set_input(input, host_data, host_bytes);
session.infer();

auto outputs = session.outputs();
```

CUDA 后处理场景可使用：

```cpp
session.set_input_device(input, device_data, device_bytes);
session.set_output_host_copy(false);
session.infer();
const void* output = session.device_output(output_name);
```

外部 device pointer 的生命周期由调用方负责，必须持续有效到本次推理和后处理完成。

## 构建产物

```text
alanxinn_cvinfer_core.lib
```

`core` 不依赖 `yolo`、`api` 或私有 CUDA backend。新增模型时，如果模型不属于
YOLO，可以直接依赖 `core`，实现自己的输入预处理和输出解码。
