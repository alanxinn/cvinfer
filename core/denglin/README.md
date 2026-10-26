# Denglin core backend

此目录提供独立的登临（Denglin）dlNNE Core。默认通过
`BUILD_DENGLIN_BACKEND=OFF` 保持 TensorRT-only 构建；启用 Denglin 时，
通过 `DENGLIN_SDK_ROOT` 指向 `E:\denglin-hamming\sdk`。根目录的 Ninja
preset 已将该选项打开。

当前阶段实现：

- ONNX Parser + Builder 构建 Denglin engine；
- Denglin engine 反序列化；
- 输入/输出名称、shape、dtype 查询；
- Host/Device 输入和多输出 Host/Device 输出；
- 单 CUDA stream 上的异步 H2D、`ExecutionContext::Enqueue`/`EnqueueV3`、D2H；
- `infer()` 返回前同步 stream，保证 Host 输出可直接读取；
- Denglin engine 原始序列化/反序列化；
- 带 SDK、设备、模型 hash、shape、dtype、cluster 和 builder 配置校验的缓存；
- 动态输入 shape 的 min/opt/max optimization profile；
- device input/device output 直连和 `EnqueueV3` tensor-address API；
- `infer_async()`/`enqueue()` + `synchronize()` 异步执行接口。

执行接口按 engine 的 batch 模式选择：SDK `nnexec` 默认生成的隐式 batch
engine 使用 `Enqueue(batch_size, bindings, stream, nullptr)`，显式 batch
engine 使用 `EnqueueV3(stream)`。两条路径均支持 Host/Device 输入输出；
隐式 batch 的 binding 数组由 session 持有，覆盖异步执行期间的生命周期。

YOLO/RF-DETR 不直接依赖本目录的具体类，而是通过
`core::create_inference_session()` 和 `Config::inference_backend` 选择
Denglin；C ABI/C# 的 `Denglin` 选项也沿用这条链路。

核心接口：

```text
core/denglin/include/core/denglin_core.h
core/denglin/src/denglin_core.cpp
```

构建示例：

```text
cmake -S . -B build-denglin `
  -DBUILD_DENGLIN_BACKEND=ON `
  -DDENGLIN_SDK_ROOT=E:/denglin-hamming/sdk `
  -DCUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" `
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6_cuda-11.8" `
  -DOPENCV_ROOT=E:/opencv/build
```

也可以直接使用根目录的 `ninja-msvc-release` preset；它已包含上述 CUDA、
TensorRT、OpenCV 和 Denglin SDK 路径，并默认启用 Denglin。

TensorRT engine 与 Denglin engine 不兼容；ONNX 解析时默认显式注册
`kNCHW`，因为当前 YOLO/RF-DETR 前处理输出是 NCHW。

调用示例：

```cpp
cvinfer::core::DenglinSessionOptions options;
options.input_name = "images";
options.input_shape = {1, 3, 640, 640};

auto session = cvinfer::core::DenglinSession::load("model.onnx", options);
session.set_input("images", host_input, input_bytes);
session.infer();
auto outputs = session.outputs();
```

动态 shape 和异步 device I/O：

```cpp
cvinfer::core::DenglinSessionOptions dynamic_options;
dynamic_options.input_name = "images";
dynamic_options.input_shape = {-1, 3, -1, -1};
dynamic_options.max_batch_size = 4;
dynamic_options.optimization_profiles.push_back({
    {1, 3, 320, 320}, {1, 3, 640, 640}, {4, 3, 1280, 1280}});
auto dynamic_session = cvinfer::core::DenglinSession::load(
    "model.onnx", dynamic_options);
dynamic_session.set_shape("images", {2, 3, 640, 640});
dynamic_session.set_input_device("images", device_input, input_bytes);
dynamic_session.set_output_host_copy(false);
dynamic_session.enqueue();
// Device input/output pointers remain owned by the caller until synchronize().
dynamic_session.synchronize();
auto output_ptr = dynamic_session.device_output("output");
```

Core 统一使用 SDK 的 `SetTensorAddress`/`EnqueueV3` Tensor API，不提供旧
执行接口回退。不支持 IO Tensor API 的旧 engine 会
直接报告不兼容。调用 `set_output_device()` 可提供调用方自己的 device output
buffer；未提供时 Core 会按当前 shape 自动分配显存。shape 改变后必须重新
调用 `tensor_shape()` 或在执行后重新查询 `device_output_shape()`。

纯 Tensor API 要求 engine 使用 Explicit Batch 构建。旧版本生成的隐式 Batch
`.raw.engine` 不能继续复用，需从 ONNX 重新转换；cache envelope 已通过版本号
自动失效并重新构建。

## 序列化和缓存

导出不带项目元数据的原始 Denglin engine：

```cpp
session.serialize_engine("model.denglin.engine");

cvinfer::core::DenglinSessionOptions serialized_options;
serialized_options.model_format = cvinfer::core::DenglinModelFormat::SerializedEngine;
auto restored = cvinfer::core::DenglinSession::load(
    "model.denglin.engine", serialized_options);
```

ONNX 构建缓存使用版本化 envelope，命中前会校验 SDK 版本、设备架构、
ONNX FNV-1a hash、输入 shape/name、输出选择、dtype、cluster、batch、
weight-share、动态 shape profiles、builder flags 和插件配置。缓存不与
TensorRT engine 兼容：

```cpp
cvinfer::core::DenglinSessionOptions cached_options;
cached_options.input_name = "images";
cached_options.input_shape = {1, 3, 640, 640};
cached_options.cache_mode = cvinfer::core::DenglinCacheMode::ReadWrite;
cached_options.engine_cache_path = "cache/yolo26n.denglin.cache";

auto session = cvinfer::core::DenglinSession::load(
    "test/yolo26n.onnx", cached_options);
if (session.cache_hit()) {
    // No ONNX parser/builder work was needed for this load.
}
```

`ReadOnly` 不写缓存但默认允许 miss 后临时构建；
`build_on_cache_miss=false` 可将缓存 miss 变成错误。
缓存写入使用临时文件替换，避免进程中断留下半个 engine。

## 推理 Smoke Test

配置时打开可选测试目标：

```powershell
cmake -S . -B build-denglin `
  -DBUILD_DENGLIN_BACKEND=ON `
  -DBUILD_DENGLIN_SMOKE_TEST=ON `
  -DDENGLIN_SDK_ROOT=E:/denglin-hamming/sdk `
  -DCUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8" `
  -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-8.6.1.6_cuda-11.8" `
  -DOPENCV_ROOT=E:/opencv/build
cmake --build build-denglin --target alanxinn_cvinfer_denglin_smoke
```

运行 ONNX smoke test 前，需要同时准备 SDK DLL、Denglin Python 转换环境和
TVM 测试数据目录。ONNX Parser + Builder 会调用 `python -m dl convert`，
因此 Python 环境必须能导入 `numpy` 和 SDK 的 `dl` 模块：

```powershell
$env:PATH="C:/Program Files/Python310;E:/denglin-hamming/sdk/lib;E:/denglin-hamming/sdk/bin;$env:PATH"
$env:PYTHONPATH="E:/denglin-hamming/sdk/python;E:/denglin-hamming/sdk/python/tvm/python;E:/denglin-hamming/sdk/python/tvm/dl/python;$env:PYTHONPATH"
$env:PYTHONNOUSERSITE="1"
$env:TEST_DATA_ROOT_PATH=(Join-Path (Get-Location) ".tvm_test_data")
python -c "import numpy; import dl; print('Denglin Python environment OK')"
.\build-denglin\bin\Debug\alanxinn_cvinfer_denglin_smoke.exe `
  test/yolo26n.onnx 1 images 1x3x640x640 cache/yolo26n.denglin.cache
```

如果使用 Ninja 单配置构建，smoke 可执行文件位于
`build/<preset>/bin/alanxinn_cvinfer_denglin_smoke.exe`；上例中的
`bin\Debug` 适用于 Visual Studio 多配置构建。缺少 `numpy` 时会在 builder
阶段报告 `ModuleNotFoundError`，这不是 C++ Core 编译错误。

测试会依次验证 ONNX 构建、Host 输入、`Enqueue` 推理、Host 输出复制，
关闭 Host 复制后的 Device 输出地址和 shape、原始 engine 导出/反序列化，
以及缓存写入和下一次缓存命中。成功时打印 `Denglin Core smoke PASS`。
