# tensorrt_builder

TensorRT ONNX→engine 的 Windows x64 C++ DLL，通过 C ABI 供 C# 调用。

## 要求

- Visual Studio C++17
- CMake 3.23+
- CUDA 11.8
- TensorRT 8.6.1.6
- OpenCV

## 构建

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
```

输出：

```text
build/<preset>/tensorrt_builder/
├─ alanxinn_tensorrt_builder.dll
├─ nvinfer.dll
├─ nvinfer_plugin.dll
├─ nvonnxparser.dll
├─ cudart64_110.dll
├─ cublas64_11.dll
├─ cublasLt64_11.dll
├─ cudnn64_8.dll
└─ zlibwapi.dll
```

## C ABI

头文件：`include/tensorrt_builder/tensorrt_builder_api.h`

主要接口：

```text
trt_builder_config_init
trt_builder_build_onnx
trt_builder_build_onnx_with_progress
trt_builder_last_error
```

动态输入通过 min/opt/max profile 传入。
`trt_builder_build_onnx_with_progress` 只提供阶段进度和构建心跳。

INT8 适用于已包含 Q/DQ 节点的 ONNX。
engine 与 TensorRT、CUDA 和 GPU 架构相关。
