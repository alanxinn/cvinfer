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
└─ alanxinn_tensorrt_builder.dll
```

该目录只放 builder DLL。`nvinfer_plugin.dll`、`nvonnxparser.dll`、CUDA、
cuBLAS、cuDNN 和 zlib 等运行库由 C# 包的 `TensorRtRuntime` 单独配置或按需
下载，避免与 `bintrt`/推理包的运行时混放。

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
