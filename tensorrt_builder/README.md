# tensorrt_builder

TensorRT ONNX→engine 的 Windows x64 C++ DLL，通过 C ABI 供 C# 调用。

`trt_builder_config_t.strict_types` 支持 0（默认）、1（优先精度约束）、2（ONNX 强类型网络）。
模式 2 要求 `precision=TRT_BUILDER_PRECISION_FP32`，不添加自动 FP16／INT8 提示，实际精度按 ONNX 张量类型执行。
这可保留 FP16 模型中的 FP32 归一化，防止 WeVisOne-VL 语言图出现归一化溢出及全零 logits。

## 要求

- Visual Studio C++17
- CMake 3.23+
- CUDA 12.9
- TensorRT 10.16.1.11
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

该目录只放 builder DLL。`nvinfer_plugin_10.dll`、`nvonnxparser_10.dll`、CUDA、
cuBLAS、cuDNN 9 组件及 TRT 10 builder resource DLL 部署到独立的
`build/<preset>/builder_runtime`。`TensorRtRuntime` 优先发现该目录或本机安装；
如需网络分发，设置自有 TRT 10 runtime ZIP URL，不会下载旧 TRT 8 runtime。

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
