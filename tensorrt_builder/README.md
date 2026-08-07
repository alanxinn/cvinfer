# tensorrt_builder

`tensorrt_builder` 是 TensorRT ONNX→engine 的 Windows x64 C++ native builder。
它直接调用 TensorRT 的 `IBuilder`、`IBuilderConfig`、`INetworkDefinition` 和
`nvonnxparser::IParser`，通过稳定的 C ABI 供 C# NuGet 包调用，不启动
`trtexec.exe`。

## 构建要求

- Windows x64、Visual Studio C++17、CMake 3.23 或更高版本；
- CUDA 11.8；
- TensorRT 8.6.1.6 CUDA 11.8；
- OpenCV 配置（根项目 CMake 配置需要）。

预设中的 `CUDA_ROOT`、`TENSORRT_ROOT` 和 `OPENCV_ROOT` 需要按本机安装路径修改。

## 构建

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
```

启用私有 CUDA backend 时使用：

```powershell
cmake --preset ninja-msvc-private-cuda-release
cmake --build --preset build-ninja-msvc-private-cuda-release
```

## 输出目录

ONNX 转换 DLL 和指定运行时依赖单独输出到：

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

常规 `alanxinn_cvinfer.dll` 和其他推理产物仍在 `build/<preset>/bin`，不会与
转换器 DLL 混用。CMake 使用固定白名单，不会把 TensorRT/CUDA 目录中的其他 DLL
复制到转换器输出目录。

NuGet 包只收集 `alanxinn_tensorrt_builder.dll`，其余八个大型 DLL 用于制作 GitHub
Release runtime ZIP。可在仓库根目录执行：

```powershell
.\csharp\alanxinn.TensorRtOnnxBuilder\tools\Pack-TensorRtRuntime.ps1 `
  -SourceDirectory .\build\ninja-msvc-release\tensorrt_builder `
  -OutputZip .\tensorrt8.6.1.6-cuda11.8-runtime-win-x64-0.1.2.zip
```

## C ABI

头文件位于 `include/tensorrt_builder/tensorrt_builder_api.h`，主要接口为：

- `trt_builder_config_init`：初始化精度、workspace、device 和其他构建选项；
- `trt_builder_build_onnx`：解析 ONNX、构建 engine 并写入目标文件；
- `trt_builder_last_error`：读取线程本地 UTF-8 错误信息。

动态输入通过 min/opt/max optimization profile 传入。INT8 适用于已包含 Q/DQ
量化节点的 ONNX；自定义 calibrator 需要扩展 native builder API。

engine 与 TensorRT 主版本、CUDA 版本和 GPU 架构相关，不能保证跨环境通用。
