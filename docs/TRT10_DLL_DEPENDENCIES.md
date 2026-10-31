# TensorRT10 DLL 与精简打包

2026-10-31：基于本机 10.16.1.11 DLL 的普通／延迟 PE 导入、此前模型调用的模块记录及 NVIDIA 文档调整。
本次只修改配置，未编译、重打包或重新运行模型；10-30 的完整包验证结果不等于精简包验证结果。

## 当前项目需要什么

| 场景 | TensorRT DLL | CUDA DLL | cuDNN DLL |
| --- | --- | --- | --- |
| 已有 engine 推理 | `nvinfer_10.dll`、`nvinfer_plugin_10.dll` | `cudart64_12.dll` | 当前模型不要求 |
| ONNX 构建 | 上述文件＋`nvonnxparser_10.dll`、`nvinfer_builder_resource_ptx_10.dll`、目标 SM 的 resource | `cudart64_12.dll` | 默认不要求 |

本机 RTX 4060 对应 `nvinfer_builder_resource_sm89_10.dll`；PTX 资源保留作为公共／JIT 回退资源，
没有仅根据一次模块快照将它删除。插件 DLL 也必须保留，因为项目显式调用 `initLibNvInferPlugins`。
推理 NuGet 还包含项目 API DLL 和 OpenCV；`nvcuda.dll` 由 NVIDIA 驱动提供，不从 SDK 打包。

默认 builder ZIP 白名单共 **6 个 DLL**：上表的两个 TensorRT inference DLL、parser、PTX、SM89 resource 和 CUDA runtime。
本机源文件未压缩合计 1,227,442,224 bytes；之前完整环境 26 个 DLL 合计 5,409,523,504 bytes。
这是静态文件大小计算，尚未生成新的 ZIP，不能作为实际压缩后大小。

## 默认去掉什么

- `cudnn*64_9.dll` 全部组件。
- `cublas64_12.dll`、`cublasLt64_12.dll`。
- `nvrtc64_120_0.dll`、`nvrtc-builtins64_129.dll`。
- 目标架构之外的 `nvinfer_builder_resource_sm*_10.dll`。
- inference NuGet 中的非 CUDA12 runtime 和旧 `zlibwapi.dll` 通配收集。

cuDNN/cuBLAS 仍可能被特定弃用算子或自定义插件动态加载，不能推广为所有 TensorRT 模型都不需要。
需要时显式启用可选库；cuDNN 应携带完整组件和 cuBLAS/NVRTC 依赖，不能只复制主 DLL。

## 配置与产物

```powershell
# 下次编译/打包时生效，默认 SM89、无可选库
.\scripts\Pack-Trt10.ps1
# 指定目标架构
.\scripts\Pack-Trt10.ps1 -CudaArchitectures 86,89
# 确有插件/弃用算子需要时才包含可选库；全架构分发另加 -AllArchitectures
.\scripts\Pack-Trt10.ps1 -IncludeOptionalLibraries
```

默认 ZIP 名：`tensorrt10.16.1.11-cuda12.9-sm89-runtime-win-x64.zip`。
多架构、全架构和可选库使用独立文件名，ZIP 内 `runtime-manifest.json` 记录实际 DLL 白名单。
旧完整 ZIP 保留在磁盘但不再列入新产物清单；`-SkipRuntimeArchive` 需要新的对应 profile ZIP，不能复用旧完整包。

CMake 使用 `CVINFER_TRT_BUILDER_ARCHITECTURES`（默认跟随 `CMAKE_CUDA_ARCHITECTURES`，也可 `ALL`）
和 `CVINFER_TRT_INCLUDE_OPTIONAL_LIBRARIES=OFF`，不再强制安装 cuDNN 头文件。
只清理当前 build 的 builder runtime 中不再需要的 SDK DLL，不删除 SDK 或自定义插件。

C# 本地 SDK 缓存默认不复制 cuDNN/cuBLAS/NVRTC，可通过
`TensorRtRuntime.IncludeOptionalLibraries = true` 或 `ALANXINN_TENSORRT_OPTIONAL_LIBRARIES=1` 启用。
默认缓存路径增加 `minimal` / `optional` 子目录，区分旧完整缓存。
NuGet 测试脚本按 ZIP hash 隔离解压目录，并检查精简模式没有加载可选库；这些修改尚未执行新的消费测试。

依据：[NVIDIA TensorRT10 可选依赖](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/prerequisites.html#optional-dependencies)、
[cuDNN 运行依赖](https://docs.nvidia.com/cudnn/latest/installation/build-run-cudnn/)。
