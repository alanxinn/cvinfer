# TensorRT10：RTX 30/40/50 与 NuGet 自动部署

2026-11-04 已编译、打包 0.2.0 并通过 11 项隔离消费测试，尚未远程发布。实机记录只覆盖 RTX 4060，
不表示 30/50 各型号均已实测。旧 0.2.0-trt10 包不包含下面的自动依赖方案。

| GeForce 系列 | 架构 | 默认 builder resource |
| --- | --- | --- |
| RTX 30 | SM86 | nvinfer_builder_resource_sm86_10.dll |
| RTX 40 | SM89 | nvinfer_builder_resource_sm89_10.dll |
| RTX 50 | SM120 | nvinfer_builder_resource_sm120_10.dll |

默认同时覆盖三种架构，包括对应桌面/移动系列；使用 Ada 的 30 系列变体也由 SM89 覆盖。
使用 CUDA 12.9 / TensorRT 10.16.1.11，Windows x64 需匹配显卡的 NVIDIA 驱动和 .NET。
**运行库跨系列支持不等于 engine 文件跨显卡通用。** 默认 engine 应在目标 GPU 上用 ONNX 构建。

## 安装方式

全部新包发布后，用户仅安装需要的主包：

```powershell
dotnet add package alanxinn.CvinferSharp.TensorRT10 --version 0.2.0
# 需要在应用内从 ONNX 构建时安装：
dotnet add package alanxinn.TensorRtOnnxBuilder.TensorRT10 --version 0.2.0
```

NuGet 自动还原依赖并部署到 runtimes/win-x64/native，无需手动安装各个子包。
无需另外下载 ZIP、安装开发 SDK、配置 PATH 或运行库 URL。
构建资源首次使用时自动合并到用户缓存并校验 SHA256；跨进程锁和临时文件保护并发初始化。
Core 包携带 app-local x64 MSVC runtime，避免要求用户另装 VC++ Redistributable。
模型仍受显存容量约束，支持某一显卡架构不代表所有大模型都能装入该型号显存。
源码项目引用保留本地 CMake SDK 流程；发布时必须用统一脚本，避免产生缺少依赖的包。

## 依赖及包大小

推理需要 nvinfer_10.dll、nvinfer_plugin_10.dll、cudart64_12.dll；API 和 OpenCV 随推理包分发。
构建另需 nvonnxparser_10.dll、PTX 和 SM86/89/120 resource。
nvcuda.dll 由 NVIDIA 驱动提供，不打包。默认不包含 cuDNN、cuBLAS、NVRTC 及无关服务器架构资源。
特定插件/弃用算子仍可能需要可选库，不能推广为任意模型均不需要这些依赖。

NuGet.org 单包上限 250MB；本机 PTX/SM120 文件压缩后仍超限，故拆成自动依赖包：

- alanxinn.TensorRtRuntime.Core.TensorRT10：三个 inference runtime DLL、x64 MSVC runtime 与资源索引。
- alanxinn.TensorRtRuntime.SM86.TensorRT10、alanxinn.TensorRtRuntime.SM89.TensorRT10：两个架构资源。
- alanxinn.TensorRtRuntime.PTX.Part1.TensorRT10、Part2、Part3：PTX 三个编号分片包。
- alanxinn.TensorRtRuntime.SM120.Part1.TensorRT10、Part2：SM120 两个编号分片包。

缩写 Part2/Part3 使用同样完整前缀和 .TensorRT10 后缀，共 8 个子包。
分片未压缩最大 240,000,000 bytes；打包与消费检查限制单包不超过 250,000,000 bytes。
两个主包锁定同版本依赖，不再重复内嵌大体积 Core DLL。

## 维护者操作

运行 scripts/Pack-Trt10.ps1：编译多架构 public native，生成 8 个 runtime 子包，再用本地 feed
还原依赖并打两个主包。发布时先上传全部同版本 runtime 子包，再上传主包；不能只上传主包。
独立 ZIP 仅为备份，默认文件名包含 sm86-sm89-sm120。
IncludeOptionalLibraries 参数只给备份 ZIP 加入可选库，不改变标准 NuGet 依赖。

消费检查脚本不再提供 SDK/ZIP DLL，仅设置独立空缓存并强制使用 NuGet 自带资源，
检查自动还原、资源合并、校验和、DLL 来源及模型调用。此流程已在 RTX 4060 上通过；
SAM2.1 仅完成编译，DINOv3 两项仍是随机权重算子验证，见 [0.2.0 验证记录](TRT10_NUGET_0.2.0_VALIDATION.md)。

参考：[NVIDIA GPU 架构表](https://developer.nvidia.com/cuda/gpus)、
[NuGet.org 大小限制](https://learn.microsoft.com/en-us/nuget/nuget-org/nuget-org-faq)、
[TensorRT10 可选依赖](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/installing-tensorrt/prerequisites.html#optional-dependencies)。
