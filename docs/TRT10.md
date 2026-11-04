# TensorRT 10 / CUDA 12.9 / cuDNN 9.27

本分支 `trt10` 使用以下 Windows x64 SDK：

| SDK | 路径 |
| --- | --- |
| TensorRT 10.16.1.11 | `C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.16.1.11-cuda-12` |
| CUDA 12.9 | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9` |
| cuDNN 9.27 | `C:/Program Files/NVIDIA/CUDNN/v9.27` |
| OpenCV 4.8 | `E:/opencv/build/x64/vc16/lib` |

## 编译

```powershell
.\scripts\Build-Trt10.ps1
.\scripts\Build-Trt10.ps1 -Preset trt10-cuda-release
```

第二条使用本机已有的 `private/cuda_backend`，无需修改私有仓库源码。
无私有源码时使用第一条，TensorRT 仍然在 NVIDIA GPU 上执行，前后处理在 CPU 上执行。
脚本自动使用 VS 2022 Build Tools 和自带 Ninja。默认 SM86/SM89/SM120，覆盖 RTX 30/40/50 系列。
这是运行库与原生代码的架构覆盖；engine 默认仍须在实际目标 GPU 上从 ONNX 构建。

## 下载和导出

使用独立 CPython 3.10 环境，安装 `scripts/requirements-trt10-models.txt`。
导出使用 CPU，Python 中 PyTorch 的 CUDA wheel 版本不影响 C++ 的 CUDA 12.9 链接。

```powershell
python scripts/prepare_trt10_models.py
```

默认从 Ultralytics 官方 GitHub assets v8.4.0 下载 YOLO11n、YOLO26n 的检测、
分割和姿态模型；从 Roboflow 官方存储下载 RF-DETR Nano 检测和分割权重，验证官方 MD5。
所有权重记录 SHA256，ONNX 使用 opset 17，原始输出和静态 batch=1。
缓存位于 `cache/trt10-models`；`models.json` 包含来源、版本、路径和哈希。
可用 `--models yolo11n rfdetr-nano` 只准备指定模型。

## 引擎及推理验证

```powershell
python scripts/validate_trt10.py
python scripts/validate_trt10.py --build build/trt10-cuda-release --compute cuda --reuse-engines
python scripts/compare_trt10_outputs.py
python scripts/test_trt10_contract.py
```

验证脚本使用本项目的 ONNX builder C ABI 构建引擎，再用 public inference C ABI
验证真实 bus 图片、重复推理、置信度、分割掩码和 17 个姿态关键点。
每个模型单独进程运行，日志、引擎、结果图片及 CPU/CUDA JSON 报告保存在
`cache/trt10-validation`。构建可能耗时数分钟；TensorRT 会针对本机 GPU 选择和编译 tactics。
YOLO 使用 FP16，RF-DETR 保持 FP32 并关闭 TF32。
数值验证使用独立 ONNX Runtime CPU 推理，与 C++ `TrtSession` 导出的各个原始 tensor 比较。
YOLO26 的 TopK 检测集合按类别和 IoU 匹配，避免把行排序差异当作数值错误。
完整检查也可运行 `scripts/Test-Trt10.ps1 -Python <CPython路径> -WithCuda -ReuseEngines`。

## 运行库

### NuGet 与环境打包

```powershell
.\scripts\Pack-Trt10.ps1
# 已编译原生库、已生成 runtime ZIP 时：
.\scripts\Pack-Trt10.ps1 -SkipNativeBuild -SkipRuntimeArchive
```

产物保存在 `package/TensorRT10`：

命名沿用 main 分支格式：只在原 PackageId 后加 `.TensorRT10`，版本号使用普通数字，不再附加 `-trt10`。
例如 main 的 `alanxinn.CvinferSharp.<版本>.nupkg` 对应 `alanxinn.CvinferSharp.TensorRT10.<版本>.nupkg`。
当前源码及已生成包版本为 `0.2.0`；旧 `0.2.0-trt10` 包已删除，历史验证记录仍保留原名。

- `alanxinn.CvinferSharp.TensorRT10.0.2.0.nupkg`：推理封装，自动依赖 Core runtime 包。
- `alanxinn.TensorRtOnnxBuilder.TensorRT10.0.2.0.nupkg`：ONNX builder，自动依赖三代 GPU 的全部资源包。
- `alanxinn.TensorRtRuntime.*.TensorRT10`：8 个自动还原的 runtime 依赖包，无需用户逐个安装。
- `tensorrt10.16.1.11-cuda12.9-sm86-sm89-sm120-runtime-win-x64.zip`：可选备份，使用 NuGet 不需要它。
- `artifacts.json`：产物大小和 SHA256。

两个 NuGet 支持 net472/net6.0/net8.0；程序集名和 C# 命名空间不变。
新包通过 `.TensorRT10` 后缀与其他 TensorRT 大版本区分，Denglin 包名不变。
默认不包含 cuDNN/cuBLAS/NVRTC；特定插件需要时加 `-IncludeOptionalLibraries`。
公开 NuGet 默认必须包含 SM86/SM89/SM120，完整服务器架构备份 ZIP 可加 `-AllArchitectures`。
具体 DLL 白名单及本次静态检查范围见 [DLL 依赖清单](TRT10_DLL_DEPENDENCIES.md)。
新包全部发布后，只安装主 NuGet 即可自动还原依赖；不需要安装开发 SDK、解压 ZIP、设置 PATH 或 runtime URL。
PTX/SM120 大文件首次初始化自动合并到用户缓存并校验 SHA256。
Windows x64 仍需适配显卡的 NVIDIA 驱动和对应 .NET 运行时；标准包不依赖私有 CUDA DLL。

Python 测试、验证及结果对比脚本仅保留在本地，不随 Git 提交；本地缺少测试脚本时，打包流程明确提示跳过测试。

本地存在 `test_trt10_nuget.py` 时，打包脚本默认执行它：从本地 feed 用 **PackageReference** 还原独立消费项目，
核对三种 TFM、自动依赖、native 目录隔离和实际 DLL 来源；禁止回退到 SDK/外部 ZIP；自动查找 engine/ONNX，
通过 NuGet builder 重建 YOLO11 检测引擎，再运行 YOLO11/26 det/seg/pose、RF-DETR det/seg、
本地 SAM3.1 和可用 DINOv3。缺失引擎但有 ONNX 时自动构建。
另会发现完整的 SAM2.1 engine bundle，验证框、正负点、mask 反馈和纯点提示；
没有 SAM2.1 engine 时明确记录跳过。SAM2.1 tiny 已通过 NuGet builder 构建 engine，并完成官方 PyTorch 与 C# NuGet 真实图片对照，见 [验证记录](../sam2/VALIDATION.md)。
无真实 DINOv3 权重时仅运行明确标记为 synthetic 的算子检查。
报告在 `cache/nuget-trt10-validation/validation.json`，日志位于该目录和产物 `logs` 目录。
2026-11-04 新生成包及 11 项消费测试结果见 [0.2.0 验证记录](TRT10_NUGET_0.2.0_VALIDATION.md)。
2026-10-30 的历史打包产物、哈希和 11 项消费测试结果见 [NuGet 验证记录](TRT10_NUGET_VALIDATION.md)。

- `bin/alanxinn_cvinfer_trt.dll`：原生推理 API。
- `bintrt`：`nvinfer_10.dll`、`nvinfer_plugin_10.dll`、`cudart64_12.dll`、OpenCV，及可选私有 CUDA DLL。
- `tensorrt_builder/alanxinn_tensorrt_builder.dll`：ONNX builder。
- `builder_runtime`：parser、PTX 和目标 SM 的 builder resource、CUDA12 runtime；可选库显式开启后才包含。

运行原生 demo 时将 `bintrt` 加到当前进程 PATH。C# demo 默认引用本分支源码，
不会误用 NuGet 中旧的 TRT 8 native DLL。公开 NuGet 打包仍禁止私有 CUDA DLL 依赖。
本分支没有已发布的 TRT 10 runtime ZIP 默认地址；需要网络分发时设置
`ALANXINN_TENSORRT_RUNTIME_URL`，并建议配置 `ALANXINN_TENSORRT_RUNTIME_SHA256`。
`Pack-TensorRtRuntime.ps1` 可从 `builder_runtime` 生成匹配 ZIP。

TensorRT 10 的当前项目路径不要求 cuDNN；特定插件或弃用算子所需 cuDNN 9.27 及依赖可按需部署。
旧 TRT 8 engine、CUDA 11 DLL、cuDNN 8 DLL 不可用于当前构建，需要重新生成 engine。

接口迁移依据：[NVIDIA TensorRT 8 → 10 C++ 迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-c-api.html)。
