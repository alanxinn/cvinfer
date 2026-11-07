# TensorRT10 Core 合并记录

日期：2026-11-07。本轮仅本地重新打包，没有上传。

`alanxinn.CvinferSharp.TensorRT10.0.2.0.nupkg` 已直接包含 Core 的 TensorRT/CUDA/MSVC 运行库，
没有其他 NuGet 依赖。仅推理时只需安装这个包。

- 文件：`package/TensorRT10/alanxinn.CvinferSharp.TensorRT10.0.2.0.nupkg`。
- 大小：252,962,503 bytes（241.24 MiB）。
- SHA256：`d660824af74033adc647aa106d2f01d4e5a91d8ff68a82b624d823a2181cde18`。
- 包大小上限：250 MiB（262,144,000 bytes），依据 [NuGetGallery 官方配置](https://github.com/NuGet/NuGetGallery/blob/main/src/NuGetGallery/Web.config)。

Core 包仍由独立转换包依赖；转换包可以不安装推理包而使用。两者同时安装的重复运行库资产
也经过统一 Demo 验证。Core 文件仍保留在产物目录，供转换包发布和安装使用。

## 修改

推理 csproj 移除 Core PackageReference，打包时从指定同版本 Core nupkg 提取白名单 DLL，
直接放入 runtimes/win-x64/native，包含 TensorRT inference、plugin、CUDA12 及 app-local MSVC runtime。
统一打包脚本将实际输出目录中的 Core 包传入，支持自定义产物目录。
沿用已修复的 Windows x64 RID 部署规则，保留 net472/net6.0/net8.0。
未增加 cuDNN/cuBLAS/NVRTC、ONNX parser 或 SM builder resource 到推理包。

打包大小检查从十进制 250,000,000 修正为官方配置使用的 250×1024×1024，避免误判。
版本继续使用 0.2.0。此前已安装旧同版本包时，应清理该版本缓存再重新还原。

## 检查

使用新的空 NuGet 缓存，从实际 nupkg 安装，没有 ProjectReference 或手动复制 native DLL。
仅推理的项目资产表中只有 1 个 NuGet 包；转换项目为主包和 8 个依赖，共 9 个。

| 独立安装项目 | net472 | net6.0 | net8.0 |
| --- | --- | --- | --- |
| 推理：真实 YOLO 图像 | PASS | PASS | PASS |
| 转换：实际 ONNX → engine | PASS | PASS | PASS |

统一 CvinferDemo 通过 PackageReference 安装推理包和转换包，14 项模型检查通过，
包含 YOLO11/26 det/seg/pose、RF-DETR det/seg、SAM3.1、SAM2.1 tiny、
DINOv3 两项随机权重算子和 MediaPipe hands FP32/FP16。
独立项目无编译警告；Demo 有一条现存 OpenCvSharp 分析器与 SDK 编译器版本警告，运行检查通过。

共 20 项实际调用检查通过。包 CRC、原生运行库、无依赖 metadata 与大小检查通过。
最后只更新包内 README，所有 DLL 和 buildTransitive targets 的 SHA256 与已测试包完全相同。
实机仅 RTX 4060；DINOv3 预训练效果仍待授权权重，RTX30/50 没有本次实机结论。

- 独立检查：`cache/core-merge-validation-20261107/summary.json`。
- Demo 检查：`cache/core-merge-demo-20261107/validation.json`。
- 产物 SHA256：`package/TensorRT10/artifacts.json`。

Python 检查脚本按用户要求保留本地并忽略，不随提交分发。
