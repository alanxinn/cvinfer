# TensorRT10 NuGet 与运行环境验证

日期：2026-10-30。平台：Windows x64 / RTX 4060 8GB。

已编译 public native、两个 NuGet 的 net472/net6.0/net8.0 程序集，生成完整环境 ZIP。
通过本地 feed 的 **PackageReference** 构建 .NET 8 消费项目；没有项目引用、手动复制 native DLL，
实际加载路径均在消费项目输出或环境 ZIP 解压目录。编译零警告、零错误。

## 产物

目录：`package/TensorRT10`。未发布到远程 NuGet。

| 文件 | bytes | SHA256 |
| --- | ---: | --- |
| alanxinn.CvinferSharp.TensorRT10.0.2.0-trt10.nupkg | 252015335 | `fd0752d0ee3e2f681375b9fab4abac64c4fec621701b1538c4a3aa7d1c4c181d` |
| alanxinn.TensorRtOnnxBuilder.TensorRT10.0.2.0-trt10.nupkg | 126267 | `d50bbd2fd93dc7177105496bff71417b7f83dfffec73df9893483512afe9b4c3` |
| tensorrt10.16.1.11-cuda12.9-cudnn9.27-runtime-win-x64.zip | 3566570554 | `e2491e1503c3f59ca10a2c80baac66bc795e5e3e0d9a26e9df900c8a414d86e1` |

包 ID 使用 `.TensorRT10` 后缀，程序集名和命名空间保持兼容。
推理包包含 public native、SAM3、DINOv3、TensorRT inference/CUDA/OpenCV runtime；
builder 包包含 native builder，另外的 ZIP 含 26 个 NVIDIA DLL（原始合计 5,409,523,504 bytes）。
环境含 TensorRT 10.16.1.11、CUDA 12.9、cuDNN 9.27 及各 GPU 架构 builder resource DLL。

直接加载 ZIP 中的 DLL 调用版本 API：

- `cudaRuntimeGetVersion`：12090。
- `cudnnGetVersion`：92700。
- `getInferLibVersion`：101601（TensorRT 10.16.1）。

## 模型测试

自动读取 `cache/trt10-models/models.json` 并查找 engine/ONNX。
通过新 NuGet 的 builder 从 ONNX 重建 YOLO11 检测 engine，推理后再次构建 INT64 用例，
验证同一进程 builder → inference → builder 的 DLL 资源发现。
其余检测模型使用本地匹配的 TRT10 引擎，所有检测模型均做三次重复推理。

| 模型 | 实际结果 | 状态 |
| --- | --- | --- |
| yolo11n | 对象 5，非空 mask 0，有效关键点 0 | PASS |
| yolo11n-seg | 对象 5，非空 mask 5，有效关键点 0 | PASS |
| yolo11n-pose | 对象 4，非空 mask 0，有效关键点 50 | PASS |
| yolo26n | 对象 5，非空 mask 0，有效关键点 0 | PASS |
| yolo26n-seg | 对象 4，非空 mask 4，有效关键点 0 | PASS |
| yolo26n-pose | 对象 4，非空 mask 0，有效关键点 47 | PASS |
| rfdetr-nano | 对象 7，非空 mask 0，有效关键点 0 | PASS |
| rfdetr-seg-nano | 对象 8，非空 mask 8，有效关键点 0 | PASS |
| sam3.1 | person 4 个实例，白色外套描述 1 个实例 | PASS |
| dinov3-dinov3_convnext | 32 维特征；随机权重算子验证 | PASS |
| dinov3-dinov3_vit | 32 维特征；随机权重算子验证 | PASS |

SAM3.1 覆盖框、增点、减点、清空点恢复框、person 和英文白色外套描述、结果生命周期。
DINOv3 两项使用真实算子但随机权重，**不表示预训练语义效果验证**；仍待补充授权权重。
检测/分割/pose 检查有置信度的框、非空 mask、17 点结构及有限坐标；本次是 NuGet 安装与
实际调用验证，之前的原生数值对照记录见 [TRT10 验证](TRT10_VALIDATION.md) 和 [SAM3 验证](../sam3/VALIDATION.md)。

改名后的 `buildTransitive/alanxinn.CvinferSharp.TensorRT10.targets` 已被消费项目自动导入，
七个 inference native DLL 保存在 `runtimes/win-x64/native`，没有平铺到输出根目录。
另用按包 SHA256 隔离的新 NuGet 缓存还原/编译消费项目，YOLO26 实际推理通过；
后续同版本重打包按内容 hash 切换缓存，避免测到旧包。

## 复现与记录

```powershell
.\scripts\Pack-Trt10.ps1
# 已编译并已有正确环境 ZIP：
.\scripts\Pack-Trt10.ps1 -SkipNativeBuild -SkipRuntimeArchive
```

- 产物及完整 SHA256 清单：`package/TensorRT10/artifacts.json`。
- 完整测试报告：`cache/nuget-trt10-validation/validation.json`。
- 独立新缓存检查：`cache/nuget-trt10-validation/fresh-cache-check.json`。
- 编译/打包/测试日志：`package/TensorRT10/logs` 和 `cache/nuget-trt10-validation`。
- 本次首次 native build / ZIP 打包日志：`cache/nuget-trt10-native-build.log`、`cache/nuget-trt10-runtime-pack.log`。
