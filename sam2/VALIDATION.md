# SAM2.1 tiny 真实权重与 NuGet 验证

日期：2026-11-05。GPU：RTX 4060 8GB。

Meta 官方 SAM2.1 Hiera tiny 预训练权重，通过严格加载后导出 ONNX。
**两个 engine 均使用实际安装的 `alanxinn.TensorRtOnnxBuilder.TensorRT10/0.2.0` 转换。**
消费者只有 PackageReference，使用独立 NuGet 缓存和本地 feed；未使用 ProjectReference、trtexec 或 Python C ABI builder 转换。
清除 SDK 环境变量/PATH，强制使用包依赖及自动合并的运行库；实际 DLL 来源检查通过。

## 权重与转换

官方源码固定提交：`2b90b9f5ceec907a1c18123530e92e794ad901a4`。
权重：`models/sam2.1/sam2.1_hiera_tiny.pt`。
SHA256：`7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`。
PyTorch 2.7.1，官方 SAM2 源码；PyTorch 参考计算使用 CPU，TensorRT 使用 GPU。

FP32，关闭 TF32，workspace 2048 MiB。图像输入 1×3×1024×1024。
掩码解码器点输入 profile：min=1、opt=4、max=34；最大包含框角 2 个和用户点 32 个。
静态特征及 mask 输入也通过 TensorRtOptimizationProfile 指定。

| Engine | 大小 bytes | NuGet 转换秒数 |
| --- | ---: | ---: |
| image_encoder | 111123836 | 27.84 |
| mask_decoder | 22036964 | 11.52 |

## 真实图片及接口对照

输入：`cache/trt10-models/bus.jpg`，810×1080。
框 `(50,390,245,910)`，正点 `(150,600)`，负点 `(240,850)`。
参考为官方 SAM2ImagePredictor；比较的是转换一致性，不是人工标注准确率。

| 操作 | Mask IoU 对照 PyTorch | 分数绝对误差 |
| --- | ---: | ---: |
| point-only | 1.000000 | 1.19e-07 |
| box | 1.000000 | 0 |
| box-positive | 1.000000 | 0 |
| box-positive-negative-refined | 1.000000 | 0 |
| remove-negative | 1.000000 | 0 |
| clear-box-keeps-points | 1.000000 | 1.19e-07 |
| set-box-resets-points | 1.000000 | 0 |
| clear-points-keeps-box | 1.000000 | 0 |
| 框 + 32 个正点 | 1.000000 | 2.14e-07 |

8 种交互操作中，C# NuGet 输出 mask 与原生输出逐像素完全相同；原生结果与官方 PyTorch mask 一致。
另外，C# NuGet 在 32 点上限时与 PyTorch 的 IoU 也是 1.0，第 33 个用户点被正确拒绝。
预测前未设置图片/提示、缺失反馈 mask、非法坐标、删除点索引以及重置图片等状态检查均通过。

本地结果图：`cache/sam2.1/validation/comparison.jpg`。

## 本地记录

- 汇总、权重与 engine SHA256：`cache/sam2.1/validation/summary.json`。
- ONNX：`cache/sam2.1/onnx`；NuGet 生成的 engines：`cache/sam2.1/engines`。
- 实际 PackageReference 工程与源码：`cache/sam2.1/nuget-consumer`。
- 转换日志：`cache/sam2.1/nuget-build-engines.log`。
- 官方对照：`cache/sam2.1/validation/validation.json`。
- C# 对照：`cache/sam2.1/nuget-predictions/parity.json`。
- 动态上限：`cache/sam2.1/nuget-max-points/parity.json`。

NuGet consumer 的两个 PackageReference 均锁定 `[0.2.0]`，10 个包的 NuGet SHA512 与磁盘包逐一核对。
汇总中另保存 NuGet SHA256。模型与测试产物保留本地，不随 Git 提交。

## 验证范围

已验证 SAM2.1 tiny、单张真实图片和上述提示组合，尚未覆盖其余三个模型规格或 RTX30/50 实机。
DINOv3 预训练权重仍缺失；经用户确认，待补充授权权重后验证，当前没有新增预训练效果结论。
