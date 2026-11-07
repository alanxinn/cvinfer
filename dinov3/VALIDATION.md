# DINOv3 接入验证记录

## 2026-11-08：真实预训练权重与 NuGet 转换

环境：Windows x64、RTX 4060 8GB（SM89）、TensorRT 10.16.1.11、CUDA 12.9，
PyTorch 2.7.1、Transformers 5.8.0；C# 消费程序为 .NET 8。
本次使用用户提供的本地 safetensors snapshot，不需要联网下载权重或 Hugging Face 登录。

模型根目录：`D:/BaiduNetdiskDownload/DINOV3/DINOV3-safetensor`。
以下三个目录均使用严格加载，未缺失或额外补入模型参数：

| 目录名 | 全局维度 | patch / 特征图 | FP32 最大相对 L2 误差 | 结果 |
| --- | ---: | --- | ---: | --- |
| `dinov3-vits16-pretrain-lvd1689m` | 384 | 196×384 / 384×14×14 | 2.53576e-5 | PASS |
| `dinov3-vits16plus-pretrain-lvd1689m` | 384 | 196×384 / 384×14×14 | 9.19062e-6 | PASS |
| `dinov3-convnext-tiny-pretrain-lvd1689m` | 768 | 49×768 / 768×7×7 | 6.43369e-6 | PASS |

输入统一为 `1×3×224×224`；ViT 排除 CLS 和 4 个 register tokens，
ConvNeXt 排除全局池化 token。误差为原图 BGR 端到端四种输出相对 HF PyTorch 的最大值。

### 实际 NuGet 转换与运行库来源

ONNX 先通过 PyTorch／ONNX Runtime 对照，再由隔离的纯 PackageReference 消费程序调用
`alanxinn.TensorRtOnnxBuilder.TensorRT10 [0.2.0]` 转换，FP32、关闭 TF32、workspace 2048 MiB。
没有调用 Python C ABI builder，也没有使用 trtexec。

| 模型 | engine 字节数 | 转换耗时 |
| --- | ---: | ---: |
| ViT-S/16 | 87,364,172 | 8.88 s |
| ViT-S+/16 | 115,837,196 | 10.33 s |
| ConvNeXt-Tiny | 113,422,188 | 11.93 s |

耗时为本机此次采样，可能复用 TensorRT／驱动缓存，不代表性能基准。
转换包与 8 个运行库包均从本地 feed 还原；缓存按包哈希隔离。
强制 `ALANXINN_TENSORRT_REQUIRE_BUNDLED=1`、`ALANXINN_TENSORRT_NO_DOWNLOAD=1`，
报告确认 parser、builder、CUDA 和 SM89 resource DLL 来自实际 NuGet 输出／自动合并缓存，
没有回退到本机 SDK 或 GitHub runtime ZIP。

推理另使用仅引用 `alanxinn.CvinferSharp.TensorRT10 [0.2.0]` 的独立 C# 项目，
检查主包与 Core 的已还原 SHA512 与实际 nupkg 相符，并确认原生 DLL 均来自包输出目录。
原生推理 API 无需修改或重新编译，原有 0.2.0 包可执行本次真实模型。

### 图片与接口对照

使用已有真实图片 `bus.jpg` 和 `woman_hands.jpg`，每个模型覆盖：

- 原图、水平翻转、90° 旋转、非等比缩放至 317×421。
- 人物裁剪、公交裁剪和另一张手部图片。
- 每张图的 BGR 完整预处理与官方 processor 生成的 CHW 直接输入。
- 全局向量、L2 归一化向量、patch tokens、NCHW 特征图；形状、有限值、patch/map 排列和单位范数。
- C# 重复推理、独立模型重载、释放模型后旧托管输出仍有效。

共 21 组真实图像用例全部通过。C# BGR 四种输出与 C ABI 的 FP32 字节逐一相同；
C# CHW 输出也通过 PyTorch 数值对照。声明的相对 L2 阈值为 0.003。
此外，移除一个预训练 tensor 和构造缺失分片索引的负例均被正确拒绝。

### 预训练特征示例

| 模型 | 原图与水平翻转余弦相似度 | 原图与手部图片余弦相似度 |
| --- | ---: | ---: |
| ViT-S/16 | 0.97019 | 0.02305 |
| ViT-S+/16 | 0.97309 | 0.04529 |
| ConvNeXt-Tiny | 0.97791 | 0.08884 |

已生成 ViT-S/16 的 PCA 特征图、NPZ 特征和 JSON 相似度结果。
PCA 颜色不代表类别；这些示例展示当前图片的特征行为，不是分类／分割精度评估。

### 产物与复现

真实模型产物根目录为 `cache/dinov3-real`，每个 `vits16`、`vits16plus`、`convnext-tiny` 子目录包含：

- `onnx/bundle.json`：配置、processor、权重 SHA256 和 ONNX 对照结果。
- `engine/model.engine`、`engine/bundle.json`、`engine/model.build.json`：转换来源、包哈希、engine 哈希和 DLL 路径。
- `validation/real-validation.json`：7 组 PyTorch 数值对照及相似度。
- `validation/csharp-parity.json`：C# 逐项对照结果。
- `validation/fixtures`、`validation/csharp`：原始 BGR／CHW、参考特征与逐次输出。

C# 汇总为 `cache/dinov3-real/csharp-summary.json`；
ViT-S/16 预览为 `cache/dinov3-real/vits16/preview/feature_map_pca.png`。
权重、engine、缓存和 Python 验证脚本均不提交 Git。
导出、NuGet 构建和可视化命令见 [模块 README](README.md#构建与预训练权重)。

本次未验证目录中的其余大模型和 adapter head；7B 权重约 26.9 GB，未尝试在 8GB 显卡上完整推理。
本次真实权重仅验证 FP32 和 RTX 4060，FP16／其他 GPU 不属于本记录的验证范围。

## 2026-10-29：随机权重算子与接口

日期：2026-10-29。分支：`trt10`。RTX 4060，TensorRT 10.16.1.11，CUDA 12.9。

本记录使用缩小的随机初始化 DINOv3 模型，验证算子／接口／预处理正确性，**不代表预训练语义效果验证**。

| 架构 | 输入 | 全局维度 | 网格 | register 剔除 | FP32 | 允许 FP16 |
| --- | --- | --- | --- | --- | --- | --- |
| ViT | 1×3×64×64 | 32 | 4×4 | 4 | PASS | PASS |
| ConvNeXt | 1×3×64×64 | 32 | 2×2 | 0 | PASS | PASS |

与官方 DINOv3 processor + ONNX Runtime 的端到端误差：

| 模式 | 测试图像 | 最大绝对误差（所有输出） |
| --- | --- | --- |
| FP32 | 43×97、113×191、17×21、64×64、1×1 | 3.2186508e-06 |
| 允许 FP16 | 43×97、113×191、17×21、64×64、1×1 | 0.0089366436 |

还验证了：

- CHW 直接输入与 BGR 完整预处理两条路径。
- patch tokens 与 NCHW feature map 一一对应。
- L2 单位特征、余弦相似度、重复推理后旧结果有效。
- 无效图像、非有限值、错误输入长度、输出 buffer 不足、未知 tensor 名称和错误后的恢复。
- C ABI tensor metadata 的布局和复制元素计数。
- C++ 图像 demo、Python NPZ/JSON/PCA CLI，以及 .NET 4.7.2/6/8 编译与 .NET 8 实际调用。
- 零全局向量的归一化保持为零；余弦相似度拒绝零向量；释放模型后旧托管结果仍有效。

输出与日志保存在 `cache/dinov3-tiny` 和 `cache/dinov3-*.log`。
FP16 为允许 TensorRT 选择 FP16 tactics，部分层仍可能使用 FP32；外部 I/O 始终为 FP32。

以上保留当时的合成测试范围；真实预训练权重的后续验证见本页 2026-11-08 记录。
