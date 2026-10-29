# SAM3.1 真实权重图像推理验证

日期：2026-10-29。分支：`trt10`。RTX 4060 8GB；TensorRT 10.16.1.11 / CUDA 12.9。
本记录使用本地完整预训练权重，和此前随机微型 SAM3 测试分开。

## 权重与严格加载

- 来源：`models/sam3.1/sam3.1_multiplex.pt`（3,502,755,717 bytes）。
- SHA256：`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。
- 检测分支：1,464 个张量；独立交互分支：681 个张量。缺失、额外参数和形状不匹配均为零。
- 权重转为 FP32；ONNX opset 17；六个 TensorRT 引擎均为 FP32，关闭 TF32。
- 输入完整 1008×1008；原图 `cache/trt10-models/bus.jpg` 为 810×1080。
- Python 3.10、Torch 2.7.1、Transformers 5.8.0；参考计算在 CPU，TensorRT 在 GPU。

## 端到端数值对照

与转换后 HF 模型的完整 PyTorch 推理比较；交互候选取最高 IoU 评分，文本实例按框 IoU 匹配。
最低要求：mask IoU ≥ 0.95、评分绝对误差 < 0.05，且有非空结果。

| 提示阶段 | mask IoU | 评分绝对误差 | 结果 |
| --- | --- | --- | --- |
| box | 0.99946061 | 0.00085521 | PASS |
| positive | 0.99944026 | 0.00338411 | PASS |
| negative | 0.99909256 | 0.00312471 | PASS |
| text: person（4 个实例） | 最小 0.99885736 | 最大 0.00047594 | PASS |

框为 `[50,390,245,910]`，正点 `[150,600]`，负点 `[240,850]`，均为原图坐标。
三个阶段均输出左侧白色外套人物的非空掩码，逐次加点会更新掩码。
已人工查看文本与交互叠加图，分割位置符合提示。
原生 BPE 与 HF tokenizer 编码路径的 `person` 结果逐像素一致。

额外原生真实图片提示：

| 文本 | 实例数 | 原生 BPE / HF tokenizer 结果一致 |
| --- | --- | --- |
| bus | 1 | PASS |
| person wearing a white jacket | 1 | PASS |

这些额外提示验证了真实图像效果及两种编码入口一致；未单独做 PyTorch 数值对照。

## Meta 原始交互头对照

使用未修改的 Meta `PromptEncoder`、`MaskDecoder`、`TwoWayTransformer` 和真实权重，
以随机空间特征分别测试框、框＋正点、框＋正负点。Meta 仓库 revision：
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`。

- mask logits 最大绝对误差：3.4332275e-05。
- IoU 评分最大绝对误差：3.5762787e-07。
- 已修正 HF 5.8.0 对交互 IoU 头强制 sigmoid 的差异。
- 三层 FPN、独立 `interactive_sam_*` 头和 no-memory embedding 均按 SAM3.1 源码选取。

此项直接对照的是交互提示／掩码头；完整图像端到端的数值参考是上面的转换后 HF 模型。

## C# 与显存

.NET 8 实测：框、增点、减点、清空点恢复框、文本、编码 tokens、结果释放后的存活。
所有掩码与 Python 调用 C ABI 保存的结果逐像素一致；文本实例数为 4。
集成测试项目编译零警告、零错误。C# 公共封装保持原有 net472/net6/net8 兼容目标。

FP32 单图流程在 8GB GPU 完成；nvidia-smi 约每 0.55 秒采样的**全卡总显存峰值**为
7404MiB，测试前为 1759MiB，包含桌面及其他程序，不是精确分配峰值。
视觉特征缓存到 CPU；大型编码器在计算完成后释放。并发、多图、FP16 和视频 multiplex 未验证。
首次框调用约 3.14 秒、首次文本调用约 5.14 秒，含模型加载及图像编码；
后续加点约 0.014–0.199 秒（含 decoder 首次加载），仅为本次功能验证观测，不是稳定性能基准。

## 复现与产物

转换、导出、构建、真实验证命令见 [README](README.md)。默认真实引擎目录为
`cache/sam31/engines`（原始默认 CLI 目录仍是 `cache/sam3/engines`，运行时请传 `--engines`）。

- `cache/sam31/real-validation.json`：完整数值报告。
- `cache/sam31/real-validation-images`：框、加点、减点及文本的 PNG mask / JPG 叠加图 / 原始二值 buffer。
- `cache/sam31/phrase-validation`：额外文本提示的图像与编码入口核对。
- `cache/sam31/meta-head-validation.json`：Meta 原始头对照。
- `cache/sam31/gpu-memory.json`：显存采样。
- `cache/sam31-*.log`：转换、导出、构建、Python 与 C# 实际调用日志。

```powershell
# 可选：原始 Meta 交互头数值检查（需本地官方源码 clone 和 einops）
python scripts/test_sam31_meta_heads.py --meta-source cache/sam31-meta
# C# 真实调用；先设置 ALANXINN_SAM3_NATIVE_DIR 和 bintrt 的 PATH，见部署说明
# 最后一个参数是 validate_sam3_real.py 生成的原始 mask 目录
dotnet csharp/alanxinn.TensorRtIntegrationSmoke/bin/Release/net8.0-windows/alanxinn.TensorRtIntegrationSmoke.dll `
  --real-sam3 cache/sam31/engines cache/trt10-contract/bus.bgr cache/sam31/real-validation-images
```

模型、转换权重、引擎及图片只保留在本机，不提交 Git。
参考：[Meta SAM3 源码](https://github.com/facebookresearch/sam3)、
[HF SAM3 转换映射](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam3/convert_sam3_to_hf.py)。
