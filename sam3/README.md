# SAM3 TensorRT 图像分割

当前已实现 C++、C ABI、C# 和 Python CLI 接口，并通过缩小的随机权重 SAM3
算子图验证。**完整官方预训练权重尚未验证**：本机没有已授权的 Hugging Face
登录或 SAM3 权重，官方 `facebook/sam3` 仓库需要人工授权。当前不把合成测试
当作真实模型推理成功，也未测定完整模型在 RTX 4060 8GB 上的显存峰值。

## 功能

- 原图坐标的框提示分割。
- 保留框并逐次添加正点（前景）和负点（背景），用全部累积提示重新预测。
- 清空点保留框；设置新框清空旧点；设置新图像清空框和点。
- 文本短语概念分割，返回多个实例的框、置信度和原图大小的二值掩码。
- 原生 CLIP BPE 支持英文 ASCII 短语；`PredictTokens` / `predict_tokens` 使用官方
  tokenizer 编码，可传入 Unicode。模型对不同语言的实际能力需另行验证。

交互框使用 SAM3 Tracker 的 PVS 分支，文本使用 SAM3 的 PCS 分支。交互预测选择
IoU 质量最高的候选掩码；文本分数为分类 sigmoid × presence sigmoid。
交互结果的框是提示框，文本结果的框是模型预测框。所有 mask 为原图完整尺寸、0/255。

## 权重、导出与构建

在 [官方模型页面](https://huggingface.co/facebook/sam3) 获得访问权限，然后在本机登录。
不要把 token 写进脚本、Git 或命令行参数。

```powershell
py -3.10 -m venv cache/sam3-python
cache/sam3-python/Scripts/python.exe -m pip install -r scripts/requirements-sam3.txt
cache/sam3-python/Scripts/hf.exe auth login
cache/sam3-python/Scripts/python.exe scripts/export_sam3.py
cache/sam3-python/Scripts/python.exe scripts/build_sam3_engines.py
cache/sam3-python/Scripts/python.exe scripts/validate_sam3_real.py
```

也可传 `--checkpoint D:/models/sam3-hf` 使用已有的 Hugging Face 格式 snapshot
目录（含 config、safetensors 和 processor/tokenizer 文件）。原始 Meta `sam3.pt`
不是 Hugging Face snapshot，本导出器不直接加载该格式，需要先转换为相应 HF 格式。

完整 ONNX 编码器可能使用 external data，必须保留整个导出文件夹。
默认 FP32、关闭 TF32。`build_sam3_engines.py --vision-fp16` 可选择视觉编码器 FP16，
但需先用真实权重对照验证精度。生成以下六个引擎：

```text
detector_vision.engine
tracker_vision.engine
text_encoder.engine
text_decoder.engine
box_decoder.engine
interactive_decoder.engine
```

同时部署 `vocab.tsv`、`merges.tsv`、`bundle.json`。交互点数动态范围默认为 1–32，
零点时走独立框解码器。图像特征缓存在 CPU 内存；视觉编码器完成后释放，再加载
解码器和文本编码器，避免同时保留两个大型视觉上下文。

导出包含对 Torch 2.7 tracing 不兼容的 Transformers mask factory 的等价展开；
非 tracing 的 PyTorch 参考路径保留官方实现。text encoder 的 causal/padding mask
显式展开，数值已用缩小的真实 SAM3 算子图对照。

## Python CLI

```powershell
python scripts/sam3_infer.py --image cache/trt10-models/bus.jpg `
  --box 50 390 245 910 --positive 150 600 --negative 240 850
python scripts/sam3_infer.py --image cache/trt10-models/bus.jpg --text "person"
# 使用官方 tokenizer 编码接口：
python scripts/sam3_infer.py --image cache/trt10-models/bus.jpg --text "person" --tokenizer facebook/sam3
```

交互 CLI 分别保存初始框、每次正点、每次负点后的结果。
默认引擎位置 `cache/sam3/engines`；输出 `cache/sam3/result`。

## C#

```csharp
using var sam = new Sam3Segmentor(@"cache\sam3\engines");
sam.SetImage(bgr, width, height);
sam.SetBox(50, 390, 245, 910);
var initial = sam.Predict();
sam.AddPoint(150, 600, positive: true);
var foreground = sam.Predict();
sam.AddPoint(240, 850, positive: false);
var refined = sam.Predict();
var people = sam.PredictText("person");
```

`Sam3Mask.Pixels` 是独立的托管数组，结果不依赖原生临时 buffer。
每个并发图像使用独立实例。使用自定义点数 profile 时，构造函数 `maxPoints`
应与导出参数一致。部署 `alanxinn_cvinfer_sam3.dll` 和同版本 `bintrt` 中的 runtime。

## 已完成的测试与边界

```powershell
python scripts/test_sam3_export.py
python scripts/build_sam3_engines.py --onnx cache/sam3-tiny/onnx --output cache/sam3-tiny/engines
python scripts/test_sam3_native.py
```

这些命令使用**随机初始化的微型 SAM3**，覆盖真实模型算子、六个 ONNX 子图、
TensorRT 构建、动态点数 1/4/8、框/正负点状态、坐标变换、文本 BPE、mask 还原、
非法参数、图像重置，以及 C# 调用。PyTorch→ONNX 的子图最大误差 < 8e-8；
使用官方图像 processor 做参考时，当前原生交互 mask 像素一致率为 99.61%–99.93%。
官方 uint8 resize 的量化与本地浮点抗锯齿 resize 存在细小差异。

实际预训练模型的精度、完整分辨率导出、8GB 显存可用性和真实图片交互效果，
均需在取得授权权重后继续验证。导出和构建状态在 `bundle.json` 中明确记录为待验证。

参考：[Meta SAM3](https://github.com/facebookresearch/sam3)、
[Transformers SAM3](https://huggingface.co/docs/transformers/model_doc/sam3)、
[SAM3 Tracker](https://huggingface.co/docs/transformers/model_doc/sam3_tracker)。
