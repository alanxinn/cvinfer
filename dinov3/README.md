# DINOv3 图像特征推理

在 TensorRT 10 / CUDA 12.9 分支上提供 ViT 与 ConvNeXt backbone 的单图推理，
包括 C++、C ABI、C#、Python CLI、ONNX 导出和引擎构建。
特征可用于图像检索、相似度计算，以及后续分类、检测或分割任务头。

2026-11-08 已使用本地 ViT-S/16、ViT-S+/16、ConvNeXt-Tiny 预训练权重，
完成 ONNX 导出、实际 NuGet builder 转换、真实图片 PyTorch 对照和 C# NuGet 推理。
3 个模型各覆盖 7 组图像，FP32 最大相对 L2 误差为 `2.536e-5`，C# 与 C ABI 输出逐字节一致。
相似度示例与 PCA 特征图已生成；这些是定性示例，不代表有标注数据集上的语义精度评估。
具体来源和范围见 [验证记录](VALIDATION.md)。此前的随机权重算子测试保留于 `cache/dinov3-tiny`，
与真实权重产物 `cache/dinov3-real` 分开，CLI 会明确标记 `synthetic`。

## 输出

| 名称 | Shape | 内容 |
| --- | --- | --- |
| `global_embedding` | `[1,D]` | ViT 的 CLS，或 ConvNeXt 的归一化池化特征 |
| `normalized_embedding` | `[1,D]` | 全局特征的 L2 单位向量 |
| `patch_tokens` | `[1,Hf×Wf,D]` | 局部特征，已剔除 CLS 和 register tokens |
| `feature_map` | `[1,D,Hf,Wf]` | 同一局部特征的 NCHW 排列 |

所有外部输入／输出 tensor 均为 FP32。默认导出固定 `1×3×224×224`，支持配置导出尺寸。
零全局向量的归一化结果保持为零；C# 余弦相似度接口对零向量报错。
ViT-S/16 默认 `D=384`、特征网格 `14×14`；标准 ConvNeXt 的网格 stride 为 32。
导出尺寸必须是模型 stride 的正整数倍。当前接口每次处理一张原图，原图尺寸可任意，
内部缩放到引擎输入尺寸。额外的分类或检测任务头需单独接入。

## 构建与预训练权重

```powershell
.\scripts\Build-Trt10.ps1
py -3.10 -m venv cache/dinov3-python
cache/dinov3-python/Scripts/python.exe -m pip install -r scripts/requirements-dinov3.txt
```

默认权重为 [Meta ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)。
已有本地 safetensors snapshot 时无需登录，直接指定目录。本机真实模型示例：

```powershell
$checkpoint = 'D:/BaiduNetdiskDownload/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m'
python scripts/export_dinov3.py --checkpoint $checkpoint --output cache/dinov3-real/vits16/onnx
python scripts/build_dinov3_engine.py --onnx cache/dinov3-real/vits16/onnx --output cache/dinov3-real/vits16/engine
python scripts/dinov3_infer.py --engine cache/dinov3-real/vits16/engine/model.engine --image your-image.jpg --visualize
```

导出使用严格权重加载，缺失、额外或尺寸不匹配的参数都会报错，禁止静默补入随机参数。
`bundle.json` 保存本地配置、processor、safetensors 权重／分片的 SHA256。
更换模型规格时分别使用独立的输出目录。

需要从 Hugging Face 下载时，先获得模型访问权限，在本机登录后执行：

```powershell
cache/dinov3-python/Scripts/hf.exe auth login
cache/dinov3-python/Scripts/python.exe scripts/export_dinov3.py
cache/dinov3-python/Scripts/python.exe scripts/build_dinov3_engine.py
cache/dinov3-python/Scripts/python.exe scripts/validate_dinov3_real.py --image your-image.jpg
```

`--checkpoint D:/models/dinov3-hf` 可使用本地 HF snapshot，包含 config、safetensors、processor 文件。
原始 Meta `.pth` 文件需先转换为 HF 格式；本导出器不直接加载原始 `.pth`。
其他官方 ViT/ConvNeXt 变体可通过 `--checkpoint` 选择，实际显存需求随模型规模增加。
选择 ConvNeXt-tiny 时使用 `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`。

默认构建 FP32、关闭 TF32；`build_dinov3_engine.py --precision fp16` 允许 TensorRT 使用
FP16 tactics，外部 I/O 仍为 FP32，精度应按具体预训练模型对照验证。
构建脚本默认 `--builder nuget`，通过纯 PackageReference 的隔离 .NET 8 消费程序调用
`alanxinn.TensorRtOnnxBuilder.TensorRT10/0.2.0`，需要 .NET 8 SDK 和 `package/TensorRT10`
内的转换包及 8 个依赖包；可用 `--packages`、`--package-version` 指定来源与版本。
按包哈希隔离 NuGet 缓存，强制使用包内运行库，禁止回退到本机 TensorRT SDK 或外部 ZIP。
`model.build.json` 记录实际加载 DLL、包哈希、engine 哈希和转换耗时。
源码 C ABI 构建仍可显式选择 `--builder native`，不属于 NuGet 转换验证。
`bundle.json` 保存来源、revision、ONNX SHA256、维度和验证状态；配置的 processor
与引擎一起保存。大型 ONNX 的 external data 文件需保留在原导出目录中。

## 图像命令行

```powershell
python scripts/dinov3_infer.py --image your-image.jpg --visualize
python scripts/dinov3_infer.py --image first.jpg --compare second.jpg
```

默认读取 `cache/dinov3/engine/model.engine`，可使用 `--engine` 改路径。
结果写入 `cache/dinov3/result/features.npz` 和 `result.json`。
`--visualize` 生成 PCA 特征图预览，不代表语义类别；`--compare` 输出两张图像全局特征的余弦相似度。

原生 C++ demo：

```powershell
$env:PATH = "$PWD/build/trt10-release/bintrt;$env:PATH"
build/trt10-release/bin/alanxinn_dinov3_demo.exe `
  cache/dinov3/engine/model.engine your-image.jpg cache/dinov3/cpp-output
```

demo 保存四个 `.bin` FP32 tensor 和 shape 元数据 `tensors.tsv`。

## C#

```csharp
using var dino = new DinoV3FeatureExtractor(@"cache\dinov3\engine\model.engine");
var first = dino.InferBgr(bgr1, width1, height1);
var second = dino.InferBgr(bgr2, width2, height2);
double similarity = DinoV3FeatureExtractor.CosineSimilarity(first.GlobalEmbedding, second.GlobalEmbedding);
float[] embedding = first.NormalizedEmbedding;
long[] mapShape = first.FeatureMap.Shape;
float[] mapValues = first.FeatureMap.Values; // NCHW, row-major
```

支持 `net472`、`net6.0`、`net8.0`。结果是独立托管数组，后续推理／模型释放不影响旧结果。
`InferChw` 接受已预处理的 FP32 RGB CHW，可与官方 processor 或其他数据源对接。
原生 DLL 为 `alanxinn_cvinfer_dinov3.dll`，与同版本 `bintrt` runtime 一起部署。
每个实例避免并发调用；Denglin-only 包不包含 DINOv3 封装。

## 预处理和验证

预处理严格采用官方顺序：BGR→RGB→除以 255→浮点抗锯齿双线性缩放→ImageNet mean/std。
与 SAM3 的 uint8 resize 路径分开，避免额外量化。

```powershell
python scripts/test_dinov3_export.py
python scripts/build_dinov3_engine.py --onnx cache/dinov3-tiny/vit/onnx --output cache/dinov3-tiny/vit/engine
python scripts/build_dinov3_engine.py --onnx cache/dinov3-tiny/convnext/onnx --output cache/dinov3-tiny/convnext/engine
python scripts/test_dinov3_native.py
# 或一条命令：
.\scripts\Test-DinoV3.ps1 -Python path/to/python.exe -Synthetic
```

使用 `requirements-dinov3.txt` 对应环境。真实权重一键验证可执行：

```powershell
.\scripts\Test-DinoV3.ps1 -Python path/to/python.exe -Image your-image.jpg `
  -Checkpoint D:/models/dinov3-hf -OutputDirectory cache/dinov3-real/custom -SkipNativeBuild
```

`-SkipNativeBuild` 复用已有原生 DLL；转换仍通过 NuGet builder 执行。
`-PackagesDirectory` 可指定本地 NuGet 源目录。Python 测试／验证脚本按仓库约定仅保留本地，
不随 Git 分发；上述验证命令需要这些本地脚本。

目前测试覆盖两种架构、ViT 的 4 个 register token 剔除、FP32／允许 FP16、
大小图与 1×1 图、官方 processor 数值对照、CHW 输入、特征 map 排列、L2 归一化、
连续推理的数据有效性、非法输入／缓冲区大小／模型恢复，以及 C++／C# 调用。
实际结果见 [验证记录](VALIDATION.md)。

参考：[Meta DINOv3](https://github.com/facebookresearch/dinov3)、
[Transformers DINOv3](https://huggingface.co/docs/transformers/model_doc/dinov3)。
