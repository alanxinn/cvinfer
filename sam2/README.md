# SAM2.1 图片交互分割

按已确认的范围，提供框选、正点、负点和连续掩码修正；支持纯点提示。
2026-11-05 已使用官方 tiny 预训练权重完成真实图片对照；ONNX 通过实际安装的
`alanxinn.TensorRtOnnxBuilder.TensorRT10/0.2.0` 转换，再用 C# NuGet 验证推理。
9 组提示的 mask IoU 对照官方 PyTorch 均为 1.0，详见 [验证记录](VALIDATION.md)。
语言描述和视频功能留待后续。

## 权重与导出

支持 Meta 官方 tiny、small、base-plus、large，默认 tiny。四种模型使用同一组公开接口，
但分别导出 engine。权重不随 NuGet 分发。

已下载 `models/sam2.1/sam2.1_hiera_tiny.pt`，156,008,466 字节，SHA256：

```text
7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69
```

该文件已通过官方模型严格加载和真实图片推理对照验证。

在单独 Python 3.10+ 环境准备依赖（SAM2 官方源码固定到明确提交，关闭可选 CUDA 扩展）：

```powershell
$env:SAM2_BUILD_CUDA = '0'
python -m pip install -r scripts/requirements-sam2.txt
python scripts/export_sam2.py --variant tiny
python scripts/build_sam2_engines.py
```

导出脚本默认从 Meta 下载缺失权重，可用 `--no-download` 禁止下载。
本地权重用 `--checkpoint PATH`，同时选择匹配的 `--variant`；不匹配会在官方严格加载时失败。
其他规格建议分别指定 `--output cache/sam2.1-large/onnx`、`--onnx` 和 engine 输出目录，避免互相覆盖。
导出记录权重 SHA256、配置和转换状态，绝不以随机权重替代。

Bundle 包含 `image_encoder.engine`、`mask_decoder.engine`、`bundle.json`。
默认 FP32，禁用 TF32；编码器 FP16 为显式选项，须先对照验证精度。
动态提示范围为 1 至 `max_points + 2`，额外两个位置用于框角。
`--max-points` 默认 32；C++／C# 构造参数须不超过导出的限制。
需要与本机 TensorRT10 SDK 相符的原生库；后续编译时 CMake 默认启用 `BUILD_SAM2=ON`。
每台目标 GPU 构建自己的 engine；RTX30/40/50 沿用本分支运行环境，仍受显存和驱动约束。

## C# 用法

```csharp
using alanxinn;

using var sam = new Sam2Segmentor("cache/sam2.1/engines");
sam.SetImage(bgrBytes, width, height); // 连续排列的 BGR uint8，无行间补齐
sam.SetBox(50, 390, 245, 910);
var initial = sam.Predict();
sam.AddPoint(150, 600, positive: true);
var positive = sam.Predict();
sam.AddPoint(240, 850, positive: false);
var refined = sam.Predict(usePreviousMask: true);
sam.RemovePoint(1); // 删除第二个用户点，不包含框角；丢弃上次 mask
var corrected = sam.Predict();
sam.ClearPoints(); // 保留框
sam.ClearBox();    // 保留点
sam.ClearPrompts();// 保留图像特征
sam.AddPoint(150, 600, positive: true);
var pointOnly = sam.Predict();
```

坐标均为原图像素 XY／XYXY。框可以达到图像右下边界，点必须在图像内。
`SetImage` 清空全部提示；`SetBox` 清空已有点；删除或清空提示会丢弃上一张低分辨率掩码。
`AddPoint` 保留上一张掩码，只有 `Predict(true)` 才显式使用它；首次预测传 true 会报错。
返回质量分数最高的三个候选之一，`Pixels` 为原图大小的 0/255 掩码，`Box` 是该掩码的紧致边界。
空掩码的框全零。每个实例独立缓存图像特征，不能由多线程同时调用。
首次设置图片运行编码器并释放其 GPU 上下文；后续点击只运行解码器。

C ABI 位于 `include/sam2/sam2_api.h`，Python 封装为 `scripts/sam2_infer.py`。
`alanxinn_cvinfer_sam2.dll` 已随 TensorRT10 0.2.0 推理包生成；Denglin 包不包含此接口。

## 推理与后续验证

以下命令用于重现原生推理及官方参考对照。已生成的 tiny engine 位于 `cache/sam2.1/engines`；本次 engine 的 NuGet 转换记录见 [验证记录](VALIDATION.md)：

```powershell
python scripts/sam2_infer.py --image cache/trt10-models/bus.jpg --box 50 390 245 910 --positive 150 600 --negative 240 850 --refine
python scripts/validate_sam2_real.py --image cache/trt10-models/bus.jpg --box 50 390 245 910 --positive 150 600 --negative 240 850
```

图片路径和坐标请按实际图片调整。验证脚本直接对照官方 `SAM2ImagePredictor`，
核对 checkpoint SHA256、纯点、框、正负点、mask 反馈、删除点、清空／重设提示和非法输入；
报告记录原图 mask IoU、分数差异和输出 PNG。默认 IoU ≥ 0.98、分数误差 ≤ 0.01，
并拒绝所有结果均为空的无效用例。该检查是转换一致性验证，不代表人工标注精度评估。
NuGet 消费测试会自动发现已有的真实 SAM2.1 engine bundle；缺失时报告跳过。

接口和预处理依据 [Meta 官方代码](https://github.com/facebookresearch/sam2)、
[图片预测器](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)；
权重来源为 [Meta 官方下载目录](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)。
