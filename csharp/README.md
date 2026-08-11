# C# 项目

目录：

```text
alanxinn.CvinferSharp/              C# P/Invoke 封装
alanxinn.CvinferDemo/               推理 Demo
alanxinn.TensorRtOnnxBuilder/       ONNX 转 engine 类库
alanxinn.TensorRtOnnxBuilderDemo/   转换 Demo
```

## 运行环境

- Windows x64
- NVIDIA 驱动
- 与 engine 匹配的 TensorRT/CUDA/OpenCV
- 应用使用 x64 平台

## 推理 Demo

```powershell
alanxinn.CvinferDemo.exe `
  --type yolo11-det `
  --model yolo11n.engine `
  --img bus.jpg `
  --warmup 100 `
  --iterations 1000
```

`--type` 只支持：

```text
yolo11-det
yolo11-seg
yolo11-pose
yolo26-det
yolo26-seg
yolo26-pose
rf-detr-det
rf-detr-seg
```

其他参数：

```text
--model PATH
--img PATH
--output PATH
--classes N                 可选；默认按 engine 输出自动推断
--preprocess auto|cpu|cuda
--postprocess auto|cpu|cuda
--warmup N
--iterations N
```

输入尺寸和类别数默认从 engine 输出读取；分割模式输出叠加图，Pose 模式输出关键点。非 benchmark 模式还保存
`<output>_mask_<index>.pgm`。

RF-DETR 分割 engine 输出为 `dets`、`labels`、`masks`。
RF-DETR 前后处理目前仅支持 `auto`/`cpu`。
YOLO26 end2end 和 Pose 后处理使用 CPU；`preprocess cuda` 仍可用。

## C# 调用

```csharp
using alanxinn;

var config = new CvinferConfig
{
    ClassCount = 0,
    PreprocessBackend = CvinferComputeBackend.Auto,
    PostprocessBackend = CvinferComputeBackend.Auto
};

using var model = new CvinferModel(
    "rfdetr-seg-nano.engine",
    CvinferModelType.RfDetrSegmentation,
    config);

using var result = model.InferBgr(bgrBytes, width, height, 3);
var segmentation = result.GetSegmentation(0);
var mask = result.CopyMask(0);
```

## 构建和打包

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
dotnet pack .\alanxinn.CvinferSharp\alanxinn.CvinferSharp.csproj -c Release
```

native DLL 位于 NuGet 包的 `runtimes/win-x64/native/`。
修改同版本包后，清理 NuGet 缓存再还原。

ONNX 转换见：

- [alanxinn.TensorRtOnnxBuilder](alanxinn.TensorRtOnnxBuilder/README.md)
- [转换 Demo](alanxinn.TensorRtOnnxBuilderDemo/README.md)
