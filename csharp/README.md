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
- TensorRT backend 使用与 engine 匹配的 TensorRT/CUDA/OpenCV；Denglin backend
  使用匹配的 Denglin SDK/device runtime
- 应用使用 x64 平台

## 推理 Demo

```powershell
alanxinn.CvinferDemo.exe `
  --type yolo11-det `
  --model yolo11n.engine `
  --img bus.jpg `
  --warmup 100 `
  --iterations 1000 `
  --timing
```

`--type` 只支持：

```text
yolo11-det
yolo11-seg
yolo11-pose
rf-detr-det
rf-detr-seg
```

其他参数：

```text
--model PATH
--img PATH
--output PATH
--classes N                 可选；默认按 engine 输出自动推断
--backend tensorrt|denglin  推理引擎后端（Denglin 需启用 native backend）
--preprocess auto|cpu
--postprocess auto|cpu
--warmup N
--iterations N
--timing                    可选；开启预处理、推理和后处理计时
```

输入尺寸和类别数默认从 engine 输出读取；分割模式输出叠加图，Pose 模式输出关键点。非 benchmark 模式还保存
`<output>_mask_<index>.pgm`。

RF-DETR 分割 engine 输出为 `dets`、`labels`、`masks`。

计时默认关闭，添加 `--timing` 后才输出耗时统计。

## C# 调用

```csharp
using alanxinn;

var config = new CvinferConfig
{
    ClassCount = 0,
    InferenceBackend = CvinferInferenceBackend.TensorRt,
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

将 `InferenceBackend` 设置为 `CvinferInferenceBackend.Denglin` 后，native
层会通过公共 Session 工厂加载 Denglin ONNX/engine。未启用 Denglin 或缺少
运行时 DLL 时，模型创建会返回可读错误。

## 构建和打包

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release
.\csharp\alanxinn.CvinferSharp\pack-both.ps1 -Configuration Release
```

会生成两个包：

- `alanxinn.CvinferSharp`：TensorRT/CUDA/OpenCV runtime；
- `alanxinn.CvinferSharp.Denglin`：Denglin SDK runtime。

包输出位置：

```text
alanxinn.CvinferSharp/bin/TensorRT/Release/
alanxinn.CvinferSharp/bin/Denglin/
```

两个包的 native DLL 都位于 `runtimes/win-x64/native/`。应用只需引用
对应的 NuGet 包，不需要手工复制 DLL。Denglin 包另外通过包内构建目标
自动部署 `runtimes/win-x64/hc2`、`runtimes/win-x64/hc3`，供 SDK 的
`hcrt.dll` 相对路径加载。

TensorRT 包加载 `alanxinn_cvinfer_trt.dll`，Denglin 包加载
`alanxinn_cvinfer_denglin.dll`。两个包可以同时引用，运行时通过
`CvinferInferenceBackend` 选择后端；两套 native API DLL 不能互相替代，
也不能改回旧的无后缀 `alanxinn_cvinfer.dll`。

当前 Demo 已同时引用两个 0.1.4 包，直接通过 `--backend` 选择后端：

```powershell
dotnet run --project .\csharp\alanxinn.CvinferDemo `
  -- --type yolo11-det --backend tensorrt --model yolo11n.engine --img bus.jpg

dotnet run --project .\csharp\alanxinn.CvinferDemo `
  -- --type yolo11-det --backend denglin --model yolo11n.denglin.cache.raw.engine --img bus.jpg
```

修改同版本包后，清理 NuGet 缓存再还原。

ONNX 转换见：

- [alanxinn.TensorRtOnnxBuilder](alanxinn.TensorRtOnnxBuilder/README.md)
- [转换 Demo](alanxinn.TensorRtOnnxBuilderDemo/README.md)
