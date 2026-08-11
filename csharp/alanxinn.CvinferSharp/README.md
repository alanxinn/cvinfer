# alanxinn.CvinferSharp

`.NET` 封装 `alanxinn_cvinfer.dll`，支持 `net472`、`net6.0`、`net8.0`。

提供：

- YOLO11/RF-DETR 检测、分割和 Pose；
- BGR 图像推理；
- 检测框、Mask、耗时读取；
- CPU/CUDA backend 配置；
- 非托管资源释放。

## 安装

```powershell
dotnet add package alanxinn.CvinferSharp --version 0.1.3
```

## 示例

```csharp
using alanxinn;

var config = new CvinferConfig
{
    // 0 或负值：按 engine 输出自动推断类别数
    ClassCount = 0,
    PreprocessBackend = CvinferComputeBackend.Auto,
    PostprocessBackend = CvinferComputeBackend.Auto
};

using var model = new CvinferModel(
    "rf-detr-seg.engine",
    CvinferModelType.RfDetrSegmentation,
    config);

using var result = model.InferBgr(bgrBytes, width, height, 3);
var item = result.GetSegmentation(0);
var mask = result.CopyMask(0);
```

`CvinferConfig.EnableTiming` 默认为 `false`；需要读取 `result.Timing` 时设为 `true`。

Pose 使用 `CvinferModelType.Yolo11Pose`，通过 `result.GetPose(index)` 读取框和关键点。

支持的模型类型：

```text
CvinferModelType.Yolo11Detection
CvinferModelType.Yolo11Segmentation
CvinferModelType.Yolo11Pose
CvinferModelType.RfDetrDetection
CvinferModelType.RfDetrSegmentation
```

输入宽高从 engine 自动读取。动态输入使用 profile 0 的 `OPT` shape。

## Native DLL

NuGet native 文件位于：

```text
runtimes/win-x64/native/
```

至少需要：

```text
alanxinn_cvinfer.dll
nvinfer.dll
cudart64_*.dll
zlibwapi.dll
opencv_world*.dll
```

CUDA 前后处理还需要可选的 `alanxinn_cvinfer_cuda.dll`。
