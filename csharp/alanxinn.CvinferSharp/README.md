# alanxinn.CvinferSharp

`.NET` 封装后端隔离的 native API，支持 `net472`、`net6.0`、`net8.0`。

提供：

- YOLO11/RF-DETR 检测、分割和 Pose；
- BGR 图像推理；
- 检测框、Mask、耗时读取；
- CPU/CUDA backend 配置；
- TensorRT/Denglin 推理后端选择；
- 非托管资源释放。

## 安装

```powershell
dotnet add package alanxinn.CvinferSharp --version 0.1.4
```

只使用 Denglin 后端时安装：

```powershell
dotnet add package alanxinn.CvinferSharp.Denglin --version 0.1.4
```

需要在同一个应用中按配置切换两个后端时，可以同时引用以上两个包；
Demo 就采用这种方式。每个包只加载自己对应的 native API DLL。

## 示例

```csharp
using alanxinn;

var config = new CvinferConfig
{
    // 0 或负值：按 engine 输出自动推断类别数
    ClassCount = 0,
    InferenceBackend = CvinferInferenceBackend.TensorRt,
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

`CvinferConfig.InferenceBackend` 默认为 `TensorRt`。使用 Denglin 时，请安装
`alanxinn.CvinferSharp.Denglin` 包；该包包含启用 `BUILD_DENGLIN_BACKEND` 构建的
native DLL 及其 SDK runtime。

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

`alanxinn.CvinferSharp`（TensorRT 包）包含：

```text
alanxinn_cvinfer_trt.dll
nvinfer.dll
cudart64_*.dll
zlibwapi.dll
opencv_world*.dll
```

`nvinfer_plugin.dll`、`nvonnxparser.dll`、cuBLAS、cuDNN 等 builder 依赖不在
该推理包中；它们由 `alanxinn.TensorRtOnnxBuilder` 的 `TensorRtRuntime`
单独提供。若某个已构建 engine 依赖额外 TensorRT plugin，应用还需自行部署
对应 plugin DLL。

CUDA 前后处理还需要可选的 `alanxinn_cvinfer_cuda.dll`，仍放在同一个
`runtimes/win-x64/native/` 资产目录。

`alanxinn.CvinferSharp.Denglin`（Denglin 包）包含：

```text
alanxinn_cvinfer_denglin.dll
dlnne.dll
dlnne_plugin.dll
curt.dll、Executable_tc.dll、dleol.dll、hcrt.dll
dlhal2.dll、dlutil.dll、utils_common.dll、utils_log.dll
device_properties.dll、dltu.dll
hc2/hc3 下的 dlhc2.dll（SDK 提供 dlhc3.dll 时使用 dlhc3.dll）
```

两个包可以同时引用，但模型后端必须与对应的 native API DLL 匹配。
NuGet 会把对应资产放到应用输出的 `runtimes/win-x64/native/`，应用无需把
DLL 手工复制到 exe 根目录；Denglin 的 `hc2`/`hc3` 目录由包内构建目标
自动部署。
