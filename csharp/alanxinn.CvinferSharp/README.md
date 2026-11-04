# alanxinn.CvinferSharp.TensorRT10

`.NET` 封装后端隔离的 native API，支持 `net472`、`net6.0`、`net8.0`。

提供：

- YOLO11/YOLO26 检测、分割和 Pose，RF-DETR 检测与分割；
- SAM3 框/正负点/文本接口，真实权重验证状态见 `sam3/README.md`；
- SAM2.1 图片框/正负点/连续修正接口，待编译及真实推理验证，见 [SAM2.1 说明](../../sam2/README.md)；
- DINOv3 图像特征、patch/feature map 和余弦相似度，见 `dinov3/README.md`；
- BGR 图像推理；
- 检测框、Mask、耗时读取；
- TensorRT/Denglin 推理后端选择；
- 非托管资源释放。

## 安装

TensorRT 10 包名为 `alanxinn.CvinferSharp.TensorRT10`，本分支版本为 `0.2.1-trt10`。
包名按 TensorRT 大版本区分，程序集名及 `using alanxinn` 保持兼容。
0.2.1 默认覆盖 RTX 30/40/50，Core runtime 自动安装，无需开发 SDK 或手动配置 PATH。
Engine 仍需在目标显卡生成。新包尚需打包发布，见 [部署说明](../../docs/TRT10_DLL_DEPENDENCIES.md)。
已发布的 0.1.x 无后缀 TensorRT 包使用旧 SDK。

```powershell
dotnet add package alanxinn.CvinferSharp.TensorRT10 --version 0.2.1-trt10 --source E:/gitproject/DualAIVison/cvinfer/package/TensorRT10
```

只使用 Denglin 后端时安装：

```powershell
dotnet add package alanxinn.CvinferSharp.Denglin --version 0.1.4
```

需要在同一个应用中按配置切换两个后端时，可以同时引用以上两个包；
本分支 Demo 默认引用本地 TensorRT 项目。每个包只加载自己对应的 native API DLL。

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

`alanxinn.CvinferSharp.TensorRT10` 及其自动依赖的 Core runtime 合计提供：

```text
alanxinn_cvinfer_trt.dll
alanxinn_cvinfer_sam3.dll
alanxinn_cvinfer_sam2.dll
alanxinn_cvinfer_dinov3.dll
nvinfer_10.dll
nvinfer_plugin_10.dll
cudart64_12.dll
opencv_world*.dll
```

`nvonnxparser_10.dll`、TRT builder resource 不在该推理包中，由 builder 的 `TensorRtRuntime`
单独提供。cuBLAS/cuDNN/NVRTC 是特定插件或旧算子的可选依赖，默认环境也不包含，
需要时显式启用 `IncludeOptionalLibraries`。若某个已构建 engine 依赖额外 TensorRT plugin，应用还需自行部署
对应 plugin DLL。

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
