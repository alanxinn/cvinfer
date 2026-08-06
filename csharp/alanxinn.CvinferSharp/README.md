# alanxinn.CvinferSharp

版本：`0.1.0`

`alanxinn.CvinferSharp` 是 `alanxinn_cvinfer.dll` 的 .NET C# 封装，支持 `net472`、`net6.0` 和 `net8.0`，提供：

- YOLO11 检测和分割模型加载；
- BGR 图像推理；
- 检测框、分割框和 Mask 读取；
- CPU/CUDA 前处理和后处理配置；
- 前处理、推理、后处理耗时读取；
- P/Invoke 句柄和非托管资源释放。

## 安装

```powershell
dotnet add package alanxinn.CvinferSharp --version 0.1.0
```

## 使用示例

```csharp
using alanxinn;

var config = new CvinferConfig
{
    ClassCount = 80,
    PreprocessBackend = CvinferComputeBackend.Cuda,
    PostprocessBackend = CvinferComputeBackend.Cuda
};

using var model = new CvinferModel(
    "yolo11n.engine",
    CvinferModelType.YoloDetection,
    config);

using var result = model.InferBgr(bgrBytes, width, height, 3);
for (var i = 0; i < result.Count; i++)
{
    var detection = result.GetDetection(i);
    Console.WriteLine(detection.Box);
}
```

输入宽高由 TensorRT engine 自动确定。固定输入 engine 读取固定 shape；
动态输入 engine 读取 optimization profile 0 的 `OPT` shape。`CvinferConfig`
中的 `InputWidth` 和 `InputHeight` 仅为底层 ABI 兼容保留，默认值为 0；
默认值为 0 时自动读取 engine，设置为正值时才会显式覆盖自动尺寸。

## Native DLL 部署

类库打包时会从仓库构建目录复制 native 文件。CUDA DLL 只有在私有 CUDA
构建产物存在时才会被打包：

```text
build/ninja-msvc-private-cuda-release/bin/
```

没有私有构建目录时使用：

```text
build/ninja-msvc-release/bin/
```

复制到：

```text
native/win-x64/alanxinn_cvinfer.dll
native/win-x64/alanxinn_cvinfer_cuda.dll（可选，启用 CUDA 前后处理）
native/win-x64/nvinfer.dll
native/win-x64/cudart64_*.dll
native/win-x64/opencv_world*.dll（仅保留 Release 版本）
```

打包后文件位于 NuGet 包：

```text
runtimes/win-x64/native/
```

如果构建目录不存在或缺少某个 DLL，项目不会自动下载依赖；需要先完成 C++ 项目构建，或者手动将对应 DLL 放入 `native/win-x64` 后再打包。`native/win-x64` 是类库项目的打包暂存目录，不是使用方应用的运行目录。

NuGet 包安装到使用方项目后，native DLL 通过 NuGet 的 runtime native 资产参与运行时加载，不复制到应用输出根目录。主 C++ DLL 会从自身所在的 `runtimes/win-x64/native` 目录定位 CUDA backend。为保留该目录结构，应用只需使用 x64 平台，不要额外设置 `RuntimeIdentifier=win-x64`。

重新生成相同版本的本地 NuGet 包后，使用方需要清理 NuGet 全局缓存并重新还原，
否则可能继续使用旧包中的 native 文件：

```powershell
dotnet nuget locals global-packages --clear
dotnet restore --force-evaluate
```

## 版本说明

- NuGet 包版本：`0.1.0`
- AssemblyVersion：`0.1.0.0`
- FileVersion：`0.1.0.0`
- TargetFrameworks：`net472`、`net6.0`、`net8.0`
- 平台：`x64`
