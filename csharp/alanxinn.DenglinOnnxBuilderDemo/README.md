# Denglin ONNX 转换 Demo

该 Demo 通过 `alanxinn.DenglinOnnxBuilder` NuGet 包调用 Denglin SDK 的 `nnexec.exe`，
使用 `--saveEngine` 直接将 ONNX 转为 Denglin engine，不执行额外的 Python/C++ Builder 流程。

## 准备

先生成并打包 Builder：

```powershell
dotnet pack .\csharp\alanxinn.DenglinOnnxBuilder\alanxinn.DenglinOnnxBuilder.csproj -c Release
```

Demo 默认从 `csharp/alanxinn.DenglinOnnxBuilder/bin/Denglin` 查找本地 NuGet 包。

## 默认用法

Demo 会直接读取 ONNX protobuf 元数据，自动识别唯一的运行时输入、输入 shape、数据类型和所有输出；
构建时由 ONNX 元数据自动发现图输出，并将输入 shape 传给 Denglin CLI。
默认使用版本化下载缓存中的 SDK/Python；无需手动填写 SDK、Python、input name 或 input
shape：

```powershell
dotnet run --project .\csharp\alanxinn.DenglinOnnxBuilderDemo -- `
  --onnx .\test\yolo11n.onnx `
  --engine .\cache\yolo11n.denglin.raw.engine
```

构建前会打印类似以下信息：

```text
Input : images [1,3,640,640] float32
Output: output0 [1,84,8400] float32
```

当前转换链路使用一个 runtime 输入并输出 raw Denglin engine。动态 ONNX shape 默认将
batch 设为 1、空间维度设为 640；也可用静态 `--input-shape` 显式物化。

## Denglin 运行时

转换阶段需要完整 Denglin SDK（包含 `bin\nnexec.exe` 及其 Python/运行时依赖）。默认启用下载缓存独占模式：

1. 检查 `%LOCALAPPDATA%\alanxinn\DenglinOnnxBuilder\0.1.4\win-x64`；
2. 缓存不存在或不完整时，下载 GitHub Release runtime ZIP；
3. 解压后只使用缓存中的 `sdk/` 和 `python/`，不扫描本机 PATH、Conda 或其他 SDK。

Demo 提供 `--sdk-root` 或 `--python` 时会优先使用对应的本地运行时；Python 环境必须包含
SDK 所需依赖。运行前会实际导入 NumPy、ONNX、SciPy、setuptools、dataclasses 和 SDK `dl`
模块。未提供时，Demo 只使用下载缓存。

```powershell
dotnet run --project .\csharp\alanxinn.DenglinOnnxBuilderDemo -- `
  --onnx .\test\yolo11n.onnx `
  --engine .\cache\yolo11n.denglin.raw.engine `
  --sdk-root E:\dlgpu\sdk
```

默认下载地址为：

```text
https://github.com/alanxinn/cvinfer/releases/download/v0.1.4/alanxinn-denglin-runtime-win-x64-0.1.4.zip
```

企业内网或离线环境可通过环境变量覆盖：

```powershell
$env:ALANXINN_DENGLIN_RUNTIME_URL = "https://mirror.example/denglin-runtime.zip"
$env:ALANXINN_DENGLIN_RUNTIME_SHA256 = "<sha256>"
$env:ALANXINN_DENGLIN_RUNTIME_CACHE = "D:\runtime\denglin"
```

设置 `ALANXINN_DENGLIN_NO_DOWNLOAD=1` 可禁止网络下载，此时必须预置完整的下载缓存。
首次下载时 Demo 会显示下载百分比、已下载大小和速度；若服务器没有返回文件总长度，
百分比会显示为 `?%`，但仍会显示实时下载量。
由于 Denglin SDK/Python 体积较大且可能受厂商授权限制，主 NuGet 包不包含 C++ builder；
`tools\Pack-DenglinRuntime.ps1` 用于将授权的 SDK/Python 目录打成 Release ZIP，发布后由
Builder 按版本和 SHA256 校验自动缓存。
