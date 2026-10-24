# alanxinn.DenglinOnnxBuilder

通过 Denglin SDK `nnexec` 工具将 ONNX 转为 Denglin serialized engine，不执行推理。
转换过程由下载缓存中的 Denglin SDK/nnexec 直接完成，不依赖项目内 C++ builder DLL。

```powershell
dotnet add package alanxinn.DenglinOnnxBuilder --version 0.1.4
```

```csharp
using alanxinn.Denglin;

var result = DenglinOnnxBuilder.Build(
    "model.onnx",
    "model.denglin.engine");
```

Builder 会从 ONNX graph 自动选择唯一运行时输入，读取静态 shape 和输出名称。当前封装
面向单 runtime 输入的完整 ONNX graph；动态 ONNX 维度会按下方 fallback 规则物化。

动态 shape 在未提供 profile 时会按 `DynamicBatchSize` 和 `DynamicSpatialSize`
物化为静态 shape；当前 nnexec 链路暂不接受 `OptimizationProfiles`，`InputShape`
也必须是已物化的静态 shape。

`EngineFormat` 当前必须使用默认的 `RawEngine`，输出可直接加载的 Denglin engine。

## 环境要求

- Windows x64 和登临计算卡驱动。
- 完整 Denglin SDK 和 SDK Python 依赖（NumPy、ONNX、SciPy、setuptools、dataclasses 及其传递依赖）。默认只使用版本化下载缓存；缓存缺少时
  自动下载 GitHub Release ZIP，不扫描本机 Conda、PATH 或其他 SDK。
- 默认只使用版本化下载缓存。提供 `DenglinBuildOptions.SdkRoot` 或 `PythonDirectory`
  时会优先使用指定的本地 SDK/Python；运行前会实际导入 NumPy、ONNX、SciPy、setuptools、
  dataclasses 和 SDK `dl` 模块进行依赖检查。

Builder 会直接调用 SDK 官方 `nnexec.exe`：传入 ONNX、输入/输出节点、shape、cluster
配置和 `--saveEngine`，由 nnexec 完成 Parser、Builder 和序列化；不再调用
`python -m dl convert`、`python -m dl lower` 或项目内 C++ Builder DLL。运行时自动设置
`PATH`、`PYTHONPATH`、`PYTHONNOUSERSITE` 和 `TEST_DATA_ROOT_PATH`。

对应的 SDK 命令形式如下（`--nocheck` 用于只构建并保存 engine）：

```powershell
& E:\dlgpu\sdk\bin\nnexec.exe .\test\yolo11n.onnx `
  --input images --output output0 --shape images:1x3x640x640 `
  --maxBatch 1 --clusterConfig 0 --batch 1 --nocheck `
  --saveEngine .\cache\yolo11n.denglin.raw.engine
```

## 自动验证

仓库根目录执行以下命令可批量转换五个测试模型：

```powershell
.\scripts\test_denglin_models.ps1 -BuildOnly -ContinueOnError
```

去掉 `-BuildOnly` 并指定可执行文件时，还会用 C# Demo 对每个 engine 执行一次
Denglin 推理。脚本会写入 `validation.json`；只有所有模型都生成非空 engine（以及
可选的推理也成功）时才返回退出码 0。nnexec 返回 Windows `0xC0000005`/`0xC0000409`
且没有 engine 会被明确记为失败，不会把中间文件误判为转换成功。

运行时下载配置：

```text
ALANXINN_DENGLIN_RUNTIME_URL       GitHub/内网 ZIP 地址
ALANXINN_DENGLIN_RUNTIME_SHA256   可选 SHA256 校验值
ALANXINN_DENGLIN_RUNTIME_CACHE     可选缓存目录
ALANXINN_DENGLIN_NO_DOWNLOAD=1    禁止自动下载
ALANXINN_DENGLIN_ALLOW_LOCAL_RUNTIME=1  开发时允许使用本地 SDK/Python
```

SDK/Python 由于体积和授权原因不默认放进主 NuGet 包。使用
`tools\Pack-DenglinRuntime.ps1` 可以从已授权安装目录创建
`alanxinn-denglin-runtime-win-x64-<version>.zip`，再上传到 Release 或内部镜像。

```powershell
.\tools\Pack-DenglinRuntime.ps1 `
  -SdkRoot E:\dlgpu\sdk `
  -PythonRoot "C:\Program Files\Python312" `
  -OutputZip .\artifacts\alanxinn-denglin-runtime-win-x64-0.1.4.zip
```

脚本默认排除 TensorFlow、PyTorch、Paddle、JAX 等与 ONNX 转换无关的大型训练框架，避免
无用文件和 Windows 超长路径。确实需要这些包时可追加 `-IncludeOptionalPythonPackages`。

若必须发布离线 NuGet，可在打包 Builder 时设置
`-p:DenglinRuntimeBundleSource=<包含 sdk/ 和 python/ 的目录>`；普通 NuGet 用户则由
Builder 首次转换时按 Release URL 下载并缓存。