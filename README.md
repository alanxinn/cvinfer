# alanxinn_cvinfer

## 项目概览

本项目将 TensorRT 通用推理能力、YOLO 模型适配、C ABI 动态库和 C# NuGet 封装分层组织。新增模型只需要复用 `core`，实现模型自己的前处理和输出解码，不需要重复编写 TensorRT runtime、context、stream 和显存管理代码。

```text
C# / C++ 外部应用
        ↓
alanxinn_cvinfer.dll
        ↓
alanxinn_cvinfer_yolo
        ↓
alanxinn_cvinfer_core
        ↓
TensorRT / CUDA / OpenCV
```

可选加速路径：

```text
alanxinn_cvinfer_yolo
        ↓（显式选择 CUDA backend 时动态检查）
alanxinn_cvinfer_cuda.dll
```

只有使用私有 backend 构建主库时才启用该路径；公开 CPU 构建不会链接或加载 CUDA backend。

## NuGet 开箱即用

C# 应用可以直接从 [NuGet 包页面](https://www.nuget.org/packages/alanxinn.CvinferSharp) 下载并引用 `alanxinn.CvinferSharp` NuGet 包，无需在开发机上
额外安装或配置复杂的 CUDA、TensorRT、OpenCV 开发环境。NuGet 包已经包含
运行所需的 `alanxinn_cvinfer.dll`、TensorRT、CUDA Runtime 和 OpenCV Runtime
DLL，引用类库后即可加载 engine 并执行推理。
当前 `alanxinn.CvinferSharp` NuGet 包对应 CUDA Runtime 11.8 和 TensorRT Runtime 8.1.6，相关运行库已随包提供。

运行时仍需要满足以下条件：

- Windows x64；
- 已安装兼容的 NVIDIA 显卡驱动；
- 准备对应的 TensorRT engine 文件和输入图像；
- 如果显式选择 CUDA 前处理或后处理，需要使用包含
  `alanxinn_cvinfer_cuda.dll` 的私有 CUDA NuGet 包；使用 `auto` 时无需该 DLL。

当前 NuGet/C# Demo 支持的模型如下：

| 模型 | 类型参数 | 输出 |
| --- | --- | --- |
| YOLO11 Detection | `det` | 检测框、类别、置信度 |
| YOLO11 Segmentation | `seg` | 检测框、类别、置信度和 Mask |

输入宽高由 TensorRT engine 自动读取，固定输入使用固定 shape，动态输入使用
optimization profile 0 的 `OPT` shape。C# Demo 默认使用
`--preprocess auto --postprocess auto`。

## CPU 前后处理基准测试

测试日期：2026-08-06  
测试设备：NVIDIA GeForce RTX 4060（8 GB，驱动 591.86）；CPU 为 Intel 12490F。TensorRT 推理使用该 NVIDIA GPU，
前处理和后处理均使用 CPU。

测试条件：

```text
输入图像：810x1080
warmup：100 次
正式统计：1000 次
检测模型：yolo11n.engine
分割模型：yolo11n-seg.engine
```

| 模型 | 阶段 | 平均耗时（ms） | 最小耗时（ms） | 最大耗时（ms） |
| --- | --- | ---: | ---: | ---: |
| `yolo11n.engine` | 前处理 | 1.9984 | 1.7884 | 3.0753 |
| `yolo11n.engine` | TensorRT 推理 | 3.5155 | 2.8160 | 5.9006 |
| `yolo11n.engine` | 后处理 | 2.0024 | 1.7712 | 2.8704 |
| `yolo11n.engine` | 总耗时 | 8.3656 | 7.2045 | 11.5039 |
| `yolo11n-seg.engine` | 前处理 | 1.9867 | 1.7937 | 3.6404 |
| `yolo11n-seg.engine` | TensorRT 推理 | 4.8961 | 4.2666 | 8.2253 |
| `yolo11n-seg.engine` | 后处理 | 4.0293 | 3.6666 | 8.6139 |
| `yolo11n-seg.engine` | 总耗时 | 11.7439 | 10.6901 | 19.4811 |

## 模块文档

- [core：TensorRT 核心库](core/README.md)
- CUDA backend：私有仓库维护，构建时通过 `ALANXINN_CUDA_BACKEND_DIR` 接入
- [yolo：YOLO 检测与分割适配库](yolo/README.md)
- [api：C ABI 动态库接口](api/README.md)
- [csharp：C# 类库与 Demo](csharp/README.md)

## 目录结构

```text
core/                     TensorRT 通用推理、Tensor、CUDA stream 和设备内存
yolo/                     YOLO 前处理、检测/分割解码、NMS、Mask 和 Demo
api/                      alanxinn_cvinfer.dll 的稳定 C ABI
csharp/                   C# 类库、NuGet 包和 Demo
private/cuda_backend/     私有 CUDA 前处理/后处理实现，不提交公开仓库
test/                     本地测试和生成文件目录
```

公开仓库只维护 `core`、`yolo`、`api` 和 `csharp`。`private/cuda_backend`
应作为独立私有仓库或本地私有 checkout 管理，根目录 `.gitignore` 已忽略该目录。

## 构建

### 普通 CPU/Auto 构建

普通构建不会编译 `private/cuda_backend`，`Cuda` backend 在该构建中不可用：

```powershell
cmake -S . -B build/public-release `
  -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCUDA_ROOT="C:/your/path/CUDA/v11.8" `
  -DTENSORRT_ROOT="C:/your/path/TensorRT-8.1.6_cuda-11.8" `
  -DOPENCV_ROOT="C:/your/path/opencv"
cmake --build build/public-release
```

### 私有 CUDA 构建

内部构建显式传入私有 backend 路径：

```powershell
cmake -S . -B build/private-cuda-release `
  -G Ninja `
  -DBUILD_CUDA_BACKEND=ON `
  -DALANXINN_CUDA_BACKEND_DIR="C:/your/private/cuda_backend" `
  -DALANXINN_PUBLIC_YOLO_INCLUDE_DIR="C:/your/project/cvinfer/yolo/include" `
  -DALANXINN_PUBLIC_CORE_INCLUDE_DIR="C:/your/project/cvinfer/core/include" `
  -DCUDA_ROOT="C:/your/path/CUDA/v11.8" `
  -DTENSORRT_ROOT="C:/your/path/TensorRT-8.1.6_cuda-11.8" `
  -DOPENCV_ROOT="C:/your/path/opencv"
cmake --build build/private-cuda-release
```

`yolo` 只依赖公开的 `yolo/yolo_cuda.h` 接口。未启用私有 backend 时，
项目使用 CPU stub 编译，显式选择 `Cuda` 会返回 backend 缺失错误；启用私有
backend 后，`alanxinn_cvinfer_cuda.dll` 通过延迟加载提供 CUDA 加速。生成的
CUDA DLL 会复制到当前构建目录的 `bin/`，导入库位于 `lib/`。

也可以使用预设：

```powershell
cmake --preset ninja-msvc-release
cmake --build --preset build-ninja-msvc-release

cmake --preset ninja-msvc-private-cuda-release
cmake --build --preset build-ninja-msvc-private-cuda-release
```

## 主要构建产物

```text
alanxinn_cvinfer.dll
alanxinn_cvinfer.lib
alanxinn_cvinfer_core.lib
alanxinn_cvinfer_yolo.lib
alanxinn_yolo11_det_core_demo.exe
alanxinn_yolo11_seg_core_demo.exe
```

启用私有 CUDA backend 后，额外生成：

```text
alanxinn_cvinfer_cuda.dll
alanxinn_cvinfer_cuda.lib
```

## 调用原则

- `core` 不依赖 YOLO 业务语义。
- `yolo` 通过 `core::TrtSession` 使用 TensorRT。
- `api` 只负责 C ABI、句柄、错误码和结果转换。
- C# 通过 `alanxinn_cvinfer.dll` 的 P/Invoke 调用，不直接接触 TensorRT 或 CUDA 类型。
- 单个模型句柄默认串行使用；并发推理应创建独立模型句柄。

## 当前能力

- YOLO11 检测和分割。
- CPU/CUDA 前处理和后处理配置。
- CUDA 前后处理独立 DLL，可选部署，缺失时 CPU/Auto 路径仍可用。
- 动态输入 shape 和按 tensor 名称选择输入输出；模型输入尺寸自动读取固定
  shape 或 optimization profile 0 的 `OPT` shape。
- 检测框、分割 Mask 和前处理/推理/后处理耗时统计。
- C# 命名参数 Demo、warmup 和 benchmark。
- `alanxinn.CvinferSharp` 支持 `net472`、`net6.0` 和 `net8.0`。
- `version.json` 固定主项目、CUDA backend 和 ABI 兼容版本。

详细接口、配置和使用方式请进入上面的模块文档。

