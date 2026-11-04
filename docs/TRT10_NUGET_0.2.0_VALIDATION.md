# TensorRT10 NuGet 0.2.0 验证记录

日期：2026-11-04。Windows x64，RTX 4060 8GB。仅生成本地包，未远程发布。

已编译 native（含 SAM2.1）及 net472/net6.0/net8.0 封装。实际运行验证使用 .NET 8。
默认携带 SM86/SM89/SM120 资源；没有 RTX30/50 实机测试，不表示 engine 可跨 GPU 使用。

## 本地产物

目录：`package/TensorRT10`。旧 `0.2.0-trt10` 包和旧 cuDNN 完整环境 ZIP 已删除。
所有包均低于 250,000,000 字节，版本和同版本依赖已核对。两个主包及 8 个运行库依赖包需一起发布。

| 文件 | bytes |
| --- | ---: |
| alanxinn.CvinferSharp.TensorRT10.0.2.0.nupkg | 23289002 |
| alanxinn.TensorRtOnnxBuilder.TensorRT10.0.2.0.nupkg | 1485703 |
| alanxinn.TensorRtRuntime.Core.TensorRT10.0.2.0.nupkg | 229618233 |
| alanxinn.TensorRtRuntime.PTX.Part1.TensorRT10.0.2.0.nupkg | 138787096 |
| alanxinn.TensorRtRuntime.PTX.Part2.TensorRT10.0.2.0.nupkg | 144974050 |
| alanxinn.TensorRtRuntime.PTX.Part3.TensorRT10.0.2.0.nupkg | 11823685 |
| alanxinn.TensorRtRuntime.SM120.Part1.TensorRT10.0.2.0.nupkg | 166499446 |
| alanxinn.TensorRtRuntime.SM120.Part2.TensorRT10.0.2.0.nupkg | 103767515 |
| alanxinn.TensorRtRuntime.SM86.TensorRT10.0.2.0.nupkg | 166646394 |
| alanxinn.TensorRtRuntime.SM89.TensorRT10.0.2.0.nupkg | 176479469 |
| tensorrt10.16.1.11-cuda12.9-sm86-sm89-sm120-runtime-win-x64.zip | 1142350910 |

完整 SHA256：`package/TensorRT10/artifacts.json`。

## 安装与实际调用

独立消费项目仅通过 PackageReference 从本地 feed 还原，不复制 SDK/ZIP DLL，
空运行缓存强制使用 NuGet 依赖。PTX/SM120 分片自动合并、SHA256 校验和运行库加载成功。
YOLO11n 用新包重建 engine，并验证同进程 builder → inference → builder。
未加载 cuDNN/cuBLAS/NVRTC。运行 API 返回 CUDA 12090、TensorRT 101601。

| 用例 | 状态 |
| --- | --- |
| yolo11n | PASS |
| yolo11n-seg | PASS |
| yolo11n-pose | PASS |
| yolo26n | PASS |
| yolo26n-seg | PASS |
| yolo26n-pose | PASS |
| rfdetr-nano | PASS |
| rfdetr-seg-nano | PASS |
| sam3.1 | PASS |
| dinov3-dinov3_convnext | PASS |
| dinov3-dinov3_vit | PASS |

11 项均通过。DINOv3 两项为随机权重算子检查，不是预训练语义效果验证。
SAM2.1 已编译并打入 NuGet，tiny 权重已下载；尚无 engine bundle，真实推理验证明确跳过。

最后只更新包内 README 后重新打包，所有 managed/native DLL SHA256 与已测试包完全一致。
完整报告：`cache/nuget-trt10-validation/validation.json`；日志：`package/TensorRT10/logs`。
Python 测试/验证脚本仅保留本地并由 Git 忽略。
