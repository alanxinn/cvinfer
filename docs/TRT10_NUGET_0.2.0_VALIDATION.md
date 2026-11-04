# TensorRT10 NuGet 0.2.0 验证记录

日期：2026-11-04。Windows x64，RTX 4060 8GB。仅生成本地包，未远程发布。

后续 2026-11-05 已使用本页同一批 NuGet 包完成 SAM2.1 tiny 的 ONNX 转换及真实推理验证，见 [SAM2.1 记录](../sam2/VALIDATION.md)。下文保留 11-04 的通用包验证范围。

已编译 native（含 SAM2.1）及 net472/net6.0/net8.0 封装。两个主包分别在 net472/net6.0/net8.0 中独立还原、编译和实际运行，6 项均通过。
默认携带 SM86/SM89/SM120 资源；没有 RTX30/50 实机测试，不表示 engine 可跨 GPU 使用。

## 本地产物

目录：`package/TensorRT10`。旧 `0.2.0-trt10` 包和旧 cuDNN 完整环境 ZIP 已删除。
所有包均低于 250,000,000 字节，版本和同版本依赖已核对。两个主包及 8 个运行库依赖包需一起发布。

| 文件 | bytes |
| --- | ---: |
| alanxinn.CvinferSharp.TensorRT10.0.2.0.nupkg | 23289164 |
| alanxinn.TensorRtOnnxBuilder.TensorRT10.0.2.0.nupkg | 1485866 |
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

修复部署规则没有改变 managed/native DLL；最终包还完成了下面的独立复核。
完整报告：`cache/nuget-trt10-validation/validation.json`；日志：`package/TensorRT10/logs`。
Python 测试/验证脚本仅保留本地并由 Git 忽略。

## 最终包独立复核与修复

用户要求再次确认后，发现 .NET Framework 4.7.2 自动推断 `win7-x64`，原部署规则把
原生 DLL 复制到该 RID 目录，而加载器查找 `win-x64`，导致默认安装后不能运行。
已修复共享 buildTransitive targets，Windows x64 派生 RID 统一部署到 `runtimes/win-x64/native`；
两个 0.2.0 主包已覆盖重打包，artifacts.json 的大小和 SHA256 已更新。

修复后使用新的独立 NuGet 缓存复核：

| 独立消费项目 | net472 | net6.0 | net8.0 |
| --- | --- | --- | --- |
| 仅安装推理主包，真实 YOLO 推理 | PASS | PASS | PASS |
| 仅安装转换主包，实际 ONNX → engine | PASS | PASS | PASS |

推理项目只还原主包和 Core，共 2 包；转换项目只还原主包及其 8 个依赖，共 9 包。
清除进程内 SDK 环境变量与 PATH，禁止运行库下载和 SDK 回退，核对 native DLL 均来自
消费输出或 NuGet 资源合并缓存；CUDA/OpenCV 也纳入来源检查。

对修复后最终包再运行 11 项模型消费检查，全部通过；本轮模型检查复用匹配引擎，
转换功能由上述三个独立转换项目实际验证。共 17 项实际调用检查通过。
10 个包均完成 CRC、SHA256、依赖版本与包大小检查，不只是编译检查。

本轮最终报告：`cache/nuget-trt10-recheck-20261104/audit-summary.json`。
失败复现与修复前日志保留在 `cache/nuget-trt10-final-audit-20261104`，用于追溯。
SAM2.1 真实推理、DINOv3 预训练语义效果以及 RTX30/50 实机仍未验证。
此前若已安装旧的同版本 0.2.0，需清除该包版本的缓存后重新还原，不能复用旧文件。
