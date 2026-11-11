# WeVisOne-VL 自动验证记录

日期：2026-11-08。分支：`trt10`。

**结论：官方 2B、512×512 视觉画布、4 实例像素 profile 下，全部 11 个 task 的真实图片功能检查通过。**
C ABI、C#、C++ 共 33 项 task 检查通过；缺失目标拒绝和指定框抠图另外 5 项检查通过。
双实例抠图及图像缓存复用另外 3 项检查通过。

## 环境与模型

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 8GB，驱动 616.92 |
| 原生环境 | Windows x64、TensorRT 10.16.1.11、CUDA 12.9、OpenCV 4.8 |
| 模型 | `tencent/WeVisOne-VL-2B` |
| 权重 revision | `fcf8c8ca7fa481b514005ef591c8b72bed39618a` |
| `model.safetensors` SHA256 | `5a843ab41b29946a712608a8f27e04149c5923d979d93973dac14ca4565f5f9e` |
| 官方推理源码 revision | `9a6bcd85404505df0d05cfb5d97512ccacf35dee` |
| 导出环境 | Python 3.12、PyTorch 2.11.0+cpu、Transformers 5.12.1、ONNX 1.23.2 |
| 视觉／语言 engine | FP16 张量与 FP32 归一化，强类型网络，关闭 TF32；workspace 1024 MiB |
| 像素 engine | FP32、1024×1024 解码画布、最多 4 实例；workspace 4096 MiB |
| 文本 profile | 最大预填充 2048，最大上下文 4096 |
| C# | 类库 net472／net6.0／net8.0 编译通过；实际推理使用 .NET 8 Windows Demo |

## 真实图片结果

图片来自固定版本官方仓库的 `livingroom.jpg` 和 `coffee_sign.jpg`。
测试器先生成统一的无损 PNG 输入，保证三种接口收到相同像素。

| task | 样例及检查 | C ABI | C# | C++ |
| --- | --- | --- | --- | --- |
| `detection` | 查询 sofa，返回非空有效框 | PASS | PASS | PASS |
| `referring` | 自然语言定位 sofa | PASS | PASS | PASS |
| `point` | sofa 点坐标，检查点类型与原图范围 | PASS | PASS | PASS |
| `count_box` | 图中一个 sofa，返回框及数量 1 | PASS | PASS | PASS |
| `count_point` | 图中一个 sofa，返回点及数量 1 | PASS | PASS | PASS |
| `ocr` | 招牌文字，识别到预期单词 | PASS | PASS | PASS |
| `ocr_grounding` | 定位招牌文字 METERS | PASS | PASS | PASS |
| `segmentation` | sofa 二值掩码，原图尺寸，值为 0/255 | PASS | PASS | PASS |
| `matting` | sofa 连续 Alpha，前景／背景样点及数值范围有效 | PASS | PASS | PASS |
| `visual_prompt` | 用参考框查找 sofa | PASS | PASS | PASS |
| `freeform` | 单句描述图片，回复非空且未截断 | PASS | PASS | PASS |

所有样例检查了非空回复、截断状态、几何坐标范围和对应输出类型。
原始回复、数量、拒绝及截断标记在三种接口之间一致。
C ABI 与 C# 的原图 FP32 Alpha 按绝对误差 ≤1e-6 比较通过；C++ Alpha PNG 按灰度差 ≤1 比较通过。
本记录覆盖样例功能及接口一致性，未运行公开数据集的完整精度 benchmark。

额外检查：

- 缺失目标 `a purple elephant`：C ABI／C#／C++ 均返回拒绝状态。
- 指定原图框抠图：C ABI／C# 均通过。
- 同时传入两个不同枕头框：C ABI／C# 均输出两个不同实例的有效 Alpha，接口间数值一致。
- 同一 C# 模型设置图片后，warmup=1、iterations=3 连续预测：结果与首次图片预测一致，图像特征缓存复用正常。

## 测试发现并修复的问题

1. **NFC 归一化缺失**：真实 tokenizer 带有 NFC normalizer，原导出器会拒绝加载。已支持导出该配置，Windows 原生使用 NormalizeString 对齐。
2. **FP16 视觉特征类型不一致**：位置插值特征为 FP32，与 FP16 patch 特征相加后导致 LayerNorm 类型错误。已按官方实现转换到输入类型。
3. **普通 FP16 构建使语言 logits 全零**：真实模型归一化中间值溢出，推理重复输出 `!`。视觉／语言图现使用强类型网络，保留 FP32 归一化。
4. **像素 profile 资源限制**：本机 16 实例 profile 在 1GB／4GB workspace 尝试中均未完成构建。4 实例、4GB workspace 成功；构建脚本默认使用不超过 4 实例，可显式调整。
5. **长转换失败后缺少可复用记录**：现在每个成功 graph 都保存输入哈希、engine 哈希、profile 和类型策略；失败时保留不可加载的 partial manifest，支持核验后分图重建。

归一化回归使用 FP16 输入 `[1000, 600]`：普通 FP16 构建得到 `[0, 0]`；
强类型构建得到预期的 `[1.212890625, 0.7275390625]`，最大绝对误差为 0。

## 其他检查与复现

- 官方 tokenizer 对照：13 组，覆盖中文、组合音标、emoji、空白、特殊 token 和 NUL。
- 小模型架构对照：29 项，覆盖视觉／DeepStack／特征图、语言预填充、KV Cache、三步解码、像素 cascade 和 FP16 视觉路径。
- C ABI 参数／布局／错误处理：5 项。
- 现有 TensorRT core／builder 回归：8 项，含动态 profile、host/device 输入、输出回拷开关、缓冲复用及 INT64。
- 动态语言与像素 ONNX 小图均实际转换为 TensorRT engine。

从仓库根目录执行：

```powershell
python scripts/wevisone_validate.py
python scripts/wevisone_validate_extra.py
python scripts/wevisone_validate_wrappers.py
python scripts/wevisone_validate_types.py
python scripts/test_trt10_contract.py --output cache/wevisone/validation/trt-contract
```

主报告：[真实 task JSON](../cache/wevisone/validation/real/report.json)。
额外报告：[双实例与缓存](../cache/wevisone/validation/extra/report.json)、
[归一化类型回归](../cache/wevisone/validation/types/report.json)、
[架构对照](../cache/wevisone/validation/wrapper_parity.json)。
每个 task／binding 的目录内保留原始日志、JSON、覆盖图及对应掩码／Alpha。
测试脚本使用的 Python 环境位于 `cache/wevisone/python`；上述命令中的 `python` 应替换为该环境或等价依赖环境。

## 尚未验证的范围

- 4B 模型，以及其他 GPU 架构。
- 16 实例像素 profile；本机本次构建失败，不能视为可用配置。
- 其他画布／上下文 profile 和公开 benchmark 精度。

## NuGet 0.2.1 独立消费验证

已生成两个 0.2.1 主包，复用八个未变化的 0.2.0 运行库包，保存在 `package/TensorRT10`，未远程发布。
重复的 0.2.1 环境包已逐项校验 payload 一致后移除；主包不再精确绑定相同版本的运行库。
旧 0.2.0 包保留，不包含本次新增代码。

- 全部十个包通过 ZIP 完整性、版本及大小检查；每包小于 250MB。
- 两个主包均包含 net472／net6.0／net8.0 程序集；原生 WeVisOne／builder DLL 与已验证源码产物的 SHA256 一致。
- 推理包包含 WeVisOne DLL 和四份第三方许可证／说明。
- 使用独立 NuGet 缓存创建仅含 PackageReference 的 .NET 8 消费项目，十个包全部从本地源还原，无 ProjectReference。
- 清除 SDK PATH 并禁止下载／SDK fallback 后，强类型 ONNX 转换、真实 sofa 检测及 matting 均通过。
- NuGet 抠图的 FP32 Alpha 哈希与源码验证结果一致；此前归一化数值回归见上述类型回归报告。
- 核对进程模块路径，相关原生 DLL 全部来自消费项目目录或其运行库拼接缓存。
- 版本解析三项通过：主包 0.2.1 搭配运行库 0.2.0；缺少 0.2.0 时允许较新运行库；安装脚本在多个版本中选择最新稳定运行库。后两项使用仅供还原测试的 0.2.9 元数据包，不代表该版本已发布或通过推理验证。

详细报告：[NuGet 0.2.1 验证 JSON](../package/TensorRT10/verification-0.2.1.json)。
此前的 11 task／33 个接口用例为源码产物验证；本节独立检查包的转换、检测与抠图功能。
