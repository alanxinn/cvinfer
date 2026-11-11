# WeVisOne-VL · TensorRT 10

参考 SAM2／SAM3／DINOv3 的模块布局，提供独立原生 DLL、C++、C ABI、C#、
Python 导出／构建／原生推理脚本和统一 C# Demo。

**当前状态：已编译并导出官方 2B 真实权重，自动验证记录见 [VALIDATION.md](VALIDATION.md)。**
本模块的验证与已有 SAM／DINOv3 的验证记录分开维护。

上游：[Tencent/WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL)。
默认模型为 `tencent/WeVisOne-VL-2B`，同样读取 4B 的模型配置。
导出所用官方源码默认固定于 `9a6bcd85404505df0d05cfb5d97512ccacf35dee`。

## 任务

| task | 输入 | 结构化结果 |
| --- | --- | --- |
| `detection` | 类别列表 | 类别及 XYXY 框 |
| `referring` | 自然语言描述 | 实例框、缺失标签／拒绝状态 |
| `point` | 类别／描述 | 原图点坐标 |
| `count_box` | 类别／描述 | 框、数量 |
| `count_point` | 类别／描述 | 点、数量 |
| `ocr` | 无 | 文字、框 |
| `ocr_grounding` | 指定文字 | 文字、框 |
| `segmentation` | 类别／描述 | 实例框、原图 0/255 二值掩码、浮点 Alpha |
| `matting` | 类别／描述 | 实例框、原图 FP32 连续 Alpha |
| `visual_prompt` | 一个或多个参考框 | 相似实例的框 |
| `freeform` | 自定义指令 | 原始回复及能识别的框／点／像素结果 |

另提供 `MatteBoxes`：使用框和 `<OBJ>` 构造 teacher-forced 回复，提取对应隐藏状态；
像素解码使用未量化的原图输入框，不依赖文本定位先找到目标。
参考框查找与指定框抠图是两个独立入口。

所有任务保留 `Response` 和 `Prompt`。`Count=-1` 表示不是计数任务或计数回复不完整，
`Rejected` 与 `AbsentLabels` 保留模型返回的缺失目标信息。
达到生成上限时 `Truncated=true`，保留已解析框／点，不返回不完整回复的像素结果。
模型没有检测置信度输出，本接口不虚构置信度。

## 导出与部署

使用独立 Python 环境安装 `scripts/requirements-wevisone.txt`。
导出需要较多 CPU 内存和磁盘空间，尤其语言模型和 ONNX external data。
以下命令用于准备和转换模型。

```powershell
pip install -r scripts/requirements-wevisone.txt
python scripts/prepare_wevisone.py
python scripts/export_wevisone.py --source cache/wevisone/source/9a6bcd85404505df0d05cfb5d97512ccacf35dee
python scripts/build_wevisone_engines.py
```

准备脚本记录 Hugging Face 的实际 commit；可用 `--revision <commit>` 固定权重版本。
也可向导出脚本传入 `--checkpoint <本地权重目录或官方模型 ID>`。
`--source` 为官方源码目录，也可预先安装官方包后省略该参数。

4B 使用独立目录：

```powershell
python scripts/prepare_wevisone.py --size 4B
python scripts/export_wevisone.py --checkpoint cache/wevisone/4B/checkpoint --output cache/wevisone/4B/onnx --source cache/wevisone/source/9a6bcd85404505df0d05cfb5d97512ccacf35dee
python scripts/build_wevisone_engines.py --onnx cache/wevisone/4B/onnx --output cache/wevisone/4B/engines
```

默认图像画布 512×512，**每个 bundle 的视觉网格固定**；原图通过双三次缩放到该画布，
不同宽高比可能发生拉伸，与官方默认保持宽高比的 smart-resize 有区别。
可按输入场景用 `--width`／`--height` 导出其他画布，尺寸须为 patch_size×merge_size 的整数倍，
当前 2B／4B 为 32 的倍数。像素解码器另使用官方配置的 1024×1024 RGB 画布，
不会复用 Qwen 的归一化数据。

导出默认 `--max-prefill 2048 --max-context 4096 --max-objects 16`；
构建默认将像素 profile 限制为 `min(4, 导出上限)`，可通过构建脚本的 `--max-objects` 调整，无需重新导出权重。
本机 RTX 4060 8GB 未能构建 16 实例像素 profile；4 实例 profile 使用 4GB workspace 构建成功。
提示长度包含图像 token；提示加生成上限不能超过上下文容量。
像素解码保留实例间自注意力，因此超出实例 profile 时明确报错，不能任意切成独立实例推理。
首版仅单图片 batch=1，不含视频或跨请求对话历史。

导出产物包含三张图：

| 图 | 内容 | 默认精度 |
| --- | --- | --- |
| `vision` | patch 编码、三层 DeepStack、四层 pre-merge 特征 | FP16，外部特征 FP32 |
| `language` | embedding、MRoPE、带 KV Cache 的预填充和逐 token 生成 | FP16，logits／末层隐藏状态 FP32 |
| `pixel` | 官方多尺度解码器及全分辨率 Alpha cascade | FP32 |

语言图共用一份 engine 权重。KV Cache 保留在 GPU，使用独立缓存缓冲区避免输出重分配导致输入指针失效。
像素解码器按需加载，`SetImage` 后多次预测复用视觉特征。
像素任务按官方流程对回复重新编码并执行 teacher-forcing，提取 `<OBJ>` 所在位置的末层、归一化前隐藏状态；
不会把预测 `<OBJ>` 的前一个 token 的隐藏状态当作目标特征。
保留完整 ONNX 外置权重文件，构建脚本在转换前检查所有文件的 SHA256。
`bundle.json` 最后写入，未完成的导出／构建不会留下可被误用的成功 manifest。

构建脚本复用现有 `tensorrt_builder` C ABI，关闭 TF32；`--dtype fp32` 可导出 FP32 backbone。
视觉和语言图使用强类型网络保留 ONNX 的 FP32 归一化，不能直接套用普通 FP16 降精度构建：
真实测试中普通 FP16 模式会导致语言归一化溢出、logits 全零并重复输出字符。
构建失败时保存不可加载的 `PARTIAL_ENGINES_BUILT` 清单；`--graphs vision language` 或 `--graphs pixel`
可以重建指定部分，其他 engine 必须通过输入文件哈希、engine 哈希及 profile 校验后才能复用。
FP32 像素解码遵循官方实现，未提供自动 INT8／INT4 量化。
显存和耗时取决于 engine、画布、实例数和上下文容量；当前实测范围见验证记录，4B 尚未实机测试。

原生运行目录需要 `bundle.json`、`tokenizer_native.json` 和三份 `.engine`。
推理不需要 Python、Transformers、vLLM 或 ONNX 文件；模型不随 NuGet 分发。

## C#

```csharp
using alanxinn;

using var model = new WeVisOneModel("cache/wevisone/2B/engines");
model.SetImage(bgrBytes, width, height); // 连续 BGR uint8
var detection = model.Detect(new[] { "person", "dog" });
var referring = model.Predict(WeVisOneTask.Referring, "the person on the left");
var count = model.Predict(WeVisOneTask.CountPoint, "person");
var ocr = model.Predict(WeVisOneTask.Ocr);
var grounding = model.Predict(WeVisOneTask.OcrGrounding, "OPEN");
var segmentation = model.Predict(WeVisOneTask.Segmentation, "person");
var matting = model.Predict(WeVisOneTask.Matting, "person");
var similar = model.Predict(WeVisOneTask.VisualPrompt,
    referenceBoxes: new[] { new WeVisOneBox(10, 20, 200, 300) });
var boxMatting = model.MatteBoxes(new[] { new WeVisOneBox(10, 20, 200, 300) });
var custom = model.Predict(WeVisOneTask.Freeform, "Describe the image.");
```

查询字符串和结果标签均为 UTF-8；原生 tokenizer 使用导出权重的 Qwen 字节 BPE、Unicode 正则和 NFC 归一化。
类别列表不按逗号拆分，指代表达中的逗号会保留。
`Predict` 可传入 `WeVisOneOptions` 配置生成长度、temperature、top-p／top-k、重复惩罚、随机种子及分割阈值。
`DecodePixels=false` 可只取得生成文本和几何结果。
所有结果数据独立持有，下一次调用或模型释放不影响已返回的数组。
每个并发调用者使用独立模型实例。

## Demo

```powershell
dotnet run --project csharp/alanxinn.CvinferDemo -c Release -- --type wevisone --model cache/wevisone/2B/engines --img image.jpg --vl-task detection --query "person`ndog"
dotnet run --project csharp/alanxinn.CvinferDemo -c Release -- --type wevisone --model cache/wevisone/2B/engines --img image.jpg --vl-task matting --query person --output wevisone_result
dotnet run --project csharp/alanxinn.CvinferDemo -c Release -- --type wevisone --model cache/wevisone/2B/engines --img image.jpg --box-matting --boxes "10,20,200,300;220,30,400,350"
python scripts/wevisone_infer.py --image image.jpg --task detection --query person --query dog
python scripts/wevisone_infer.py --image image.jpg --task visual_prompt --box 10 20 200 300
python scripts/wevisone_infer.py --image image.jpg --box-matting --box 10 20 200 300
```

C# Demo 使用 `--vl-task`，保留原有 `--task` 作为 `--type` 的别名。
输出目录含 `result.json`、框／点覆盖图、Alpha PNG、RGBA 抠图、分割 PNG，以及保留连续值的
小端 row-major FP32 `alpha_NNN.f32`。原始回复和缺失／截断标记存入 JSON。

C++ 示例为 `alanxinn_wevisone_demo <engine目录> <图片> <task> [query] [x1 y1 x2 y2]`。
C ABI 见 `include/wevisone/wevisone_api.h`，字符串查询长度包含 NUL，
Alpha 容量单位为 float 元素，二值掩码容量单位为字节。

## 构建集成

`BUILD_WEVISONE=ON` 默认加入 `alanxinn_cvinfer_wevisone.dll`；`BUILD_WEVISONE_DEMO` 控制 C++ 示例。
依赖 nlohmann/json 3.12.0 与 PCRE2 10.46，均静态链接，不增加额外运行 DLL。
首次 CMake 配置通过 FetchContent 获取源码；离线时指定
`FETCHCONTENT_SOURCE_DIR_WEVISONE_JSON` 和 `FETCHCONTENT_SOURCE_DIR_WEVISONE_PCRE2`。
第三方许可证位于 `licenses/`，随 C# NuGet 一同打包。
运行时查找模式与 DINOv3 一致，可使用 `ALANXINN_WEVISONE_NATIVE_DIR` 指定 DLL 目录。

NuGet 版本为 0.2.1，包含本模块接口和原生 DLL；模型及 engine 单独部署。旧 0.2.0 包不包含该模块。
主包版本与运行库版本独立，当前运行库仍为 0.2.0。使用 `scripts/Install-Cvinfer.ps1` 可在消费项目中按最新稳定环境包还原。
