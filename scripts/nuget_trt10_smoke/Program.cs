using System.Diagnostics;
using System.Text.Json;
using alanxinn;
using alanxinn.TensorRt;

// Compiled by test_trt10_nuget.py with PackageReference only. Each process
// loads the native assets restored from the new NuGet packages.
var input = JsonDocument.Parse(File.ReadAllText(args[0])).RootElement;
string Get(string key) => input.GetProperty(key).GetString()!;
var pixels = File.ReadAllBytes(Get("image"));
var width = input.GetProperty("width").GetInt32();
var height = input.GetProperty("height").GetInt32();
var mode = Get("mode");
var watch = Stopwatch.StartNew();
var report = new Dictionary<string, object?> { ["mode"] = mode, ["name"] = Get("name") };
void Require(bool ok, string message) { if (!ok) throw new InvalidOperationException(message); }
void Build(string source, string destination, bool half)
{
    TensorRtRuntime.DisableAutoDownload = true;
    TensorRtOnnxBuilder.Build(source, destination, new TensorRtBuildOptions {
        Precision = half ? TensorRtPrecision.Fp16 : TensorRtPrecision.Fp32,
        EnableTf32 = false, WorkspaceMiB = 2048 });
    Require(File.Exists(destination) && new FileInfo(destination).Length > 0, "Builder returned an empty engine");
    Require(Path.GetFullPath(TensorRtRuntime.InitializedDirectory!) == Path.GetFullPath(Get("runtime")),
            "Builder did not use the extracted runtime ZIP");
}
if (mode == "model")
{
    var engine = Get("engine");
    var rebuild = input.GetProperty("rebuild").GetBoolean();
    if (rebuild || !File.Exists(engine)) Build(Get("onnx"), engine, Get("precision") == "fp16");
    var type = (CvinferModelType)input.GetProperty("type").GetInt32();
    int count = 0, masks = 0, keypoints = 0;
    using (var model = new CvinferModel(engine, type, new CvinferConfig { EnableTiming = true }))
    {
        for (int iteration = 0; iteration < 3; ++iteration)
        {
            using var result = model.InferBgr(pixels, width, height);
            Require(result.Count > 0, "No objects detected in the test image");
            if (iteration > 0) Require(result.Count == count, "Repeated inference changed object count");
            count = result.Count; masks = 0; keypoints = 0;
            for (int i = 0; i < count; ++i)
            {
                CvinferBox box;
                if (type is CvinferModelType.Yolo11Segmentation or CvinferModelType.Yolo26Segmentation or CvinferModelType.RfDetrSegmentation)
                {
                    var item = result.GetSegmentation(i); box = item.Box;
                    var mask = result.CopyMask(i);
                    Require(mask.Length == item.MaskWidth * item.MaskHeight && mask.Any(x => x > 0), "Invalid or empty segmentation mask");
                    ++masks;
                }
                else if (type is CvinferModelType.Yolo11Pose or CvinferModelType.Yolo26Pose)
                {
                    var item = result.GetPose(i); box = item.Box;
                    Require(item.Keypoints.Length == 17, "Expected 17 pose keypoints");
                    Require(item.Keypoints.All(x => float.IsFinite(x.X) && float.IsFinite(x.Y) && float.IsFinite(x.Score)), "Non-finite pose output");
                    keypoints += item.Keypoints.Count(x => x.Score > .25f);
                }
                else box = result.GetDetection(i).Box;
                Require(float.IsFinite(box.Score) && box.Score >= .25f && box.X2 > box.X1 && box.Y2 > box.Y1, "Invalid detection box");
            }
            if (iteration == 2) report["inference_ms"] = result.Timing.InferenceMs;
        }
    }
    if (rebuild)
    {
        // Inference changes SetDllDirectory; exercise builder again in the same
        // process to verify resource-DLL discovery from the packaged environment.
        Build(Get("contract_onnx"), Get("contract_engine"), false);
        report["builder_inference_builder"] = "PASS";
    }
    report["objects"] = count; report["nonempty_masks"] = masks; report["valid_keypoints"] = keypoints;
}
else if (mode == "sam3")
{
    using var model = new Sam3Segmentor(Get("engine"));
    model.SetImage(pixels, width, height); model.SetBox(50, 390, 245, 910);
    var initial = model.Predict();
    Require(initial.Pixels.Any(x => x > 0), "Empty box mask");
    model.AddPoint(150, 600, true); var positive = model.Predict();
    model.AddPoint(240, 850, false); var negative = model.Predict();
    Require(positive.Pixels.Any(x => x > 0) && negative.Pixels.Any(x => x > 0), "Empty refined mask");
    model.ClearPoints(); Require(initial.Pixels.SequenceEqual(model.Predict().Pixels), "Clearing points changed the original box result");
    var people = model.PredictText("person"); var phrase = model.PredictText("person wearing a white jacket");
    Require(people.Count > 0 && phrase.Count > 0, "No instances for known real image text prompts");
    Require(people.All(x => x.Width == width && x.Height == height && x.Pixels.Any(v => v > 0)), "Invalid text masks");
    model.Dispose(); Require(initial.Pixels.Any(x => x > 0), "Model disposal invalidated the owned mask");
    report["person_instances"] = people.Count; report["phrase_instances"] = phrase.Count;
}
else if (mode == "dinov3")
{
    using var model = new DinoV3FeatureExtractor(Get("engine"));
    var first = model.InferBgr(pixels, width, height);
    var second = model.InferBgr(pixels, width, height);
    Require(first.GlobalEmbedding.All(float.IsFinite) && first.GlobalEmbedding.Length > 0, "Invalid global features");
    Require(Math.Abs(DinoV3FeatureExtractor.CosineSimilarity(first.GlobalEmbedding, second.GlobalEmbedding) - 1) < 1e-5, "Repeated image features differ");
    Require(first.PatchTokens.Shape[1] == first.FeatureMap.Shape[2] * first.FeatureMap.Shape[3], "Inconsistent spatial feature shapes");
    report["embedding_size"] = first.GlobalEmbedding.Length;
    report["synthetic"] = input.GetProperty("synthetic").GetBoolean();
}
else throw new ArgumentException("Unknown smoke mode: " + mode);

var outputRoot = Path.GetFullPath(AppContext.BaseDirectory);
var runtimeRoot = Path.GetFullPath(Get("runtime"));
var modules = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(x => x.ModuleName.StartsWith("alanxinn_", StringComparison.OrdinalIgnoreCase) ||
                x.ModuleName.StartsWith("nvinfer", StringComparison.OrdinalIgnoreCase) ||
                x.ModuleName.StartsWith("nvonnxparser", StringComparison.OrdinalIgnoreCase) ||
                x.ModuleName.StartsWith("cudnn", StringComparison.OrdinalIgnoreCase) ||
                x.ModuleName.StartsWith("cublas", StringComparison.OrdinalIgnoreCase) ||
                x.ModuleName.StartsWith("nvrtc", StringComparison.OrdinalIgnoreCase))
    .Select(x => x.FileName).ToArray();
Require(modules.Any(x => Path.GetFileName(x).StartsWith("alanxinn_")), "No package native module loaded");
foreach (var module in modules)
{
    Require(module.StartsWith(outputRoot, StringComparison.OrdinalIgnoreCase) ||
            module.StartsWith(runtimeRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase),
            "Native module was loaded outside package output / runtime ZIP: " + module);
    var name = Path.GetFileName(module);
    if (!input.GetProperty("optional_libraries").GetBoolean())
        Require(!name.StartsWith("cudnn", StringComparison.OrdinalIgnoreCase) &&
                !name.StartsWith("cublas", StringComparison.OrdinalIgnoreCase) &&
                !name.StartsWith("nvrtc", StringComparison.OrdinalIgnoreCase),
                "A minimal-runtime test unexpectedly loaded an optional library: " + module);
}
report["native_modules"] = modules;
report["status"] = "PASS"; report["seconds"] = watch.Elapsed.TotalSeconds;
File.WriteAllText(Get("report"), JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine(JsonSerializer.Serialize(report));
