using alanxinn;
using alanxinn.TensorRt;

if (args.Length == 4 && args[0] == "--real-sam3")
{
    // Same real image and prompts as validate_sam3_real.py. Compare owned mask
    // buffers bit for bit across the Python C ABI and managed wrapper.
    var reference = Path.GetFullPath(args[3]);
    var pixels = File.ReadAllBytes(args[2]);
    using var sam = new Sam3Segmentor(args[1]);
    sam.SetImage(pixels, 810, 1080);
    sam.SetBox(50, 390, 245, 910);
    void Check(string stage, Sam3Mask mask, int index = 0)
    {
        if (mask.Width != 810 || mask.Height != 1080 ||
            !mask.Pixels.SequenceEqual(File.ReadAllBytes(Path.Combine(reference, $"{stage}-{index}.mask"))))
            throw new InvalidOperationException($"Real SAM3 C# mask differs: {stage}/{index}");
    }
    var initial = sam.Predict(); Check("box", initial);
    sam.AddPoint(150, 600, positive: true); Check("positive", sam.Predict());
    sam.AddPoint(240, 850, positive: false); Check("negative", sam.Predict());
    sam.ClearPoints(); Check("box", sam.Predict());
    using var tokens = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(reference, "text-tokens.json")));
    var root = tokens.RootElement;
    var ids = root.GetProperty("ids").EnumerateArray().Select(x => x.GetInt64()).ToArray();
    var attention = root.GetProperty("attention").EnumerateArray().Select(x => x.GetInt64()).ToArray();
    var text = sam.PredictText(root.GetProperty("text").GetString()!);
    var encoded = sam.PredictTokens(ids, attention);
    var expectedCount = Directory.GetFiles(reference, "text-*.mask").Length;
    if (text.Count != expectedCount || encoded.Count != expectedCount || expectedCount == 0)
        throw new InvalidOperationException("Real SAM3 C# text instance count differs");
    for (int i = 0; i < expectedCount; ++i) { Check("text", text[i], i); Check("text", encoded[i], i); }
    sam.Dispose(); Check("box", initial);
    Console.WriteLine($"PASS: REAL SAM3 C# box/positive/negative/clear/text/tokens/lifetime; text instances={text.Count}");
    return;
}

if (args.Length < 2 || args.Length > 4)
    throw new ArgumentException("usage: TensorRtIntegrationSmoke fixture-directory yolo.engine [tiny-sam3-engine-directory] [tiny-dinov3.engine]");
var directory = Path.GetFullPath(args[0]);
var onnx = Path.Combine(directory, "int64.onnx");
var engine = Path.Combine(directory, "csharp-int64.engine");
var options = new TensorRtBuildOptions { Precision = TensorRtPrecision.Fp32, EnableTf32 = false };
TensorRtRuntime.DisableAutoDownload = true;
TensorRtOnnxBuilder.Build(onnx, engine, options);
using (var model = new CvinferModel(args[1], CvinferModelType.Yolo11Detection,
           new CvinferConfig { EnableTiming = true }))
{
    var pixels = File.ReadAllBytes(Path.Combine(directory, "bus.bgr"));
    using var result = model.InferBgr(pixels, 810, 1080);
    if (result.Count < 1 || result.GetDetection(0).Box.Score <= .25f)
        throw new InvalidOperationException("C# inference returned no confident objects");
    Console.WriteLine($"C# inference passed: {result.Count} objects");
}
// Inference changes the Windows DLL search directory; subsequent builder
// calls still need to locate architecture-specific TensorRT resource DLLs.
TensorRtOnnxBuilder.Build(onnx, engine, options);
if (!File.Exists(engine) || new FileInfo(engine).Length == 0)
    throw new InvalidOperationException("C# builder returned an empty engine");
Console.WriteLine("PASS: builder -> inference -> builder in one .NET process");
if (args.Length >= 3)
{
    using var sam = new Sam3Segmentor(args[2], confidenceThreshold: 0, maxPoints: 8);
    sam.SetImage(new byte[48 * 32 * 3], 48, 32);
    sam.SetBox(4, 3, 42, 29);
    var box = sam.Predict();
    sam.AddPoint(20, 18, positive: true);
    sam.Predict();
    sam.AddPoint(6, 5, positive: false);
    sam.Predict();
    sam.ClearPoints();
    var restored = sam.Predict();
    if (box.Width != 48 || box.Height != 32 || !box.Pixels.SequenceEqual(restored.Pixels))
        throw new InvalidOperationException("C# SAM3 visual prompt state or mask copy failed");
    var text = sam.PredictText("BUS");
    var tokens = sam.PredictTokens(new long[] { 1, 7, 2, 0, 0, 0, 0, 0 }, new long[] { 1, 1, 1, 0, 0, 0, 0, 0 });
    if (text.Count != 2 || tokens.Count != 2 || !text[0].Pixels.SequenceEqual(tokens[0].Pixels))
        throw new InvalidOperationException("C# SAM3 text/token/mask contract failed");
    Console.WriteLine("PASS: SAM3 C# box/positive/negative/text/tokens (RANDOM tiny model; quality unvalidated)");
}
if (args.Length == 4)
{
    using var dino = new DinoV3FeatureExtractor(args[3]);
    var first = dino.InferBgr(new byte[48 * 32 * 3], 48, 32);
    var saved = first.GlobalEmbedding.ToArray();
    var second = dino.InferChw(new float[3 * dino.InputWidth * dino.InputHeight]);
    if (first.GlobalEmbedding.Length != 32 || first.PatchTokens.Shape[0] != 1 || first.PatchTokens.Shape[2] != 32 ||
        first.FeatureMap.Shape.Length != 4 || !saved.SequenceEqual(first.GlobalEmbedding))
        throw new InvalidOperationException("DINOv3 C# tensor copy/shape/lifetime failed");
    if (Math.Abs(DinoV3FeatureExtractor.CosineSimilarity(first.GlobalEmbedding, first.GlobalEmbedding) - 1) > 1e-6 ||
        Math.Abs(first.NormalizedEmbedding.Select(v => (double)v * v).Sum() - 1) > 1e-5)
        throw new InvalidOperationException("DINOv3 C# normalization/similarity failed");
    dino.Dispose();
    if (!saved.SequenceEqual(first.GlobalEmbedding)) throw new InvalidOperationException("Disposed model invalidated owned DINOv3 features");
    try { DinoV3FeatureExtractor.CosineSimilarity(new float[32], first.GlobalEmbedding); throw new Exception("Zero-vector cosine was accepted"); }
    catch (ArgumentException) { }
    try { dino.InferChw(new float[3 * dino.InputWidth * dino.InputHeight]); throw new Exception("Disposed DINOv3 was accepted"); }
    catch (ObjectDisposedException) { }
    Console.WriteLine("PASS: DINOv3 C# image/CHW/features/similarity/lifetime (RANDOM tiny model; quality unvalidated)");
}
