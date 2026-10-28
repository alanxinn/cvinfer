using alanxinn;
using alanxinn.TensorRt;

if (args.Length != 2 && args.Length != 3)
    throw new ArgumentException("usage: TensorRtIntegrationSmoke fixture-directory yolo.engine [tiny-sam3-engine-directory]");
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
if (args.Length == 3)
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
