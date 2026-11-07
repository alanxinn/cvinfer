using System.Runtime.InteropServices;
using System.Text.Json;
using OpenCvSharp;
using alanxinn;
using alanxinn.TensorRt;

// Integration checks share the same executable and deployment as the image demos.
internal static class TensorRtValidation
{
    internal static int Run(Dictionary<string, string> options)
    {
        string test, model, directory, image;
        try
        {
            test = Required(options, "test").Trim().ToLowerInvariant();
            if (test is not ("integration" or "sam3-real" or "nuget"))
                throw new ArgumentException("--test must be integration, sam3-real or nuget");
            model = test == "nuget" ? "" : Required(options, "model");
            directory = Required(options, test == "integration" ? "fixtures" : "reference");
            image = test == "sam3-real" ? Required(options, "img") : "";
        }
        catch (Exception error) { Console.Error.WriteLine(error.Message); return 2; }

        try
        {
            var checks = new List<string>();
            if (test == "nuget") { NugetValidation.Run(directory); return 0; }
            if (test == "integration") Integration(directory, model, options, checks);
            else RealSam3(model, image, directory, checks);
            var report = new { status = "PASS", test, runtime = Environment.Version.ToString(), checks };
            if (options.TryGetValue("json", out var jsonPath))
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(jsonPath))!);
                File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
            }
            Console.WriteLine(JsonSerializer.Serialize(report));
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error.Message); return 1; }
    }

    private static string Required(Dictionary<string, string> options, string key) =>
        options.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value)
            ? value : throw new ArgumentException($"missing required option: --{key}");
    private static void Check(bool condition, string message)
    {
        if (!condition) throw new InvalidOperationException(message);
    }
    private static void Passed(List<string> checks, string name)
    {
        checks.Add(name);
        Console.WriteLine($"PASS: {name}");
    }

    private static void Integration(string directory, string yoloEngine,
        Dictionary<string, string> options, List<string> checks)
    {
        directory = Path.GetFullPath(directory);
        var onnx = Path.Combine(directory, "int64.onnx");
        var engine = Path.Combine(directory, "csharp-int64.engine");
        var buildOptions = new TensorRtBuildOptions { Precision = TensorRtPrecision.Fp32, EnableTf32 = false };
        TensorRtRuntime.DisableAutoDownload = true;
        TensorRtOnnxBuilder.Build(onnx, engine, buildOptions);
        using (var model = new CvinferModel(yoloEngine, CvinferModelType.Yolo11Detection,
                   new CvinferConfig { EnableTiming = true }))
        {
            var pixels = File.ReadAllBytes(Path.Combine(directory, "bus.bgr"));
            Check(pixels.Length == 810 * 1080 * 3, "Integration fixture bus.bgr must be 810x1080 BGR");
            using var result = model.InferBgr(pixels, 810, 1080);
            Check(result.Count > 0 && result.GetDetection(0).Box.Score > .25f, "YOLO returned no confident objects");
            Passed(checks, $"YOLO real-image inference: {result.Count} objects");
        }
        // Inference changes Windows DLL search state. Building again in the
        // same process verifies that TensorRT builder resources remain visible.
        TensorRtOnnxBuilder.Build(onnx, engine, buildOptions);
        Check(File.Exists(engine) && new FileInfo(engine).Length > 0, "Builder returned an empty engine");
        Passed(checks, "builder -> inference -> builder in one process (INT64 ONNX)");

        if (options.TryGetValue("sam3-model", out var samEngine))
        {
            using var sam = new Sam3Segmentor(samEngine, confidenceThreshold: 0, maxPoints: 8);
            sam.SetImage(new byte[48 * 32 * 3], 48, 32);
            sam.SetBox(4, 3, 42, 29);
            var box = sam.Predict();
            sam.AddPoint(20, 18, positive: true); sam.Predict();
            sam.AddPoint(6, 5, positive: false); sam.Predict();
            sam.ClearPoints();
            var restored = sam.Predict();
            Check(box.Width == 48 && box.Height == 32 && box.Pixels.SequenceEqual(restored.Pixels),
                  "SAM3 visual prompt state or mask ownership failed");
            var text = sam.PredictText("BUS");
            var tokens = sam.PredictTokens(new long[] { 1, 7, 2, 0, 0, 0, 0, 0 }, new long[] { 1, 1, 1, 0, 0, 0, 0, 0 });
            Check(text.Count == 2 && tokens.Count == 2 && text[0].Pixels.SequenceEqual(tokens[0].Pixels),
                  "SAM3 text/token/mask contract failed");
            sam.Dispose();
            Check(box.Pixels.SequenceEqual(restored.Pixels), "Disposed SAM3 invalidated owned masks");
            Passed(checks, "SAM3 box/positive/negative/clear/text/tokens/lifetime (RANDOM tiny fixture; quality unvalidated)");
        }
        if (options.TryGetValue("dinov3-model", out var dinoEngine))
        {
            using var dino = new DinoV3FeatureExtractor(dinoEngine);
            var first = dino.InferBgr(new byte[48 * 32 * 3], 48, 32);
            var saved = first.GlobalEmbedding.ToArray();
            dino.InferChw(new float[3 * dino.InputWidth * dino.InputHeight]);
            Check(first.GlobalEmbedding.Length == 32 && first.PatchTokens.Shape[0] == 1 && first.PatchTokens.Shape[2] == 32 &&
                  first.FeatureMap.Shape.Length == 4 && saved.SequenceEqual(first.GlobalEmbedding),
                  "DINOv3 feature copy/shape/lifetime failed");
            Check(Math.Abs(DinoV3FeatureExtractor.CosineSimilarity(first.GlobalEmbedding, first.GlobalEmbedding) - 1) <= 1e-6 &&
                  Math.Abs(first.NormalizedEmbedding.Select(v => (double)v * v).Sum() - 1) <= 1e-5,
                  "DINOv3 normalization/similarity failed");
            var elements = 3 * dino.InputWidth * dino.InputHeight;
            dino.Dispose();
            Check(saved.SequenceEqual(first.GlobalEmbedding), "Disposed model invalidated owned DINOv3 features");
            try { DinoV3FeatureExtractor.CosineSimilarity(new float[32], first.GlobalEmbedding); throw new InvalidOperationException("Zero-vector cosine was accepted"); }
            catch (ArgumentException) { }
            try { dino.InferChw(new float[elements]); throw new InvalidOperationException("Disposed DINOv3 was accepted"); }
            catch (ObjectDisposedException) { }
            Passed(checks, "DINOv3 image/CHW/features/similarity/lifetime (RANDOM tiny fixture; quality unvalidated)");
        }
    }

    private static void RealSam3(string engine, string imagePath, string reference, List<string> checks)
    {
        using var image = Cv2.ImRead(imagePath, ImreadModes.Color);
        Check(!image.Empty() && image.IsContinuous(), $"Cannot read contiguous BGR image: {imagePath}");
        var pixels = new byte[checked(image.Cols * image.Rows * 3)];
        Marshal.Copy(image.Data, pixels, 0, pixels.Length);
        using var sam = new Sam3Segmentor(engine);
        sam.SetImage(pixels, image.Cols, image.Rows);
        sam.SetBox(50, 390, 245, 910);
        void Compare(string stage, Sam3Mask mask, int index = 0)
        {
            Check(mask.Width == image.Cols && mask.Height == image.Rows &&
                  mask.Pixels.SequenceEqual(File.ReadAllBytes(Path.Combine(reference, $"{stage}-{index}.mask"))),
                  $"Real SAM3 mask differs from native reference: {stage}/{index}");
        }
        var initial = sam.Predict(); Compare("box", initial);
        sam.AddPoint(150, 600, positive: true); Compare("positive", sam.Predict());
        sam.AddPoint(240, 850, positive: false); Compare("negative", sam.Predict());
        sam.ClearPoints(); Compare("box", sam.Predict());
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(reference, "text-tokens.json")));
        var root = document.RootElement;
        var ids = root.GetProperty("ids").EnumerateArray().Select(x => x.GetInt64()).ToArray();
        var attention = root.GetProperty("attention").EnumerateArray().Select(x => x.GetInt64()).ToArray();
        var text = sam.PredictText(root.GetProperty("text").GetString()!);
        var encoded = sam.PredictTokens(ids, attention);
        var expectedCount = Directory.GetFiles(reference, "text-*.mask").Length;
        Check(text.Count == expectedCount && encoded.Count == expectedCount && expectedCount > 0, "Real SAM3 text instance count differs");
        for (var i = 0; i < expectedCount; ++i) { Compare("text", text[i], i); Compare("text", encoded[i], i); }
        sam.Dispose(); Compare("box", initial);
        Passed(checks, $"REAL SAM3 box/positive/negative/clear/text/tokens/lifetime; text instances={text.Count}");
    }
}
