using OpenCvSharp;
using System.Runtime.InteropServices;
using alanxinn;

ParsedArguments parsed;
string modelTypeName;
string enginePath;
string imagePath;
string resultPath;
CvinferModelType modelType;
CvinferConfig config;
try
{
    parsed = ParseArguments(args);
    modelTypeName = GetRequiredOption(parsed.Options, "type");
    if (!modelTypeName.Equals("yolo-det", StringComparison.OrdinalIgnoreCase) &&
        !modelTypeName.Equals("yolo-seg", StringComparison.OrdinalIgnoreCase) &&
        !modelTypeName.Equals("rf-detr-det", StringComparison.OrdinalIgnoreCase) &&
        !modelTypeName.Equals("rf-detr-seg", StringComparison.OrdinalIgnoreCase))
        throw new ArgumentException("--type must be yolo-det, yolo-seg, rf-detr-det or rf-detr-seg");
    enginePath = GetRequiredOption(parsed.Options, "model");
    imagePath = GetRequiredOption(parsed.Options, "img");
    resultPath = GetOption(parsed.Options, "output", "cvinfer_result.jpg");
    modelType = modelTypeName.Equals("yolo-seg", StringComparison.OrdinalIgnoreCase)
        ? CvinferModelType.YoloSegmentation
        : (modelTypeName.Equals("rf-detr-det", StringComparison.OrdinalIgnoreCase)
            ? CvinferModelType.RfDetrDetection
            : (modelTypeName.Equals("rf-detr-seg", StringComparison.OrdinalIgnoreCase)
                ? CvinferModelType.RfDetrSegmentation
                : CvinferModelType.YoloDetection));
    config = new CvinferConfig
    {
        ClassCount = ParseIntOption(parsed.Options, "classes", 80),
        PreprocessBackend = ParseBackend(GetOption(parsed.Options, "preprocess", "auto")),
        PostprocessBackend = ParseBackend(GetOption(parsed.Options, "postprocess", "auto"))
    };
}
catch (Exception error)
{
    Console.Error.WriteLine(error.Message);
    return 2;
}

try
{
    using var input = Cv2.ImRead(imagePath, ImreadModes.Color);
    if (input.Empty()) throw new InvalidOperationException($"failed to read image: {imagePath}");
    if (input.Channels() != 3) throw new InvalidOperationException("input image must be a 3-channel BGR Mat");
    if (!input.IsContinuous()) throw new InvalidOperationException("input Mat must be continuous");

    using var model = new CvinferModel(enginePath, modelType, config);
    var warmup = ParseNonNegativeIntOption(parsed.Options, "warmup", 0);
    var iterations = ParsePositiveIntOption(parsed.Options, "iterations", 1);
    var benchmarkMode = parsed.Options.ContainsKey("warmup") || parsed.Options.ContainsKey("iterations");

    for (var i = 0; i < warmup; ++i)
    {
        using var warmupResult = model.InferBgr(input.Data, input.Cols, input.Rows, input.Channels());
    }

    var timings = new List<CvinferTiming>(iterations);
    CvinferResult? result = null;
    try
    {
        for (var i = 0; i < iterations; ++i)
        {
            var current = model.InferBgr(input.Data, input.Cols, input.Rows, input.Channels());
            timings.Add(current.Timing);
            result?.Dispose();
            result = current;
        }

        if (result is null) throw new InvalidOperationException("No inference result was produced");
        PrintTimingStats(timings, warmup);
        Console.WriteLine($"model={modelType} image={input.Cols}x{input.Rows} results={result.Count}");

        using var output = input.Clone();
        var boxesToDraw = new List<(CvinferBox Box, Scalar Color)>();
        for (var i = 0; i < result.Count; ++i)
        {
            var isSegmentation = modelType == CvinferModelType.YoloSegmentation ||
                modelType == CvinferModelType.RfDetrSegmentation;
            var segmentation = isSegmentation
                ? result.GetSegmentation(i)
                : default;
            var box = isSegmentation
                ? segmentation.Box
                : result.GetDetection(i).Box;
            var color = ColorForClass(box.ClassId);
            boxesToDraw.Add((box, color));

            if (isSegmentation)
            {
                var mask = result.CopyMask(i);
                OverlayMask(output, mask, segmentation.MaskWidth, segmentation.MaskHeight,
                    (int)MathF.Floor(box.X1), (int)MathF.Floor(box.Y1), color, 0.42f);
                var maskPath = Path.Combine(Path.GetDirectoryName(resultPath) ?? ".",
                    $"{Path.GetFileNameWithoutExtension(resultPath)}_mask_{i}.pgm");
                if (!benchmarkMode)
                {
                    SavePgm(maskPath, mask, segmentation.MaskWidth, segmentation.MaskHeight);
                    Console.WriteLine($"{i}: class={box.ClassId} score={box.Score:F4} box={box.X1:F1},{box.Y1:F1},{box.X2:F1},{box.Y2:F1} mask={segmentation.MaskWidth}x{segmentation.MaskHeight} file={maskPath}");
                }
                else
                {
                    Console.WriteLine($"{i}: class={box.ClassId} score={box.Score:F4} box={box.X1:F1},{box.Y1:F1},{box.X2:F1},{box.Y2:F1} mask={segmentation.MaskWidth}x{segmentation.MaskHeight}");
                }
            }
            else
            {
                Console.WriteLine($"{i}: class={box.ClassId} score={box.Score:F4} box={box.X1:F1},{box.Y1:F1},{box.X2:F1},{box.Y2:F1}");
            }
        }

        foreach (var item in boxesToDraw)
        {
            var rect = ToRect(item.Box, output.Cols, output.Rows);
            Cv2.Rectangle(output, rect, item.Color, 2);
            Cv2.PutText(output, $"{item.Box.ClassId}:{item.Box.Score:F3}",
                new Point(rect.X, Math.Max(15, rect.Y - 4)), HersheyFonts.HersheySimplex,
                0.5, item.Color, 1, LineTypes.AntiAlias);
        }

        if (!Cv2.ImWrite(resultPath, output)) throw new InvalidOperationException($"failed to save result image: {resultPath}");
        Console.WriteLine($"result image saved: {resultPath}");
    }
    finally
    {
        result?.Dispose();
    }
    return 0;
}
catch (Exception error)
{
    Console.Error.WriteLine(error.Message);
    return 1;
}

static void OverlayMask(Mat image, byte[] mask, int maskWidth, int maskHeight,
    int left, int top, Scalar color, double alpha)
{
    if (maskWidth <= 0 || maskHeight <= 0 || mask.Length != checked(maskWidth * maskHeight)) return;
    var right = Math.Min(image.Cols, left + maskWidth);
    var bottom = Math.Min(image.Rows, top + maskHeight);
    var roiLeft = Math.Max(0, left);
    var roiTop = Math.Max(0, top);
    if (right <= roiLeft || bottom <= roiTop) return;

    using var maskMat = new Mat(maskHeight, maskWidth, MatType.CV_8UC1);
    Marshal.Copy(mask, 0, maskMat.Data, mask.Length);
    var maskCropRect = new Rect(roiLeft - left, roiTop - top, right - roiLeft, bottom - roiTop);
    var imageRect = new Rect(roiLeft, roiTop, right - roiLeft, bottom - roiTop);
    using var maskCrop = new Mat(maskMat, maskCropRect);
    using var roi = new Mat(image, imageRect);
    using var colorMat = new Mat(imageRect.Height, imageRect.Width, MatType.CV_8UC3, color);
    using var blended = new Mat();
    var blend = Math.Clamp(alpha, 0.0, 1.0);
    Cv2.AddWeighted(roi, 1.0 - blend, colorMat, blend, 0.0, blended);
    blended.CopyTo(roi, maskCrop);
}

static Rect ToRect(CvinferBox box, int imageWidth, int imageHeight)
{
    var left = Math.Clamp((int)MathF.Floor(box.X1), 0, Math.Max(0, imageWidth - 1));
    var top = Math.Clamp((int)MathF.Floor(box.Y1), 0, Math.Max(0, imageHeight - 1));
    var right = Math.Clamp((int)MathF.Ceiling(box.X2), left + 1, imageWidth);
    var bottom = Math.Clamp((int)MathF.Ceiling(box.Y2), top + 1, imageHeight);
    return new Rect(left, top, Math.Max(1, right - left), Math.Max(1, bottom - top));
}

static Scalar ColorForClass(int classId)
{
    var hue = ((classId * 47) % 360 + 360) % 360;
    return ColorFromHsv(hue, 0.85f, 0.95f);
}

static Scalar ColorFromHsv(float hue, float saturation, float value)
{
    var h = hue / 60f;
    var c = value * saturation;
    var x = c * (1f - Math.Abs(h % 2f - 1f));
    var m = value - c;
    var (r, g, b) = h switch
    {
        < 1f => (c, x, 0f),
        < 2f => (x, c, 0f),
        < 3f => (0f, c, x),
        < 4f => (0f, x, c),
        < 5f => (x, 0f, c),
        _ => (c, 0f, x)
    };
    return new Scalar((b + m) * 255f, (g + m) * 255f, (r + m) * 255f);
}

static void SavePgm(string path, byte[] data, int width, int height)
{
    if (data.Length != checked(width * height)) throw new InvalidDataException("Mask size does not match mask dimensions");
    using var stream = File.Create(path);
    using var writer = new StreamWriter(stream, System.Text.Encoding.ASCII, leaveOpen: true);
    writer.Write($"P5\n{width} {height}\n255\n");
    writer.Flush();
    stream.Write(data, 0, data.Length);
}

static ParsedArguments ParseArguments(string[] args)
{
    var parsed = new ParsedArguments();
    for (var i = 0; i < args.Length; ++i)
    {
        var token = args[i];
        if (!token.StartsWith("--", StringComparison.Ordinal) || token == "--")
            throw new ArgumentException($"only named options are supported: {token}");
        var key = token[2..].Trim().ToLowerInvariant() switch
        {
            "engine" => "model",
            "image" => "img",
            "result" => "output",
            "preprocess-backend" => "preprocess",
            "postprocess-backend" => "postprocess",
            "task" => "type",
            var value => value
        };
        if (key is "input-width" or "input-height")
            throw new ArgumentException($"{token} is no longer supported; input size is read from the TensorRT engine");
        if (key is not ("type" or "model" or "img" or "output" or "classes" or
            "preprocess" or "postprocess" or "warmup" or "iterations"))
            throw new ArgumentException($"unknown option: {token}");
        if (string.IsNullOrWhiteSpace(key) || i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            throw new ArgumentException($"option requires a value: {token}");
        parsed.Options[key] = args[++i];
    }
    return parsed;
}

static string GetRequiredOption(Dictionary<string, string> options, string key)
{
    if (!options.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
        throw new ArgumentException($"missing required option: --{key}");
    return value;
}

static string GetOption(Dictionary<string, string> options, string key, string fallback)
    => options.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value) ? value : fallback;

static int ParseIntOption(Dictionary<string, string> options, string key, int fallback)
    => options.TryGetValue(key, out var value) ? int.Parse(value) : fallback;

static CvinferComputeBackend ParseBackend(string value)
{
    return value.Trim().ToLowerInvariant() switch
    {
        "auto" => CvinferComputeBackend.Auto,
        "cpu" => CvinferComputeBackend.Cpu,
        "cuda" or "gpu" => CvinferComputeBackend.Cuda,
        _ => throw new ArgumentException($"backend must be auto, cpu or cuda: {value}")
    };
}

static int ParseNonNegativeIntOption(Dictionary<string, string> options, string key, int fallback)
{
    var value = ParseIntOption(options, key, fallback);
    if (value < 0) throw new ArgumentOutOfRangeException(key, "value must be non-negative");
    return value;
}

static int ParsePositiveIntOption(Dictionary<string, string> options, string key, int fallback)
{
    var value = ParseIntOption(options, key, fallback);
    if (value <= 0) throw new ArgumentOutOfRangeException(key, "value must be positive");
    return value;
}

static void PrintTimingStats(IReadOnlyList<CvinferTiming> samples, int warmup)
{
    if (samples.Count == 0) return;
    var names = new[] { "preprocess", "inference", "postprocess", "total" };
    var values = new Func<CvinferTiming, double>[]
    {
        static t => t.PreprocessMs,
        static t => t.InferenceMs,
        static t => t.PostprocessMs,
        static t => t.TotalMs
    };
    Console.WriteLine($"benchmark warmup={warmup} iterations={samples.Count}");
    for (var i = 0; i < values.Length; ++i)
    {
        var sum = 0.0;
        var min = double.PositiveInfinity;
        var max = double.NegativeInfinity;
        foreach (var sample in samples)
        {
            var value = values[i](sample);
            sum += value;
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }
        Console.WriteLine($"{names[i]} avg={sum / samples.Count:F4} min={min:F4} max={max:F4} ms");
    }
}

sealed class ParsedArguments
{
    public Dictionary<string, string> Options { get; } = new(StringComparer.OrdinalIgnoreCase);
}
