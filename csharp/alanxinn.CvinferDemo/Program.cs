using System.Drawing;
using System.Drawing.Imaging;
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
    if (!modelTypeName.Equals("det", StringComparison.OrdinalIgnoreCase) &&
        !modelTypeName.Equals("seg", StringComparison.OrdinalIgnoreCase))
        throw new ArgumentException("--type must be det or seg");
    enginePath = GetRequiredOption(parsed.Options, "model");
    imagePath = GetRequiredOption(parsed.Options, "img");
    resultPath = GetOption(parsed.Options, "output", "cvinfer_result.jpg");
    modelType = modelTypeName.Equals("seg", StringComparison.OrdinalIgnoreCase)
        ? CvinferModelType.YoloSegmentation
        : CvinferModelType.YoloDetection;
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
    var (bgr, width, height, channels) = LoadBgr(imagePath);
    using var model = new CvinferModel(enginePath, modelType, config);
    var warmup = ParseNonNegativeIntOption(parsed.Options, "warmup", 0);
    var iterations = ParsePositiveIntOption(parsed.Options, "iterations", 1);
    var benchmarkMode = parsed.Options.ContainsKey("warmup") || parsed.Options.ContainsKey("iterations");

    for (var i = 0; i < warmup; ++i)
    {
        using var warmupResult = model.InferBgr(bgr, width, height, channels);
    }

    var timings = new List<CvinferTiming>(iterations);
    CvinferResult? result = null;
    try
    {
        for (var i = 0; i < iterations; ++i)
        {
            var current = model.InferBgr(bgr, width, height, channels);
            timings.Add(current.Timing);
            result?.Dispose();
            result = current;
        }

        if (result is null) throw new InvalidOperationException("No inference result was produced");
        PrintTimingStats(timings, warmup);
        Console.WriteLine($"model={modelType} image={width}x{height} results={result.Count}");

        using var source = new Bitmap(imagePath);
        using var output = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
        using (var copyGraphics = Graphics.FromImage(output)) copyGraphics.DrawImage(source, 0, 0, source.Width, source.Height);
        var boxesToDraw = new List<(CvinferBox Box, Color Color)>();
        for (var i = 0; i < result.Count; ++i)
        {
            var segmentation = modelType == CvinferModelType.YoloSegmentation
                ? result.GetSegmentation(i)
                : default;
            var box = modelType == CvinferModelType.YoloSegmentation
                ? segmentation.Box
                : result.GetDetection(i).Box;
            var color = ColorForClass(box.ClassId);
            boxesToDraw.Add((box, color));

            if (modelType == CvinferModelType.YoloSegmentation)
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

        using (var graphics = Graphics.FromImage(output))
        using (var font = new Font(FontFamily.GenericSansSerif, 10))
        {
            foreach (var item in boxesToDraw)
            {
                using var pen = new Pen(item.Color, 2);
                using var brush = new SolidBrush(item.Color);
                var rect = RectangleF.FromLTRB(item.Box.X1, item.Box.Y1, item.Box.X2, item.Box.Y2);
                graphics.DrawRectangle(pen, rect.X, rect.Y, rect.Width, rect.Height);
                graphics.DrawString($"{item.Box.ClassId}:{item.Box.Score:F3}", font, brush, rect.X, Math.Max(0, rect.Y - 14));
            }
        }
        var extension = Path.GetExtension(resultPath);
        var format = extension.Equals(".png", StringComparison.OrdinalIgnoreCase) ? ImageFormat.Png : ImageFormat.Jpeg;
        output.Save(resultPath, format);
        Console.WriteLine($"result image saved: {resultPath}");
    }
    finally
    {
        result?.Dispose();
    }
    return 0;
}catch (Exception error)
{
    Console.Error.WriteLine(error.Message);
    return 1;
}

static (byte[] Bgr, int Width, int Height, int Channels) LoadBgr(string path)
{
    using var source = new Bitmap(path);
    using var bitmap = new Bitmap(source.Width, source.Height, PixelFormat.Format24bppRgb);
    using (var graphics = Graphics.FromImage(bitmap)) graphics.DrawImage(source, 0, 0, source.Width, source.Height);
    var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
    var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
    try
    {
        var rowBytes = checked(bitmap.Width * 3);
        var bgr = new byte[checked(rowBytes * bitmap.Height)];
        var stride = data.Stride;
        for (var y = 0; y < bitmap.Height; ++y)
        {
            var sourceOffset = stride >= 0 ? y * stride : (bitmap.Height - 1 - y) * -stride;
            Marshal.Copy(IntPtr.Add(data.Scan0, sourceOffset), bgr, y * rowBytes, rowBytes);
        }
        return (bgr, bitmap.Width, bitmap.Height, 3);
    }
    finally
    {
        bitmap.UnlockBits(data);
    }
}

static void OverlayMask(Bitmap image, byte[] mask, int maskWidth, int maskHeight, int left, int top, Color color, float alpha)
{
    if (mask.Length != checked(maskWidth * maskHeight)) return;
    var rect = new Rectangle(0, 0, image.Width, image.Height);
    var data = image.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);
    try
    {
        var blend = Math.Clamp(alpha, 0f, 1f);
        for (var y = 0; y < maskHeight; ++y)
        {
            var imageY = top + y;
            if ((uint)imageY >= (uint)image.Height) continue;
            for (var x = 0; x < maskWidth; ++x)
            {
                if (mask[y * maskWidth + x] == 0) continue;
                var imageX = left + x;
                if ((uint)imageX >= (uint)image.Width) continue;
                var offset = imageY * data.Stride + imageX * 3;
                var pixel = IntPtr.Add(data.Scan0, offset);
                var blue = Marshal.ReadByte(pixel, 0);
                var green = Marshal.ReadByte(pixel, 1);
                var red = Marshal.ReadByte(pixel, 2);
                Marshal.WriteByte(pixel, 0, (byte)(blue * (1f - blend) + color.B * blend));
                Marshal.WriteByte(pixel, 1, (byte)(green * (1f - blend) + color.G * blend));
                Marshal.WriteByte(pixel, 2, (byte)(red * (1f - blend) + color.R * blend));
            }
        }
    }
    finally
    {
        image.UnlockBits(data);
    }
}

static Color ColorForClass(int classId)
{
    var hue = ((classId * 47) % 360 + 360) % 360;
    return ColorFromHsv(hue, 0.85f, 0.95f);
}

static Color ColorFromHsv(float hue, float saturation, float value)
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
    return Color.FromArgb((int)((r + m) * 255f), (int)((g + m) * 255f), (int)((b + m) * 255f));
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
