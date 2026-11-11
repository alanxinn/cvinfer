using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using OpenCvSharp;
using alanxinn;

internal static class WeVisOneDemo
{
    public static int Run(Dictionary<string, string> args)
    {
        try
        {
            string Option(string name, string fallback = "") => args.TryGetValue(name, out var value) ? value : fallback;
            string Required(string name) => args.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value)
                ? value : throw new ArgumentException("Missing --" + name);
            float Float(string name, float fallback) => args.TryGetValue(name, out var value)
                ? float.Parse(value, CultureInfo.InvariantCulture) : fallback;
            var task = Option("vl-task", "detection") switch
            {
                "detection" => WeVisOneTask.Detection, "referring" => WeVisOneTask.Referring,
                "point" => WeVisOneTask.Point, "count_box" => WeVisOneTask.CountBox, "count_point" => WeVisOneTask.CountPoint,
                "ocr" => WeVisOneTask.Ocr, "ocr_grounding" => WeVisOneTask.OcrGrounding,
                "segmentation" => WeVisOneTask.Segmentation, "matting" => WeVisOneTask.Matting,
                "visual_prompt" => WeVisOneTask.VisualPrompt, "freeform" => WeVisOneTask.Freeform,
                _ => throw new ArgumentException("Unknown --vl-task")
            };
            var options = new WeVisOneOptions { MaxNewTokens = int.Parse(Option("max-new-tokens", "1024"), CultureInfo.InvariantCulture),
                Temperature = Float("temperature", 0), TopP = Float("top-p", .9f),
                TopK = int.Parse(Option("top-k", "50"), CultureInfo.InvariantCulture),
                RepetitionPenalty = Float("repetition-penalty", 1), MaskThreshold = Float("mask-threshold", .5f),
                Seed = ulong.Parse(Option("seed", "0"), CultureInfo.InvariantCulture), DecodePixels = !args.ContainsKey("no-pixels") };
            var boxes = new List<WeVisOneBox>();
            foreach (var value in Option("boxes").Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                var parts = value.Split(',').Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                if (parts.Length != 4) throw new ArgumentException("--boxes format: x1,y1,x2,y2;x1,y1,x2,y2");
                boxes.Add(new WeVisOneBox(parts[0], parts[1], parts[2], parts[3]));
            }
            using var image = Cv2.ImRead(Required("img"), ImreadModes.Color);
            if (image.Empty() || !image.IsContinuous()) throw new InvalidDataException("Cannot read contiguous BGR image");
            using var model = new WeVisOneModel(Required("model"));
            model.SetImage(image.Data, image.Cols, image.Rows);
            WeVisOneResult Infer() => args.ContainsKey("box-matting") ? model.MatteBoxes(boxes.ToArray(), options)
                : model.Predict(task, Option("query"), options, boxes.ToArray(), Option("prompt"));
            var warmup = int.Parse(Option("warmup", "0"), CultureInfo.InvariantCulture);
            var iterations = int.Parse(Option("iterations", "1"), CultureInfo.InvariantCulture);
            if (warmup < 0 || iterations <= 0) throw new ArgumentException("warmup must be nonnegative and iterations positive");
            for (var i = 0; i < warmup; ++i) Infer();
            WeVisOneResult? result = null;
            var samples = new List<double>(iterations);
            for (var i = 0; i < iterations; ++i) { result = Infer(); samples.Add(result.LatencyMilliseconds); }
            if (result is null) throw new InvalidOperationException("No WeVisOne result was produced");
            var directory = Path.GetFullPath(Option("output", "wevisone_result"));
            Directory.CreateDirectory(directory);
            using var overlay = image.Clone();
            foreach (var a in result.Annotations)
            {
                if (a.IsPoint) Cv2.Circle(overlay, new Point((int)a.X, (int)a.Y), 4, new Scalar(0, 255, 0), -1);
                else Cv2.Rectangle(overlay, new Point((int)a.Box.X1, (int)a.Box.Y1), new Point((int)a.Box.X2, (int)a.Box.Y2), new Scalar(0, 255, 0), 2);
            }
            for (var i = 0; i < result.Masks.Count; ++i)
            {
                var m = result.Masks[i];
                var alpha = m.Alpha.Select(x => (byte)Math.Clamp((int)Math.Round(x * 255), 0, 255)).ToArray();
                using var a = new Mat(m.Height, m.Width, MatType.CV_8UC1);
                Marshal.Copy(alpha, 0, a.Data, alpha.Length);
                Cv2.ImWrite(Path.Combine(directory, $"alpha_{i:000}.png"), a);
                using var rgba = new Mat(); Cv2.CvtColor(image, rgba, ColorConversionCodes.BGR2BGRA);
                Cv2.InsertChannel(a, rgba, 3); Cv2.ImWrite(Path.Combine(directory, $"cutout_{i:000}.png"), rgba);
                using (var writer = new BinaryWriter(File.Create(Path.Combine(directory, $"alpha_{i:000}.f32"))))
                    foreach (var value in m.Alpha) writer.Write(value);
                if (m.Binary.Length > 0)
                {
                    Marshal.Copy(m.Binary, 0, a.Data, m.Binary.Length);
                    Cv2.ImWrite(Path.Combine(directory, $"mask_{i:000}.png"), a);
                }
            }
            Cv2.ImWrite(Path.Combine(directory, "overlay.png"), overlay);
            var metadata = new { result.Prompt, result.Response, result.Count, result.Rejected, result.Truncated,
                result.LatencyMilliseconds, Warmup = warmup, Iterations = iterations, LatencySamplesMs = samples,
                Annotations = result.Annotations.Select(a => new { a.Label, a.IsPoint, a.X, a.Y, a.MaskIndex,
                    Box = new[] { a.Box.X1, a.Box.Y1, a.Box.X2, a.Box.Y2 } }), result.AbsentLabels,
                Masks = result.Masks.Select((m, i) => new { m.Width, m.Height,
                    Box = new[] { m.Box.X1, m.Box.Y1, m.Box.X2, m.Box.Y2 }, Alpha = $"alpha_{i:000}.f32",
                    Binary = m.Binary.Length == 0 ? null : $"mask_{i:000}.png" }) };
            File.WriteAllText(Option("json", Path.Combine(directory, "result.json")), JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine(result.Response);
            Console.WriteLine($"count={result.Count} rejected={result.Rejected} truncated={result.Truncated} latency_ms={result.LatencyMilliseconds:F1}");
            if (args.ContainsKey("timing"))
                Console.WriteLine($"warmup={warmup} iterations={iterations} mean_ms={samples.Average():F1} min_ms={samples.Min():F1} max_ms={samples.Max():F1}");
            return result.Truncated ? 3 : 0;
        }
        catch (Exception e) { Console.Error.WriteLine(e.Message); return 2; }
    }
}
