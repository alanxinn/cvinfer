using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using OpenCvSharp;
using alanxinn;

internal static class MediaPipeHandsDemo
{
    internal static int Run(Dictionary<string, string> options)
    {
        string palm, model, imagePath, outputPath, jsonPath;
        int maxHands, warmup, iterations;
        float palmThreshold, handThreshold, nms;
        try
        {
            palm = Required(options, "palm-model");
            model = Required(options, "model");
            imagePath = Required(options, "img");
            outputPath = options.GetValueOrDefault("output", "cvinfer_hands.jpg");
            jsonPath = options.GetValueOrDefault("json", Path.ChangeExtension(outputPath, ".json"));
            if (Path.GetFullPath(outputPath).Equals(Path.GetFullPath(jsonPath), StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("--output and --json must use different paths");
            var backend = options.GetValueOrDefault("backend", "tensorrt").Trim().ToLowerInvariant();
            if (backend is not ("tensorrt" or "trt"))
                throw new ArgumentException("mediapipe-hands requires --backend tensorrt");
            foreach (var key in new[] { "preprocess", "postprocess" })
                if (options.TryGetValue(key, out var value) && value.Trim().ToLowerInvariant() is not ("auto" or "cpu"))
                    throw new ArgumentException($"mediapipe-hands supports --{key} auto or cpu");
            palmThreshold = Threshold(options, "palm-threshold", .5f);
            handThreshold = Threshold(options, "hand-threshold", .5f);
            nms = Threshold(options, "nms", .3f);
            maxHands = Integer(options, "max-hands", 2, 1, 64);
            warmup = Integer(options, "warmup", 0, 0);
            iterations = Integer(options, "iterations", 1, 1);
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 2;
        }

        try
        {
            using var image = Cv2.ImRead(imagePath, ImreadModes.Color);
            if (image.Empty()) throw new InvalidOperationException($"failed to read image: {imagePath}");
            using var detector = new MediaPipeHandDetector(palm, model, palmThreshold, handThreshold, nms, maxHands);
            for (var i = 0; i < warmup; ++i)
                detector.InferBgr(image.Data, image.Cols, image.Rows, image.Channels());
            var timer = new Stopwatch();
            var timed = options.ContainsKey("timing");
            double total = 0, minimum = double.PositiveInfinity, maximum = 0;
            MediaPipeHand[] hands = Array.Empty<MediaPipeHand>();
            for (var i = 0; i < iterations; ++i)
            {
                if (timed) timer.Restart();
                hands = detector.InferBgr(image.Data, image.Cols, image.Rows, image.Channels());
                if (timed)
                {
                    timer.Stop();
                    var ms = timer.Elapsed.TotalMilliseconds;
                    total += ms; minimum = Math.Min(minimum, ms); maximum = Math.Max(maximum, ms);
                }
            }
            if (timed)
            {
                Console.WriteLine($"benchmark warmup={warmup} iterations={iterations}");
                Console.WriteLine($"total avg={total / iterations:F4} min={minimum:F4} max={maximum:F4} ms");
            }
            Console.WriteLine($"model=mediapipe-hands image={image.Cols}x{image.Rows} results={hands.Length}");
            using var output = image.Clone();
            for (var i = 0; i < hands.Length; ++i)
            {
                var hand = hands[i];
                Draw(output, hand);
                Console.WriteLine($"{i}: hand={hand.Label} score={hand.Confidence:F4} palm={hand.PalmScore:F4} " +
                    $"box={string.Join(",", hand.Box.Select(value => value.ToString("F1", CultureInfo.InvariantCulture)))} keypoints=21");
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath))!);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(jsonPath))!);
            if (!Cv2.ImWrite(outputPath, output)) throw new IOException($"failed to save result image: {outputPath}");
            var report = new { model = "mediapipe-hands", image = Path.GetFullPath(imagePath),
                width = image.Cols, height = image.Rows, hands };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, new JsonSerializerOptions
                { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
            Console.WriteLine($"result image saved: {outputPath}");
            Console.WriteLine($"result JSON saved: {jsonPath}");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }

    private static string Required(Dictionary<string, string> options, string key) =>
        options.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value)
            ? value : throw new ArgumentException($"missing required option: --{key}");
    private static int Integer(Dictionary<string, string> options, string key, int fallback, int minimum, int maximum = int.MaxValue)
    {
        var value = options.TryGetValue(key, out var text) ? int.Parse(text, CultureInfo.InvariantCulture) : fallback;
        if (value < minimum || value > maximum) throw new ArgumentOutOfRangeException(key, $"must be in [{minimum},{maximum}]");
        return value;
    }
    private static float Threshold(Dictionary<string, string> options, string key, float fallback)
    {
        var value = options.TryGetValue(key, out var text) ? float.Parse(text, CultureInfo.InvariantCulture) : fallback;
        if (!float.IsFinite(value) || value < 0 || value > 1) throw new ArgumentOutOfRangeException(key, "must be finite and in [0,1]");
        return value;
    }
    private static void Draw(Mat image, MediaPipeHand hand)
    {
        int[,] connections = { {0,1}, {1,2}, {2,3}, {3,4}, {0,5}, {5,6}, {6,7}, {7,8},
            {5,9}, {9,10}, {10,11}, {11,12}, {9,13}, {13,14}, {14,15}, {15,16},
            {13,17}, {0,17}, {17,18}, {18,19}, {19,20} };
        var points = Enumerable.Range(0, 21).Select(i =>
            new Point((int)MathF.Round(hand.Landmarks[i * 3]), (int)MathF.Round(hand.Landmarks[i * 3 + 1]))).ToArray();
        for (var i = 0; i < connections.GetLength(0); ++i)
            Cv2.Line(image, points[connections[i, 0]], points[connections[i, 1]], Scalar.LimeGreen, 2, LineTypes.AntiAlias);
        foreach (var point in points) Cv2.Circle(image, point, 3, Scalar.Red, -1, LineTypes.AntiAlias);
        var left = Math.Clamp((int)MathF.Floor(hand.Box[0]), 0, image.Cols - 1);
        var top = Math.Clamp((int)MathF.Floor(hand.Box[1]), 0, image.Rows - 1);
        var right = Math.Clamp((int)MathF.Ceiling(hand.Box[2]), left + 1, image.Cols);
        var bottom = Math.Clamp((int)MathF.Ceiling(hand.Box[3]), top + 1, image.Rows);
        Cv2.Rectangle(image, new Rect(left, top, right - left, bottom - top), Scalar.LimeGreen, 2);
        Cv2.PutText(image, $"{hand.Label} {hand.Confidence:F3}", new Point(left, Math.Max(15, top - 4)),
            HersheyFonts.HersheySimplex, .5, Scalar.Blue, 1, LineTypes.AntiAlias);
    }
}
