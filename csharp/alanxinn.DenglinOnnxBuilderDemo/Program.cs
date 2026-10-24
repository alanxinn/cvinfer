using System.Globalization;
using alanxinn.Denglin;

try
{
    var arguments = DemoArguments.Parse(args);
    if (arguments.ShowHelp)
    {
        PrintUsage();
        return 0;
    }

    var metadata = OnnxModelMetadata.Read(arguments.OnnxPath);
    PrintMetadata(metadata);

    var options = new DenglinBuildOptions
    {
        InputName = arguments.InputName,
        InputShape = arguments.InputShape,
        OutputNames = arguments.OutputNames,
        MaxBatchSize = arguments.MaxBatchSize,
        BatchSize = arguments.BatchSize,
        Cluster = arguments.Cluster,
        SdkRoot = arguments.SdkRoot,
        PythonDirectory = arguments.PythonDirectory,
        TestDataDirectory = arguments.TestDataDirectory,
        Progress = new Progress<DenglinBuildProgress>(PrintBuildProgress)
    };
    DenglinRuntime.DownloadProgress = new Progress<DenglinRuntimeDownloadProgress>(PrintDownloadProgress);

    Console.WriteLine($"ONNX  : {Path.GetFullPath(arguments.OnnxPath)}");
    Console.WriteLine($"Output: {Path.GetFullPath(arguments.EnginePath)}");
    if (options.InputName is not null || options.InputShape is not null)
        Console.WriteLine($"Input override: {(options.InputName ?? "auto")} {(options.InputShape?.ToString() ?? "auto")}");
    var result = DenglinOnnxBuilder.Build(arguments.OnnxPath, arguments.EnginePath, options);
    Console.WriteLine("Denglin engine build succeeded.");
    Console.WriteLine($"Output: {result.EnginePath}");
    Console.WriteLine($"Size: {result.EngineSizeBytes:N0} bytes");
    Console.WriteLine($"Elapsed: {result.Duration.TotalSeconds:F2} s");
    return 0;
}
catch (DenglinBuilderException error)
{
    Console.Error.WriteLine(error.Message);
    return 1;
}
catch (Exception error)
{
    Console.Error.WriteLine($"Error: {error.Message}");
    Console.Error.WriteLine();
    PrintUsage();
    return 2;
}

static void PrintUsage()
{
    Console.WriteLine("Denglin ONNX -> engine demo");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project .\\csharp\\alanxinn.DenglinOnnxBuilderDemo -- --onnx model.onnx --engine model.denglin.engine [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --input-name NAME           Optional input tensor name; one input is detected automatically");
    Console.WriteLine("  --input-shape 1x3x640x640  Optional materialized static shape; ONNX metadata is used by default");
    Console.WriteLine("  --output-names a,b          Optional output tensor names; graph outputs are detected automatically");
    Console.WriteLine("  --max-batch N               Maximum batch size (default: 1)");
    Console.WriteLine("  --batch N                   Build batch size (default: 1)");
    Console.WriteLine("  --sdk-root PATH             Optional local SDK override, for example E:\\dlgpu\\sdk");
    Console.WriteLine("  --python PATH               Optional local Python override with SDK dependencies");
    Console.WriteLine("  --test-data PATH            Writable SDK test-data directory");
    Console.WriteLine("  Runtime SDK/Python          Auto-discovered, bundled, cached or downloaded on first build");
    Console.WriteLine("  --help                      Show this help");
}

static void PrintMetadata(OnnxModelMetadata metadata)
{
    Console.WriteLine("ONNX metadata:");
    foreach (var input in metadata.Inputs)
        Console.WriteLine($"  Input : {input.Name} {input.ShapeText} {input.DataTypeName}");
    foreach (var output in metadata.Outputs)
        Console.WriteLine($"  Output: {output.Name} {output.ShapeText} {output.DataTypeName}");
}

static void PrintBuildProgress(DenglinBuildProgress value)
{
    const int width = 32;
    var completed = Math.Clamp(value.Percent, 0, 100) * width / 100;
    var bar = new string('#', completed) + new string('-', width - completed);
    var heartbeat = value.IsHeartbeat ? " (working)" : string.Empty;
    Console.Write($"\r[{bar}] {value.Percent,3}% {value.Message}{heartbeat}   ");
    if (value.Percent >= 100) Console.WriteLine();
}

static void PrintDownloadProgress(DenglinRuntimeDownloadProgress value)
{
    const int width = 32;
    var percent = Math.Clamp(value.Percent, 0, 100);
    var completed = value.Percent >= 0 ? percent * width / 100 : 0;
    var bar = value.Percent >= 0
        ? new string('#', completed) + new string('-', width - completed)
        : new string('.', width);
    var downloaded = FormatBytes(value.BytesDownloaded);
    var total = value.TotalBytes > 0 ? " / " + FormatBytes(value.TotalBytes) : string.Empty;
    var speed = FormatBytes((long)value.BytesPerSecond) + "/s";
    var percentText = value.Percent >= 0 ? $"{percent,3}%" : "  ?%";
    Console.Write($"\rDownloading Denglin runtime [{bar}] {percentText} {downloaded}{total} {speed}   ");
    if (value.IsCompleted) Console.WriteLine();
}

static string FormatBytes(long bytes)
{
    if (bytes < 1024) return bytes + " B";
    var value = (double)bytes;
    var units = new[] { "KB", "MB", "GB", "TB" };
    var index = -1;
    do
    {
        value /= 1024;
        index++;
    } while (value >= 1024 && index < units.Length - 1);
    return $"{value:F1} {units[index]}";
}

sealed class DemoArguments
{
    private static readonly HashSet<string> ValueOptions = new(StringComparer.OrdinalIgnoreCase)
    {
        "onnx", "engine", "input-name", "input-shape",
        "output-names", "max-batch", "batch", "sdk-root", "python", "test-data"
    };

    public string OnnxPath { get; private init; } = string.Empty;
    public string EnginePath { get; private init; } = string.Empty;
    public string? InputName { get; private init; }
    public DenglinShape? InputShape { get; private init; }
    public string[] OutputNames { get; private init; } = Array.Empty<string>();
    public int MaxBatchSize { get; private init; } = 1;
    public int BatchSize { get; private init; } = 1;
    public DenglinCluster Cluster { get; private init; } = DenglinCluster.Cluster0;
    public bool ShowHelp { get; private init; }
    public string? SdkRoot { get; private init; }
    public string? PythonDirectory { get; private init; }
    public string? TestDataDirectory { get; private init; }
    public static DemoArguments Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var index = 0; index < args.Length; index++)
        {
            var token = args[index];
            if (token is "--help" or "-h") return new DemoArguments { ShowHelp = true };
            if (!token.StartsWith("--", StringComparison.Ordinal) || token.Length <= 2)
                throw new ArgumentException($"Invalid argument: {token}");
            var name = token[2..];
            if (!ValueOptions.Contains(name)) throw new ArgumentException($"Unknown option: --{name}");
            if (++index >= args.Length || args[index].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"Option requires a value: --{name}");
            values[name] = args[index];
        }

        var onnx = Required(values, "onnx");
        var engine = Required(values, "engine");
        var inputName = Optional(values, "input-name");
        var inputShape = values.ContainsKey("input-shape")
            ? ParseShape(values["input-shape"], "input-shape")
            : null;
        if (inputShape is { IsDynamic: true })
            throw new ArgumentException("--input-shape must be a materialized static shape");

        return new DemoArguments
        {
            OnnxPath = onnx,
            EnginePath = engine,
            InputName = inputName,
            InputShape = inputShape,
            OutputNames = Optional(values, "output-names")?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) ?? Array.Empty<string>(),
            MaxBatchSize = ParseInt(values, "max-batch", 1),
            BatchSize = ParseInt(values, "batch", 1),
            SdkRoot = Optional(values, "sdk-root"),
            PythonDirectory = Optional(values, "python"),
            TestDataDirectory = Optional(values, "test-data"),
        };
    }

    private static string Required(IReadOnlyDictionary<string, string> values, string name)
        => values.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value)
            ? value : throw new ArgumentException($"Missing required option: --{name}");
    private static string Get(IReadOnlyDictionary<string, string> values, string name, string fallback)
        => values.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value) ? value : fallback;
    private static string? Optional(IReadOnlyDictionary<string, string> values, string name)
        => values.TryGetValue(name, out var value) && !string.IsNullOrWhiteSpace(value) ? value : null;
    private static int ParseInt(IReadOnlyDictionary<string, string> values, string name, int fallback)
        => int.Parse(Get(values, name, fallback.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
    private static DenglinShape ParseShape(string text, string option)
    {
        try
        {
            var dimensions = text.Split('x', StringSplitOptions.RemoveEmptyEntries)
                .Select(value => int.Parse(value, NumberStyles.Integer, CultureInfo.InvariantCulture)).ToArray();
            return new DenglinShape(dimensions);
        }
        catch (Exception error) { throw new ArgumentException($"Invalid --{option}: {error.Message}", error); }
    }
}
