using System.Globalization;
using alanxinn.TensorRt;

try
{
    var arguments = DemoArguments.Parse(args);
    if (arguments.ShowHelp)
    {
        PrintUsage();
        return 0;
    }

    var buildOptions = new TensorRtBuildOptions
    {
        Precision = arguments.Precision,
        WorkspaceMiB = arguments.WorkspaceMiB,
        DeviceId = arguments.DeviceId,
        EnableTf32 = arguments.EnableTf32,
        EnableSparseWeights = arguments.EnableSparseWeights,
        StrictTypes = arguments.StrictTypes
    };

    if (arguments.HasOptimizationProfile)
    {
        buildOptions.OptimizationProfiles = new[]
        {
            new TensorRtOptimizationProfile
            {
                MinShapes = new Dictionary<string, TensorRtShape>(StringComparer.Ordinal)
                {
                    [arguments.InputName!] = arguments.MinShape!
                },
                OptShapes = new Dictionary<string, TensorRtShape>(StringComparer.Ordinal)
                {
                    [arguments.InputName!] = arguments.OptShape!
                },
                MaxShapes = new Dictionary<string, TensorRtShape>(StringComparer.Ordinal)
                {
                    [arguments.InputName!] = arguments.MaxShape!
                }
            }
        };
    }

    Console.WriteLine($"ONNX : {Path.GetFullPath(arguments.OnnxPath)}");
    Console.WriteLine($"Engine: {Path.GetFullPath(arguments.EnginePath)}");
    Console.WriteLine($"Precision={buildOptions.Precision}, WorkspaceMiB={buildOptions.WorkspaceMiB}, DeviceId={buildOptions.DeviceId}");
    if (arguments.HasOptimizationProfile)
    {
        Console.WriteLine($"Profile: {arguments.InputName} min={arguments.MinShape} opt={arguments.OptShape} max={arguments.MaxShape}");
    }

    var result = TensorRtOnnxBuilder.Build(
        arguments.OnnxPath,
        arguments.EnginePath,
        buildOptions);

    Console.WriteLine("TensorRT engine build succeeded.");
    Console.WriteLine($"Output: {result.EnginePath}");
    Console.WriteLine($"Size: {result.EngineSizeBytes:N0} bytes");
    Console.WriteLine($"Elapsed: {result.Duration.TotalSeconds:F2} s");
    return 0;
}
catch (TensorRtBuilderException error)
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
    Console.WriteLine("TensorRT ONNX -> engine demo");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run -- --onnx model.onnx --engine model.engine [options]");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  --precision fp32|fp16|int8  Precision (default: fp16)");
    Console.WriteLine("  --workspace-mib N           TensorRT workspace in MiB (default: 4096)");
    Console.WriteLine("  --device N                  CUDA device id; -1 keeps the current device");
    Console.WriteLine("  --no-tf32                   Disable TF32 (enabled by default)");
    Console.WriteLine("  --sparse                    Enable sparse weights");
    Console.WriteLine("  --strict                    Enable strict TensorRT types");
    Console.WriteLine("  --input-name NAME           Dynamic input tensor name");
    Console.WriteLine("  --min-shape 1x3x320x320     Dynamic profile minimum shape");
    Console.WriteLine("  --opt-shape 1x3x640x640     Dynamic profile optimal shape");
    Console.WriteLine("  --max-shape 1x3x1280x1280   Dynamic profile maximum shape");
    Console.WriteLine("  --help                      Show this help");
}

sealed class DemoArguments
{
    private static readonly HashSet<string> ValueOptions = new(StringComparer.OrdinalIgnoreCase)
    {
        "onnx", "engine", "precision", "workspace-mib", "device", "input-name",
        "min-shape", "opt-shape", "max-shape"
    };

    private static readonly HashSet<string> FlagOptions = new(StringComparer.OrdinalIgnoreCase)
    {
        "no-tf32", "sparse", "strict"
    };

    public string OnnxPath { get; private init; } = string.Empty;
    public string EnginePath { get; private init; } = string.Empty;
    public TensorRtPrecision Precision { get; private init; } = TensorRtPrecision.Fp16;
    public long WorkspaceMiB { get; private init; } = 4096;
    public int DeviceId { get; private init; } = -1;
    public bool EnableTf32 { get; private init; } = true;
    public bool EnableSparseWeights { get; private init; }
    public bool StrictTypes { get; private init; }
    public bool ShowHelp { get; private init; }
    public string? InputName { get; private init; }
    public TensorRtShape? MinShape { get; private init; }
    public TensorRtShape? OptShape { get; private init; }
    public TensorRtShape? MaxShape { get; private init; }

    public bool HasOptimizationProfile => InputName is not null || MinShape is not null ||
        OptShape is not null || MaxShape is not null;

    public static DemoArguments Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var flags = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        for (var index = 0; index < args.Length; index++)
        {
            var token = args[index];
            if (token is "--help" or "-h") return new DemoArguments { ShowHelp = true };
            if (!token.StartsWith("--", StringComparison.Ordinal) || token.Length <= 2)
                throw new ArgumentException($"Invalid argument: {token}");

            var name = token[2..].Trim();
            if (FlagOptions.Contains(name))
            {
                flags.Add(name);
                continue;
            }
            if (!ValueOptions.Contains(name))
                throw new ArgumentException($"Unknown option: --{name}");
            if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException($"Option requires a value: --{name}");
            values[name] = args[++index];
        }

        var onnxPath = Required(values, "onnx");
        var enginePath = Required(values, "engine");
        var inputName = values.TryGetValue("input-name", out var input) ? input : null;
        var minShape = ParseOptionalShape(values, "min-shape");
        var optShape = ParseOptionalShape(values, "opt-shape");
        var maxShape = ParseOptionalShape(values, "max-shape");
        var hasProfileValue = inputName is not null || minShape is not null ||
            optShape is not null || maxShape is not null;
        if (hasProfileValue && (string.IsNullOrWhiteSpace(inputName) || minShape is null ||
            optShape is null || maxShape is null))
            throw new ArgumentException("--input-name, --min-shape, --opt-shape and --max-shape must be provided together");

        return new DemoArguments
        {
            OnnxPath = onnxPath,
            EnginePath = enginePath,
            Precision = ParsePrecision(values.TryGetValue("precision", out var precision) ? precision : "fp16"),
            WorkspaceMiB = ParseLong(values, "workspace-mib", 4096),
            DeviceId = ParseInt(values, "device", -1),
            EnableTf32 = !flags.Contains("no-tf32"),
            EnableSparseWeights = flags.Contains("sparse"),
            StrictTypes = flags.Contains("strict"),
            InputName = inputName,
            MinShape = minShape,
            OptShape = optShape,
            MaxShape = maxShape
        };
    }

    private static string Required(IReadOnlyDictionary<string, string> values, string name)
    {
        if (!values.TryGetValue(name, out var value) || string.IsNullOrWhiteSpace(value))
            throw new ArgumentException($"Missing required option: --{name}");
        return value;
    }

    private static TensorRtShape? ParseOptionalShape(IReadOnlyDictionary<string, string> values, string name)
        => values.TryGetValue(name, out var value) ? ParseShape(value, name) : null;

    private static TensorRtShape ParseShape(string text, string option)
    {
        try
        {
            var dimensions = text.Split('x', StringSplitOptions.RemoveEmptyEntries)
                .Select(value => int.Parse(value, NumberStyles.Integer, CultureInfo.InvariantCulture))
                .ToArray();
            return new TensorRtShape(dimensions);
        }
        catch (Exception error)
        {
            throw new ArgumentException($"Invalid --{option} value '{text}': {error.Message}", error);
        }
    }

    private static TensorRtPrecision ParsePrecision(string value)
    {
        return value.Trim().ToLowerInvariant() switch
        {
            "fp32" or "float" => TensorRtPrecision.Fp32,
            "fp16" or "half" => TensorRtPrecision.Fp16,
            "int8" => TensorRtPrecision.Int8,
            _ => throw new ArgumentException("--precision must be fp32, fp16 or int8")
        };
    }

    private static long ParseLong(IReadOnlyDictionary<string, string> values, string name, long fallback)
        => values.TryGetValue(name, out var value)
            ? long.Parse(value, NumberStyles.Integer, CultureInfo.InvariantCulture)
            : fallback;

    private static int ParseInt(IReadOnlyDictionary<string, string> values, string name, int fallback)
        => values.TryGetValue(name, out var value)
            ? int.Parse(value, NumberStyles.Integer, CultureInfo.InvariantCulture)
            : fallback;
}
