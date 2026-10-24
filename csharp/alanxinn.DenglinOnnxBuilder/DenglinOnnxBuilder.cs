using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace alanxinn.Denglin;

public static class DenglinOnnxBuilder
{
    public static DenglinBuildResult Build(string onnxPath, string enginePath,
        DenglinBuildOptions? options = null)
    {
        options ??= new DenglinBuildOptions();
        var source = NormalizeExistingFile(onnxPath, "ONNX");
        var destination = NormalizeOutputPath(enginePath);
        if (string.Equals(source, destination, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("ONNX and engine paths must be different", nameof(enginePath));

        var metadata = OnnxModelMetadata.Read(source);
        var resolved = ResolveOptions(options, metadata);
        Validate(options, resolved.InputName, resolved.InputShape, resolved.OutputNames);
        if (options.EngineFormat != DenglinEngineFormat.RawEngine)
            throw new NotSupportedException("The nnexec Denglin builder currently produces RawEngine only.");
        if (options.OptimizationProfiles is { Count: > 0 })
            throw new NotSupportedException("nnexec accepts one --shape per input; optimization profiles are not supported.");

        DenglinRuntime.Initialize(options.SdkRoot, options.PythonDirectory, options.TestDataDirectory);
        var nnexec = FindNnexec();
        Directory.CreateDirectory(Path.GetDirectoryName(destination) ?? Environment.CurrentDirectory);

        var startedAt = DateTime.UtcNow;
        Report(options.Progress, 2, 10, "nnexec Denglin builder configuration created", startedAt);
        var arguments = BuildArguments(source, destination, resolved, options);
        var result = RunNnexec(nnexec, arguments);
        if (result.ExitCode != 0 || !File.Exists(destination) || new FileInfo(destination).Length == 0)
        {
            var detail = string.IsNullOrWhiteSpace(result.Error) ? result.Output : result.Error;
            throw new DenglinBuilderException(result.ExitCode,
                "Denglin nnexec conversion failed" +
                (string.IsNullOrWhiteSpace(detail) ? string.Empty : ": " + detail.Trim()));
        }
        Report(options.Progress, 4, 90, "Denglin engine serialized by nnexec", startedAt);
        Report(options.Progress, 6, 100, "Conversion completed", startedAt);
        return new DenglinBuildResult(source, destination, options.EngineFormat,
            DateTime.UtcNow - startedAt, new FileInfo(destination).Length);
    }

    private sealed class ResolvedOptions
    {
        public ResolvedOptions(string inputName, DenglinShape inputShape,
            IReadOnlyList<string> outputNames)
        {
            InputName = inputName;
            InputShape = inputShape;
            OutputNames = outputNames;
        }
        public string InputName { get; }
        public DenglinShape InputShape { get; }
        public IReadOnlyList<string> OutputNames { get; }
    }

    private sealed class ProcessResult
    {
        public int ExitCode { get; set; }
        public string Output { get; set; } = string.Empty;
        public string Error { get; set; } = string.Empty;
    }

    private static ResolvedOptions ResolveOptions(DenglinBuildOptions options,
        OnnxModelMetadata metadata)
    {
        if (metadata.Inputs.Count == 0)
            throw new InvalidDataException("The ONNX graph has no runtime inputs.");
        OnnxTensorMetadata input;
        if (!string.IsNullOrWhiteSpace(options.InputName))
        {
            input = metadata.Inputs.FirstOrDefault(value =>
                string.Equals(value.Name, options.InputName, StringComparison.Ordinal))
                ?? throw new ArgumentException("InputName was not found in the ONNX graph: " + options.InputName,
                    nameof(options));
        }
        else
        {
            if (metadata.Inputs.Count != 1)
                throw new ArgumentException("The ONNX graph has multiple runtime inputs. Set InputName explicitly.",
                    nameof(options));
            input = metadata.Inputs[0];
        }

        var metadataShape = ToDenglinShape(input);
        var profiles = options.OptimizationProfiles ?? Array.Empty<DenglinOptimizationProfile>();
        var shape = options.InputShape ??
            (metadataShape.IsDynamic && profiles.Count > 0
                ? metadataShape : MaterializeDynamicShape(metadataShape, options));
        var outputs = options.OutputNames is { Count: > 0 }
            ? options.OutputNames.ToArray()
            : metadata.Outputs.Select(value => value.Name)
                .Where(value => !string.IsNullOrWhiteSpace(value)).ToArray();
        return new ResolvedOptions(input.Name, shape, outputs);
    }

    private static string BuildArguments(string source, string destination,
        ResolvedOptions resolved, DenglinBuildOptions options)
    {
        var arguments = new List<string>
        {
            Quote(source), "--input", Quote(resolved.InputName),
            "--output", Quote(string.Join(",", resolved.OutputNames)),
            "--shape", Quote(resolved.InputName + ":" + resolved.InputShape),
            "--maxBatch", options.MaxBatchSize.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--clusterConfig", Quote(ToClusterConfig(options.Cluster)),
            "--batch", options.BatchSize.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--iterations", "1", "--warmUp", "0", "--nocheck",
            "--saveEngine", Quote(destination)
        };

        var builderFlags = ToBuilderFlags(options.BuilderFlags);
        if (!string.IsNullOrWhiteSpace(builderFlags))
        {
            arguments.Add("--builderFlags");
            arguments.Add(Quote(builderFlags));
        }
        AddOptional(arguments, "--pluginNneLib", options.PluginLibrary);
        AddOptional(arguments, "--pluginModule", options.PluginModule);
        if (!string.IsNullOrWhiteSpace(options.PluginFirstOp))
            throw new NotSupportedException("nnexec does not expose PluginFirstOp.");
        return string.Join(" ", arguments);
    }

    private static ProcessResult RunNnexec(string nnexec, string arguments)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = nnexec,
            Arguments = arguments,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = Environment.CurrentDirectory
        };
        using var process = new Process { StartInfo = startInfo };
        if (!process.Start()) throw new InvalidOperationException("Unable to start Denglin nnexec.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        process.WaitForExit();
        TaskWait(output, error);
        var stdout = output.Result;
        var stderr = error.Result;
        if (!string.IsNullOrWhiteSpace(stdout)) Console.Write(stdout);
        if (!string.IsNullOrWhiteSpace(stderr)) Console.Error.Write(stderr);
        return new ProcessResult { ExitCode = process.ExitCode, Output = stdout, Error = stderr };
    }

    private static void TaskWait(System.Threading.Tasks.Task<string> output,
        System.Threading.Tasks.Task<string> error)
    {
        System.Threading.Tasks.Task.WaitAll(output, error);
    }

    private static string FindNnexec()
    {
        var sdk = DenglinRuntime.SdkRoot;
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(sdk))
            candidates.Add(Path.Combine(sdk, "bin", "nnexec.exe"));
        candidates.Add(Path.Combine(AppContext.BaseDirectory, "nnexec.exe"));
        candidates.Add(Path.Combine(AppContext.BaseDirectory,
            "runtimes", "win-x64", "denglin", "sdk", "bin", "nnexec.exe"));
        var executable = candidates.FirstOrDefault(File.Exists);
        if (executable == null)
            throw new FileNotFoundException("Denglin nnexec.exe was not found in the selected SDK runtime.");
        return executable;
    }

    private static string ToClusterConfig(DenglinCluster cluster) => cluster switch
    {
        DenglinCluster.Cluster0 => "0",
        DenglinCluster.Cluster1 => "1",
        DenglinCluster.Cluster2 => "2",
        DenglinCluster.Cluster3 => "3",
        DenglinCluster.Cluster01 => "01",
        DenglinCluster.Cluster23 => "23",
        DenglinCluster.Cluster02 => "02",
        DenglinCluster.Cluster13 => "13",
        DenglinCluster.Cluster03 => "03",
        DenglinCluster.Cluster12 => "12",
        DenglinCluster.Cluster0123 => "0123",
        _ => throw new ArgumentOutOfRangeException(nameof(cluster), cluster, "Unknown Denglin cluster configuration")
    };

    private static string? ToBuilderFlags(uint flags)
    {
        const uint spmAlloc = 1;
        const uint userConst = 1u << 4;
        const uint ignoreWeights = 1u << 5;
        const uint known = spmAlloc | userConst | ignoreWeights;
        if ((flags & ~known) != 0)
            throw new NotSupportedException("BuilderFlags contains an nnexec-unsupported value: " + flags);
        var values = new List<string>();
        if ((flags & spmAlloc) != 0) values.Add("SpmAlloc");
        if ((flags & userConst) != 0) values.Add("UserConst");
        if ((flags & ignoreWeights) != 0) values.Add("SlzIgnoreWeights");
        return values.Count == 0 ? null : string.Join(",", values);
    }

    private static void AddOptional(ICollection<string> arguments, string name, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value))
        {
            arguments.Add(name);
            arguments.Add(Quote(value));
        }
    }

    private static string Quote(string value) =>
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

    private static void Report(IProgress<DenglinBuildProgress>? progress, int stage,
        int percent, string message, DateTime startedAt)
    {
        progress?.Report(new DenglinBuildProgress(stage, percent, false, message,
            DateTime.UtcNow - startedAt));
    }

    private static DenglinShape ToDenglinShape(OnnxTensorMetadata tensor)
    {
        if (tensor.Shape.Count == 0) throw new InvalidDataException("ONNX input has no tensor shape: " + tensor.Name);
        return new DenglinShape(tensor.Shape.Select(value =>
            value <= 0 ? -1 : value > int.MaxValue ? throw new InvalidDataException("ONNX dimension is too large") : (int)value));
    }

    private static DenglinShape MaterializeDynamicShape(DenglinShape shape, DenglinBuildOptions options)
    {
        if (!shape.IsDynamic) return shape;
        var dimensions = shape.Dimensions.Select((value, index) => value > 0 ? value :
            index == 0 ? options.DynamicBatchSize : index == 1 && shape.Dimensions.Count >= 4 ? 3 : options.DynamicSpatialSize).ToArray();
        return new DenglinShape(dimensions);
    }

    private static void Validate(DenglinBuildOptions options, string inputName,
        DenglinShape inputShape, IReadOnlyList<string> outputs)
    {
        if (string.IsNullOrWhiteSpace(inputName) || inputShape == null)
            throw new ArgumentException("Input name and shape are required", nameof(options));
        if (inputShape.IsDynamic)
            throw new ArgumentException("InputShape must be materialized to a static shape", nameof(options));
        if (options.MaxBatchSize <= 0 || options.BatchSize <= 0 || options.BatchSize > options.MaxBatchSize)
            throw new ArgumentException("BatchSize must be positive and not exceed MaxBatchSize", nameof(options));
        if (outputs.Count == 0)
            throw new InvalidDataException("The ONNX graph has no runtime outputs.");
        if (outputs.Any(string.IsNullOrWhiteSpace)) throw new ArgumentException("OutputNames contains an empty name", nameof(options));
    }

    private static string NormalizeExistingFile(string path, string label)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException(label + " path is required", nameof(path));
        var full = Path.GetFullPath(path);
        if (!File.Exists(full)) throw new FileNotFoundException(label + " file was not found", full);
        return full;
    }

    private static string NormalizeOutputPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Engine path is required", nameof(path));
        return Path.GetFullPath(path);
    }
}
