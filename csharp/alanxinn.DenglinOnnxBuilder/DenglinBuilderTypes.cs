using System;
using System.Collections.Generic;
using System.Linq;

namespace alanxinn.Denglin;

public enum DenglinEngineFormat { RawEngine = 0 }
public enum DenglinCluster
{
    Cluster0 = 0, Cluster1 = 1, Cluster2 = 2, Cluster3 = 3,
    Cluster01 = 4, Cluster23 = 5, Cluster02 = 6, Cluster13 = 7,
    Cluster03 = 8, Cluster12 = 9, Cluster0123 = 10
}

public sealed class DenglinShape
{
    public IReadOnlyList<int> Dimensions { get; }
    public bool IsDynamic => Dimensions.Any(value => value < 0);

    public DenglinShape(params int[] dimensions) : this((IEnumerable<int>)dimensions) { }

    public DenglinShape(IEnumerable<int> dimensions)
    {
        if (dimensions is null) throw new ArgumentNullException(nameof(dimensions));
        var values = dimensions.ToArray();
        if (values.Length == 0 || values.Any(value => value == 0 || value < -1))
            throw new ArgumentException("Dimensions must be positive or -1 for a dynamic dimension", nameof(dimensions));
        Dimensions = values;
    }

    public override string ToString() => string.Join("x", Dimensions);
}

public sealed class DenglinOptimizationProfile
{
    public DenglinShape MinShape { get; set; } = new DenglinShape(1, 3, 640, 640);
    public DenglinShape OptShape { get; set; } = new DenglinShape(1, 3, 640, 640);
    public DenglinShape MaxShape { get; set; } = new DenglinShape(1, 3, 640, 640);
}

public sealed class DenglinBuildOptions
{
    /// <summary>Optional input name. A single ONNX input is selected automatically when omitted.</summary>
    public string? InputName { get; set; }
    /// <summary>Optional input shape. Static ONNX shapes are selected automatically when omitted.</summary>
    public DenglinShape? InputShape { get; set; }
    /// <summary>Optional output-name override. Denglin discovers graph outputs when omitted or empty.</summary>
    public IReadOnlyList<string>? OutputNames { get; set; }
    public int MaxBatchSize { get; set; } = 1;
    public int BatchSize { get; set; } = 1;
    public DenglinCluster Cluster { get; set; } = DenglinCluster.Cluster0;
    public uint BuilderFlags { get; set; }
    public DenglinEngineFormat EngineFormat { get; set; } = DenglinEngineFormat.RawEngine;
    public IReadOnlyList<DenglinOptimizationProfile> OptimizationProfiles { get; set; } =
        Array.Empty<DenglinOptimizationProfile>();
    public string? PluginLibrary { get; set; }
    public string? PluginModule { get; set; }
    public string? PluginFirstOp { get; set; }
    public string? SdkRoot { get; set; }
    public string? PythonDirectory { get; set; }
    public string? TestDataDirectory { get; set; }
    /// <summary>Fallback batch used for symbolic ONNX dimensions when no profile is supplied.</summary>
    public int DynamicBatchSize { get; set; } = 1;
    /// <summary>Fallback spatial size used for symbolic ONNX dimensions when no profile is supplied.</summary>
    public int DynamicSpatialSize { get; set; } = 640;
    public IProgress<DenglinBuildProgress>? Progress { get; set; }
}

public sealed class DenglinBuildProgress
{
    internal DenglinBuildProgress(int stage, int percent, bool isHeartbeat, string message, TimeSpan elapsed)
    {
        Stage = stage; Percent = percent; IsHeartbeat = isHeartbeat; Message = message; Elapsed = elapsed;
    }
    public int Stage { get; }
    public int Percent { get; }
    public bool IsHeartbeat { get; }
    public string Message { get; }
    public TimeSpan Elapsed { get; }
}

public sealed class DenglinBuildResult
{
    internal DenglinBuildResult(string onnxPath, string enginePath, DenglinEngineFormat format,
        TimeSpan duration, long engineSizeBytes)
    {
        OnnxPath = onnxPath; EnginePath = enginePath; Format = format;
        Duration = duration; EngineSizeBytes = engineSizeBytes;
    }
    public string OnnxPath { get; }
    public string EnginePath { get; }
    public DenglinEngineFormat Format { get; }
    public TimeSpan Duration { get; }
    public long EngineSizeBytes { get; }
}

public sealed class DenglinBuilderException : Exception
{
    public int Status { get; }
    internal DenglinBuilderException(int status, string message)
        : base($"Denglin builder status={status}: {message}") { Status = status; }
}
