using System;
using System.Collections.Generic;
using System.Linq;

namespace alanxinn.TensorRt;

public enum TensorRtPrecision
{
    Fp32 = 0,
    Fp16 = 1,
    Int8 = 2
}

public sealed class TensorRtShape
{
    public IReadOnlyList<int> Dimensions { get; }

    public TensorRtShape(params int[] dimensions)
        : this((IEnumerable<int>)dimensions)
    {
    }

    public TensorRtShape(IEnumerable<int> dimensions)
    {
        if (dimensions is null) throw new ArgumentNullException(nameof(dimensions));
        var values = dimensions.ToArray();
        if (values.Length == 0 || values.Any(value => value <= 0))
            throw new ArgumentException("Tensor shape dimensions must be positive", nameof(dimensions));
        Dimensions = values;
    }

    public override string ToString() => string.Join("x", Dimensions);
}

public sealed class TensorRtOptimizationProfile
{
    public IReadOnlyDictionary<string, TensorRtShape> MinShapes { get; set; } =
        new Dictionary<string, TensorRtShape>(StringComparer.Ordinal);

    public IReadOnlyDictionary<string, TensorRtShape> OptShapes { get; set; } =
        new Dictionary<string, TensorRtShape>(StringComparer.Ordinal);

    public IReadOnlyDictionary<string, TensorRtShape> MaxShapes { get; set; } =
        new Dictionary<string, TensorRtShape>(StringComparer.Ordinal);
}

public sealed class TensorRtBuildOptions
{
    public TensorRtPrecision Precision { get; set; } = TensorRtPrecision.Fp32;

    /// <summary>TensorRT builder workspace size in MiB.</summary>
    public long WorkspaceMiB { get; set; } = 4096;

    /// <summary>-1 keeps the current CUDA device.</summary>
    public int DeviceId { get; set; } = -1;
    public bool EnableTf32 { get; set; } = true;
    public bool EnableSparseWeights { get; set; }
    public bool StrictTypes { get; set; }

    /// <summary>
    /// Dynamic ONNX inputs require one min/opt/max profile per optimization profile.
    /// </summary>
    public IReadOnlyList<TensorRtOptimizationProfile> OptimizationProfiles { get; set; } =
        Array.Empty<TensorRtOptimizationProfile>();

    /// <summary>
    /// Receives coarse conversion stages. The engine build stage reports heartbeat
    /// messages because TensorRT does not expose a reliable build percentage.
    /// </summary>
    public IProgress<TensorRtBuildProgress>? Progress { get; set; }
}

public sealed class TensorRtBuildProgress
{
    internal TensorRtBuildProgress(int stage, int percent, bool isHeartbeat, string message, TimeSpan elapsed)
    {
        Stage = stage;
        Percent = percent;
        IsHeartbeat = isHeartbeat;
        Message = message;
        Elapsed = elapsed;
    }

    public int Stage { get; }
    public int Percent { get; }
    public bool IsHeartbeat { get; }
    public string Message { get; }
    public TimeSpan Elapsed { get; }
}

public sealed class TensorRtBuildResult
{
    internal TensorRtBuildResult(
        string onnxPath,
        string enginePath,
        TensorRtPrecision precision,
        TimeSpan duration,
        long engineSizeBytes)
    {
        OnnxPath = onnxPath;
        EnginePath = enginePath;
        Precision = precision;
        Duration = duration;
        EngineSizeBytes = engineSizeBytes;
    }

    public string OnnxPath { get; }
    public string EnginePath { get; }
    public TensorRtPrecision Precision { get; }
    public TimeSpan Duration { get; }
    public long EngineSizeBytes { get; }
}

public sealed class TensorRtBuilderException : Exception
{
    public int Status { get; }

    internal TensorRtBuilderException(int status, string message)
        : base($"TensorRT builder status={status}: {message}")
    {
        Status = status;
    }
}
