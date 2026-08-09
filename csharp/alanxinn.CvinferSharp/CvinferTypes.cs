namespace alanxinn;

public enum CvinferModelType
{
    YoloDetection = 1,
    YoloSegmentation = 2,
    RfDetrDetection = 3,
    RfDetrSegmentation = 4
}

public enum CvinferComputeBackend
{
    Auto = 0,
    Cpu = 1,
    Cuda = 2
}
public sealed class CvinferConfig
{
    // 由 TensorRT engine 自动读取；保留属性仅用于底层 ABI 兼容。
    // 设为正值时仍可覆盖 engine shape。
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    public int ClassCount { get; set; } = 80;
    public float ConfidenceThreshold { get; set; } = 0.25f;
    public float NmsThreshold { get; set; } = 0.45f;
    public float MaskThreshold { get; set; } = 0.5f;
    public bool ClassAgnosticNms { get; set; }
    public CvinferComputeBackend PreprocessBackend { get; set; } = CvinferComputeBackend.Auto;
    public CvinferComputeBackend PostprocessBackend { get; set; } = CvinferComputeBackend.Auto;
    public string? InputName { get; set; }
    public string? OutputName { get; set; }
    public string? PrototypeName { get; set; }
}

public readonly record struct CvinferBox(float X1, float Y1, float X2, float Y2, float Score, int ClassId);
public readonly record struct CvinferDetection(CvinferBox Box);
public readonly record struct CvinferSegmentation(CvinferBox Box, int MaskWidth, int MaskHeight, int MaskBytes);
public readonly record struct CvinferTiming(double PreprocessMs, double InferenceMs, double PostprocessMs, double TotalMs);

public sealed class CvinferException : Exception
{
    public int Status { get; }
    public CvinferException(int status, string message) : base($"cvinfer status={status}: {message}") => Status = status;
}
