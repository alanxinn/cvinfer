namespace alanxinn;

public enum CvinferModelType
{
    Yolo11Detection = 1,
    Yolo11Segmentation = 2,
    RfDetrDetection = 3,
    RfDetrSegmentation = 4,
    Yolo11Pose = 5
}

public enum CvinferComputeBackend
{
    Auto = 0,
    Cpu = 1,
    Cuda = 2
}
public sealed class CvinferConfig
{
    // 由 TensorRT engine 自动读取；设为正值时覆盖 engine shape。
    public int InputWidth { get; set; }
    public int InputHeight { get; set; }
    // <= 0 lets the native decoder infer the category count from the engine output.
    public int ClassCount { get; set; }
    public float ConfidenceThreshold { get; set; } = 0.25f;
    public float NmsThreshold { get; set; } = 0.45f;
    public float MaskThreshold { get; set; } = 0.5f;
    public bool ClassAgnosticNms { get; set; }
    public int KeypointCount { get; set; } = 17;
    public int KeypointDimensions { get; set; } = 3;
    public CvinferComputeBackend PreprocessBackend { get; set; } = CvinferComputeBackend.Auto;
    public CvinferComputeBackend PostprocessBackend { get; set; } = CvinferComputeBackend.Auto;
    public bool EnableTiming { get; set; }
    public string? InputName { get; set; }
    public string? OutputName { get; set; }
    public string? PrototypeName { get; set; }
}

public readonly record struct CvinferBox(float X1, float Y1, float X2, float Y2, float Score, int ClassId);
public readonly record struct CvinferDetection(CvinferBox Box);
public readonly record struct CvinferSegmentation(CvinferBox Box, int MaskWidth, int MaskHeight, int MaskBytes);
public readonly record struct CvinferKeypoint(float X, float Y, float Score);
public readonly record struct CvinferPose(CvinferBox Box, CvinferKeypoint[] Keypoints, int KeypointDimensions = 3);
public readonly record struct CvinferTiming(double PreprocessMs, double InferenceMs, double PostprocessMs, double TotalMs);

public sealed class CvinferException : Exception
{
    public int Status { get; }
    public CvinferException(int status, string message) : base($"cvinfer status={status}: {message}") => Status = status;
}
