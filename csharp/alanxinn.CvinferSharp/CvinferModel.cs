using System;
using System.IO;
using System.Runtime.InteropServices;

namespace alanxinn;

public sealed class CvinferModel : IDisposable
{
    private IntPtr _handle;
    private readonly NativeMethods.Api _native;
    public CvinferModelType ModelType { get; }
    public CvinferInferenceBackend InferenceBackend { get; }

    public CvinferModel(string enginePath, CvinferModelType modelType, CvinferConfig? config = null)
    {
        if (string.IsNullOrWhiteSpace(enginePath)) throw new ArgumentException("Engine path is required", nameof(enginePath));
        ModelType = modelType;
        InferenceBackend = config?.InferenceBackend ?? CvinferInferenceBackend.TensorRt;
        _native = NativeMethods.Load(InferenceBackend);
        var nativeConfig = new NativeMethods.NativeConfigV2();
        _native.ModelConfigV2Init(ref nativeConfig);
        var path = NativeMethods.StringToCoTaskMemUtf8(enginePath);
        var inputName = IntPtr.Zero;
        var outputName = IntPtr.Zero;
        var prototypeName = IntPtr.Zero;
        try
        {
            if (config is not null)
            {
                nativeConfig.InputWidth = config.InputWidth;
                nativeConfig.InputHeight = config.InputHeight;
                nativeConfig.ClassCount = config.ClassCount;
                nativeConfig.ConfidenceThreshold = config.ConfidenceThreshold;
                nativeConfig.NmsThreshold = config.NmsThreshold;
                nativeConfig.MaskThreshold = config.MaskThreshold;
                nativeConfig.ClassAgnosticNms = config.ClassAgnosticNms ? 1 : 0;
                nativeConfig.PreprocessBackend = (int)config.PreprocessBackend;
                nativeConfig.PostprocessBackend = (int)config.PostprocessBackend;
                nativeConfig.KeypointCount = config.KeypointCount;
                nativeConfig.KeypointDimensions = config.KeypointDimensions;
                nativeConfig.InferenceBackend = (int)config.InferenceBackend;
                inputName = MarshalUtf8(config.InputName);
                outputName = MarshalUtf8(config.OutputName);
                prototypeName = MarshalUtf8(config.PrototypeName);
                nativeConfig.InputName = inputName;
                nativeConfig.OutputName = outputName;
                nativeConfig.PrototypeName = prototypeName;
            }
            _native.ThrowIfFailed(_native.ModelCreateV2(
                path, (int)modelType, ref nativeConfig, out _handle));
            if (config?.EnableTiming == true)
            {
                try
                {
                    _native.ThrowIfFailed(_native.ModelSetTimingEnabled(_handle, 1));
                }
                catch
                {
                    _native.ModelDestroy(_handle);
                    _handle = IntPtr.Zero;
                    throw;
                }
            }
        }
        finally
        {
            Marshal.FreeCoTaskMem(path);
            Free(inputName);
            Free(outputName);
            Free(prototypeName);
        }
    }

    public CvinferResult InferBgr(byte[] bgr, int width, int height, int channels = 3)
    {
        if (bgr is null) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || channels < 3) throw new ArgumentOutOfRangeException(nameof(width));
        var required = checked(width * height * channels);
        if (bgr.Length < required) throw new ArgumentException("BGR buffer is smaller than width*height*channels", nameof(bgr));
        var pin = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try
        {
            return InferBgr(pin.AddrOfPinnedObject(), width, height, channels);
        }
        finally
        {
            pin.Free();
        }
    }

    public CvinferResult InferBgr(IntPtr bgr, int width, int height, int channels = 3)
    {
        if (bgr == IntPtr.Zero) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || channels < 3) throw new ArgumentOutOfRangeException(nameof(width));
        ThrowIfDisposed();
        _native.ThrowIfFailed(_native.ModelInferBgr(
            _handle, bgr, width, height, channels, out var result));
        return new CvinferResult(result, ModelType, _native);
    }

    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            _native.ModelDestroy(_handle);
            _handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    private static IntPtr MarshalUtf8(string? value) => NativeMethods.StringToCoTaskMemUtf8(value);
    private static void Free(IntPtr value) { if (value != IntPtr.Zero) Marshal.FreeCoTaskMem(value); }
    private void ThrowIfDisposed() { if (_handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(CvinferModel)); }
}

public sealed class CvinferResult : IDisposable
{
    private IntPtr _handle;
    private readonly NativeMethods.Api _native;
    public CvinferModelType ModelType { get; }

    internal CvinferResult(IntPtr handle, CvinferModelType type, NativeMethods.Api native)
    {
        _handle = handle;
        ModelType = type;
        _native = native;
    }

    public int Count => checked((int)_native.ResultCount(Handle).ToUInt64());

    public CvinferTiming Timing
    {
        get
        {
            _native.ThrowIfFailed(_native.ResultGetTiming(Handle, out var value));
            return new CvinferTiming(value.PreprocessMs, value.InferenceMs, value.PostprocessMs, value.TotalMs);
        }
    }

    public CvinferDetection GetDetection(int index)
    {
        _native.ThrowIfFailed(_native.ResultGetDetection(Handle, new UIntPtr((uint)index), out var value));
        return new CvinferDetection(ToBox(value.Box));
    }

    public CvinferSegmentation GetSegmentation(int index)
    {
        _native.ThrowIfFailed(_native.ResultGetSegmentation(Handle, new UIntPtr((uint)index), out var value));
        return new CvinferSegmentation(ToBox(value.Box), value.MaskWidth, value.MaskHeight, checked((int)value.MaskBytes.ToUInt64()));
    }

    public byte[] CopyMask(int index)
    {
        var info = GetSegmentation(index);
        if (info.MaskBytes == 0) return Array.Empty<byte>();
        var mask = new byte[info.MaskBytes];
        var pin = GCHandle.Alloc(mask, GCHandleType.Pinned);
        try
        {
            _native.ThrowIfFailed(_native.ResultCopyMask(
                Handle, new UIntPtr((uint)index), pin.AddrOfPinnedObject(), new UIntPtr((ulong)mask.Length), out var copied));
            if (copied.ToUInt64() != (ulong)mask.Length) Array.Resize(ref mask, checked((int)copied.ToUInt64()));
            return mask;
        }
        finally
        {
            pin.Free();
        }
    }

    public CvinferPose GetPose(int index)
    {
        _native.ThrowIfFailed(_native.ResultGetPose(
            Handle, new UIntPtr((uint)index), out var info));
        var keypoints = CopyKeypoints(index, info.KeypointBytes);
        return new CvinferPose(ToBox(info.Box), keypoints, info.KeypointDimensions);
    }

    public CvinferKeypoint[] CopyKeypoints(int index)
    {
        _native.ThrowIfFailed(_native.ResultGetPose(
            Handle, new UIntPtr((uint)index), out var info));
        return CopyKeypoints(index, info.KeypointBytes);
    }

    private CvinferKeypoint[] CopyKeypoints(int index, UIntPtr expectedBytes)
    {
        var bytes = checked((int)expectedBytes.ToUInt64());
        if (bytes == 0) return Array.Empty<CvinferKeypoint>();
        var nativeSize = Marshal.SizeOf<NativeMethods.NativeKeypoint>();
        if (bytes % nativeSize != 0) throw new InvalidDataException("Native keypoint buffer size is invalid");
        var native = new NativeMethods.NativeKeypoint[bytes / nativeSize];
        var pin = GCHandle.Alloc(native, GCHandleType.Pinned);
        try
        {
            _native.ThrowIfFailed(_native.ResultCopyKeypoints(
                Handle, new UIntPtr((uint)index), pin.AddrOfPinnedObject(), new UIntPtr((ulong)bytes), out var copied));
            var copiedBytes = checked((int)copied.ToUInt64());
            if (copiedBytes != bytes) Array.Resize(ref native, copiedBytes / nativeSize);
            var result = new CvinferKeypoint[native.Length];
            for (var i = 0; i < native.Length; ++i)
                result[i] = new CvinferKeypoint(native[i].X, native[i].Y, native[i].Score);
            return result;
        }
        finally
        {
            pin.Free();
        }
    }

    public void Dispose()
    {
        if (_handle != IntPtr.Zero)
        {
            _native.ResultDestroy(_handle);
            _handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    private IntPtr Handle => _handle == IntPtr.Zero ? throw new ObjectDisposedException(nameof(CvinferResult)) : _handle;
    private static CvinferBox ToBox(NativeMethods.NativeBox value) => new(value.X1, value.Y1, value.X2, value.Y2, value.Score, value.ClassId);
}
