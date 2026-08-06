using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace alanxinn;

public sealed class CvinferModel : IDisposable
{
    private IntPtr _handle;
    public CvinferModelType ModelType { get; }

    public CvinferModel(string enginePath, CvinferModelType modelType, CvinferConfig? config = null)
    {
        if (string.IsNullOrWhiteSpace(enginePath)) throw new ArgumentException("Engine path is required", nameof(enginePath));
        ModelType = modelType;
        var nativeConfig = new NativeMethods.NativeConfigV2();
        NativeMethods.cvinfer_model_config_v2_init(ref nativeConfig);
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
                inputName = MarshalUtf8(config.InputName);
                outputName = MarshalUtf8(config.OutputName);
                prototypeName = MarshalUtf8(config.PrototypeName);
                nativeConfig.InputName = inputName;
                nativeConfig.OutputName = outputName;
                nativeConfig.PrototypeName = prototypeName;
            }
            NativeMethods.ThrowIfFailed(NativeMethods.cvinfer_model_create_v2(path, (int)modelType, ref nativeConfig, out _handle));
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
        ThrowIfDisposed();
        var pin = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try
        {
            NativeMethods.ThrowIfFailed(NativeMethods.cvinfer_model_infer_bgr(
                _handle, pin.AddrOfPinnedObject(), width, height, channels, out var result));
            return new CvinferResult(result, ModelType);
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
            NativeMethods.cvinfer_model_destroy(_handle);
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
    public CvinferModelType ModelType { get; }

    internal CvinferResult(IntPtr handle, CvinferModelType type)
    {
        _handle = handle;
        ModelType = type;
    }

    public int Count => checked((int)NativeMethods.cvinfer_result_count(Handle).ToUInt64());

    public CvinferTiming Timing
    {
        get
        {
            NativeMethods.ThrowIfFailed(NativeMethods.cvinfer_result_get_timing(Handle, out var value));
            return new CvinferTiming(value.PreprocessMs, value.InferenceMs, value.PostprocessMs, value.TotalMs);
        }
    }

    public CvinferDetection GetDetection(int index)
    {
        NativeMethods.ThrowIfFailed(NativeMethods.cvinfer_result_get_detection(Handle, new UIntPtr((uint)index), out var value));
        return new CvinferDetection(ToBox(value.Box));
    }

    public CvinferSegmentation GetSegmentation(int index)
    {
        NativeMethods.ThrowIfFailed(NativeMethods.cvinfer_result_get_segmentation(Handle, new UIntPtr((uint)index), out var value));
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
            NativeMethods.ThrowIfFailed(NativeMethods.cvinfer_result_copy_mask(
                Handle, new UIntPtr((uint)index), pin.AddrOfPinnedObject(), new UIntPtr((ulong)mask.Length), out var copied));
            if (copied.ToUInt64() != (ulong)mask.Length) Array.Resize(ref mask, checked((int)copied.ToUInt64()));
            return mask;
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
            NativeMethods.cvinfer_result_destroy(_handle);
            _handle = IntPtr.Zero;
        }
        GC.SuppressFinalize(this);
    }

    private IntPtr Handle => _handle == IntPtr.Zero ? throw new ObjectDisposedException(nameof(CvinferResult)) : _handle;
    private static CvinferBox ToBox(NativeMethods.NativeBox value) => new(value.X1, value.Y1, value.X2, value.Y2, value.Score, value.ClassId);
}
