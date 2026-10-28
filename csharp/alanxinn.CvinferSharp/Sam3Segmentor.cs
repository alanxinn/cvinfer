using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

public sealed class Sam3Mask
{
    public CvinferBox Box { get; }
    public int Width { get; }
    public int Height { get; }
    /// <summary>Full original-image mask, row-major, values 0/255.</summary>
    public byte[] Pixels { get; }
    internal Sam3Mask(CvinferBox box, int width, int height, byte[] pixels)
    { Box = box; Width = width; Height = height; Pixels = pixels; }
}

/// <summary>
/// Stateful SAM3 TensorRT image segmentation. SetImage resets all visual prompts;
/// SetBox resets points. AddPoint(..., true/false) adds foreground/background
/// evidence and Predict evaluates the box and all accumulated points together.
/// Use a separate instance for each concurrent image/session.
/// </summary>
public sealed class Sam3Segmentor : IDisposable
{
    private IntPtr handle;
    public Sam3Segmentor(string engineDirectory, float confidenceThreshold = .3f,
        float maskThreshold = .5f, int maxPoints = 32)
    {
        if (string.IsNullOrWhiteSpace(engineDirectory)) throw new ArgumentException("SAM3 engine directory is required", nameof(engineDirectory));
        Sam3Native.EnsureLoaded();
        var path = NativeMethods.StringToCoTaskMemUtf8(Path.GetFullPath(engineDirectory));
        try { Sam3Native.Check(Sam3Native.sam3_model_create(path, confidenceThreshold, maskThreshold, maxPoints, out handle)); }
        finally { Marshal.FreeCoTaskMem(path); }
    }
    public void SetImage(byte[] bgr, int width, int height)
    {
        if (bgr is null) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || bgr.Length < checked(width * height * 3)) throw new ArgumentException("SAM3 BGR buffer is too small");
        var pinned = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try { Sam3Native.Check(Sam3Native.sam3_model_set_image(Handle, pinned.AddrOfPinnedObject(), width, height, 3)); }
        finally { pinned.Free(); }
    }
    public void SetBox(float x1, float y1, float x2, float y2) =>
        Sam3Native.Check(Sam3Native.sam3_model_set_box(Handle, x1, y1, x2, y2));
    public void AddPoint(float x, float y, bool positive) =>
        Sam3Native.Check(Sam3Native.sam3_model_add_point(Handle, x, y, positive ? 1 : 0));
    public void ClearPoints() => Sam3Native.Check(Sam3Native.sam3_model_clear_points(Handle));
    public Sam3Mask Predict()
    {
        Sam3Native.Check(Sam3Native.sam3_model_predict(Handle, out var result));
        return ReadResult(result)[0];
    }
    /// <summary>Native CLIP BPE for English ASCII phrases. Use PredictTokens with the official processor for Unicode.</summary>
    public IReadOnlyList<Sam3Mask> PredictText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("SAM3 text is required", nameof(text));
        var prompt = NativeMethods.StringToCoTaskMemUtf8(text);
        try
        {
            Sam3Native.Check(Sam3Native.sam3_model_predict_text(Handle, prompt, out var result));
            return ReadResult(result);
        }
        finally { Marshal.FreeCoTaskMem(prompt); }
    }
    public IReadOnlyList<Sam3Mask> PredictTokens(long[] inputIds, long[] attentionMask)
    {
        if (inputIds is null || attentionMask is null || inputIds.Length == 0 || inputIds.Length != attentionMask.Length)
            throw new ArgumentException("SAM3 tokens and attention mask must have the same nonzero length");
        Sam3Native.Check(Sam3Native.sam3_model_predict_tokens(Handle, inputIds, attentionMask, new UIntPtr((uint)inputIds.Length), out var result));
        return ReadResult(result);
    }
    private static List<Sam3Mask> ReadResult(IntPtr result)
    {
        try
        {
            var count = checked((int)Sam3Native.sam3_result_count(result).ToUInt64());
            var values = new List<Sam3Mask>(count);
            for (var i = 0; i < count; ++i)
            {
                Sam3Native.Check(Sam3Native.sam3_result_get_mask(result, new UIntPtr((uint)i), out var info));
                var bytes = checked((int)info.Bytes.ToUInt64());
                if (bytes != checked(info.Width * info.Height)) throw new InvalidDataException("SAM3 native mask dimensions do not match buffer size");
                var pixels = new byte[bytes];
                Sam3Native.Check(Sam3Native.sam3_result_copy_mask(result, new UIntPtr((uint)i), pixels, new UIntPtr((uint)bytes), out var copied));
                if (copied.ToUInt64() != (ulong)bytes) throw new InvalidDataException("SAM3 copied mask length mismatch");
                values.Add(new Sam3Mask(new CvinferBox(info.X1, info.Y1, info.X2, info.Y2, info.Score, 0), info.Width, info.Height, pixels));
            }
            return values;
        }
        finally { Sam3Native.sam3_result_destroy(result); }
    }
    private IntPtr Handle => handle != IntPtr.Zero ? handle : throw new ObjectDisposedException(nameof(Sam3Segmentor));
    public void Dispose() { if (handle != IntPtr.Zero) { Sam3Native.sam3_model_destroy(handle); handle = IntPtr.Zero; } GC.SuppressFinalize(this); }
    ~Sam3Segmentor() { if (handle != IntPtr.Zero) Sam3Native.sam3_model_destroy(handle); }
}

internal static class Sam3Native
{
    private const string Library = "alanxinn_cvinfer_sam3.dll";
    private static readonly object LoadLock = new();
    private static IntPtr module;
    [StructLayout(LayoutKind.Sequential)]
    internal struct MaskInfo
    { public float X1, Y1, X2, Y2, Score; public int Width, Height; public UIntPtr Bytes; }
    internal static void EnsureLoaded()
    {
        lock (LoadLock)
        {
            if (module != IntPtr.Zero) return;
            var configured = Environment.GetEnvironmentVariable("ALANXINN_SAM3_NATIVE_DIR");
            foreach (var root in new[] { configured, AppContext.BaseDirectory, Path.GetDirectoryName(typeof(Sam3Native).Assembly.Location) })
            {
                if (string.IsNullOrWhiteSpace(root)) continue;
                foreach (var directory in new[] { root!, Path.Combine(root!, "runtimes", "win-x64", "native") })
                {
                    var path = Path.Combine(directory, Library);
                    if (!File.Exists(path)) continue;
                    SetDllDirectoryW(directory);
                    module = LoadLibraryW(path);
                    if (module != IntPtr.Zero) return;
                }
            }
            throw new DllNotFoundException("SAM3 native DLL/runtime is missing. Build scripts/Build-Trt10.ps1 and deploy alanxinn_cvinfer_sam3.dll with bintrt DLLs.");
        }
    }
    internal static void Check(int status)
    {
        if (status == 0) return;
        var text = sam3_last_error();
        var length = 0;
        while (Marshal.ReadByte(text, length) != 0) ++length;
        var bytes = new byte[length]; Marshal.Copy(text, bytes, 0, length);
        throw new CvinferException(status, Encoding.UTF8.GetString(bytes));
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool SetDllDirectoryW(string directory);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] private static extern IntPtr sam3_last_error();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_create(IntPtr directory, float confidence, float threshold, int maxPoints, out IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void sam3_model_destroy(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_set_image(IntPtr model, IntPtr pixels, int width, int height, int channels);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_set_box(IntPtr model, float x1, float y1, float x2, float y2);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_add_point(IntPtr model, float x, float y, int positive);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_clear_points(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_predict(IntPtr model, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_predict_text(IntPtr model, IntPtr text, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_model_predict_tokens(IntPtr model, [In] long[] ids, [In] long[] attention, UIntPtr count, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void sam3_result_destroy(IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern UIntPtr sam3_result_count(IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_result_get_mask(IntPtr result, UIntPtr index, out MaskInfo info);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam3_result_copy_mask(IntPtr result, UIntPtr index, [Out] byte[] pixels, UIntPtr capacity, out UIntPtr copied);
}
