using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

public sealed class Sam2Mask
{
    public CvinferBox Box { get; }
    public int Width { get; }
    public int Height { get; }
    /// <summary>Full original-image mask, row-major, values 0/255.</summary>
    public byte[] Pixels { get; }
    internal Sam2Mask(CvinferBox box, int width, int height, byte[] pixels)
    { Box = box; Width = width; Height = height; Pixels = pixels; }
}

/// <summary>
/// Stateful SAM2.1 TensorRT image segmentation. SetImage resets all visual prompts;
/// SetBox resets points. AddPoint(..., true/false) adds foreground/background
/// evidence and Predict evaluates the box and all accumulated points together.
/// Point-only prompting is supported. Clear/remove operations discard previous-mask feedback.
/// Use a separate instance for each concurrent image/session.
/// </summary>
public sealed class Sam2Segmentor : IDisposable
{
    private IntPtr handle;
    public Sam2Segmentor(string engineDirectory, float maskThreshold = .5f, int maxPoints = 32)
    {
        if (string.IsNullOrWhiteSpace(engineDirectory)) throw new ArgumentException("SAM2 engine directory is required", nameof(engineDirectory));
        Sam2Native.EnsureLoaded();
        var path = NativeMethods.StringToCoTaskMemUtf8(Path.GetFullPath(engineDirectory));
        try { Sam2Native.Check(Sam2Native.sam2_model_create(path, maskThreshold, maxPoints, out handle)); }
        finally { Marshal.FreeCoTaskMem(path); }
    }
    public void SetImage(byte[] bgr, int width, int height)
    {
        if (bgr is null) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || bgr.Length < checked(width * height * 3)) throw new ArgumentException("SAM2 BGR buffer is too small");
        var pinned = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try { Sam2Native.Check(Sam2Native.sam2_model_set_image(Handle, pinned.AddrOfPinnedObject(), width, height, 3)); }
        finally { pinned.Free(); }
    }
    public void SetBox(float x1, float y1, float x2, float y2) =>
        Sam2Native.Check(Sam2Native.sam2_model_set_box(Handle, x1, y1, x2, y2));
    public void AddPoint(float x, float y, bool positive) =>
        Sam2Native.Check(Sam2Native.sam2_model_add_point(Handle, x, y, positive ? 1 : 0));
    public void ClearPoints() => Sam2Native.Check(Sam2Native.sam2_model_clear_points(Handle));
    public void ClearBox() => Sam2Native.Check(Sam2Native.sam2_model_clear_box(Handle));
    public void ClearPrompts() => Sam2Native.Check(Sam2Native.sam2_model_clear_prompts(Handle));
    public void RemovePoint(int index)
    {
        if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
        Sam2Native.Check(Sam2Native.sam2_model_remove_point(Handle, new UIntPtr((uint)index)));
    }
    /// <summary>Best of three masks. Pass true to refine using the last successful mask logits.</summary>
    public Sam2Mask Predict(bool usePreviousMask = false)
    {
        Sam2Native.Check(Sam2Native.sam2_model_predict(Handle, usePreviousMask ? 1 : 0, out var result));
        return ReadResult(result)[0];
    }
    private static List<Sam2Mask> ReadResult(IntPtr result)
    {
        try
        {
            var count = checked((int)Sam2Native.sam2_result_count(result).ToUInt64());
            var values = new List<Sam2Mask>(count);
            for (var i = 0; i < count; ++i)
            {
                Sam2Native.Check(Sam2Native.sam2_result_get_mask(result, new UIntPtr((uint)i), out var info));
                var bytes = checked((int)info.Bytes.ToUInt64());
                if (bytes != checked(info.Width * info.Height)) throw new InvalidDataException("SAM2 native mask dimensions do not match buffer size");
                var pixels = new byte[bytes];
                Sam2Native.Check(Sam2Native.sam2_result_copy_mask(result, new UIntPtr((uint)i), pixels, new UIntPtr((uint)bytes), out var copied));
                if (copied.ToUInt64() != (ulong)bytes) throw new InvalidDataException("SAM2 copied mask length mismatch");
                values.Add(new Sam2Mask(new CvinferBox(info.X1, info.Y1, info.X2, info.Y2, info.Score, 0), info.Width, info.Height, pixels));
            }
            return values;
        }
        finally { Sam2Native.sam2_result_destroy(result); }
    }
    private IntPtr Handle => handle != IntPtr.Zero ? handle : throw new ObjectDisposedException(nameof(Sam2Segmentor));
    public void Dispose() { if (handle != IntPtr.Zero) { Sam2Native.sam2_model_destroy(handle); handle = IntPtr.Zero; } GC.SuppressFinalize(this); }
    ~Sam2Segmentor() { if (handle != IntPtr.Zero) Sam2Native.sam2_model_destroy(handle); }
}

internal static class Sam2Native
{
    private const string Library = "alanxinn_cvinfer_sam2.dll";
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
            var configured = Environment.GetEnvironmentVariable("ALANXINN_SAM2_NATIVE_DIR");
            foreach (var root in new[] { configured, AppContext.BaseDirectory, Path.GetDirectoryName(typeof(Sam2Native).Assembly.Location) })
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
            throw new DllNotFoundException("SAM2 native DLL/runtime is missing. Build scripts/Build-Trt10.ps1 and deploy alanxinn_cvinfer_sam2.dll with bintrt DLLs.");
        }
    }
    internal static void Check(int status)
    {
        if (status == 0) return;
        var text = sam2_last_error();
        var length = 0;
        while (Marshal.ReadByte(text, length) != 0) ++length;
        var bytes = new byte[length]; Marshal.Copy(text, bytes, 0, length);
        throw new CvinferException(status, Encoding.UTF8.GetString(bytes));
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool SetDllDirectoryW(string directory);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] private static extern IntPtr sam2_last_error();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_create(IntPtr directory, float threshold, int maxPoints, out IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void sam2_model_destroy(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_set_image(IntPtr model, IntPtr pixels, int width, int height, int channels);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_set_box(IntPtr model, float x1, float y1, float x2, float y2);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_add_point(IntPtr model, float x, float y, int positive);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_clear_points(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_clear_box(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_clear_prompts(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_remove_point(IntPtr model, UIntPtr index);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_model_predict(IntPtr model, int usePreviousMask, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void sam2_result_destroy(IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern UIntPtr sam2_result_count(IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_result_get_mask(IntPtr result, UIntPtr index, out MaskInfo info);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int sam2_result_copy_mask(IntPtr result, UIntPtr index, [Out] byte[] pixels, UIntPtr capacity, out UIntPtr copied);
}
