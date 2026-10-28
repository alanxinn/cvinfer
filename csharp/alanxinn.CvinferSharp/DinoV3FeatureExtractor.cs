using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

/// <summary>Owned row-major FP32 tensor data.</summary>
public sealed class DinoV3Tensor
{
    public long[] Shape { get; }
    public float[] Values { get; }
    internal DinoV3Tensor(long[] shape, float[] values) { Shape = shape; Values = values; }
}
public sealed class DinoV3Features
{
    public float[] GlobalEmbedding { get; }
    public float[] NormalizedEmbedding { get; }
    public DinoV3Tensor PatchTokens { get; }
    public DinoV3Tensor FeatureMap { get; }
    internal DinoV3Features(DinoV3Tensor global, DinoV3Tensor normalized, DinoV3Tensor patches, DinoV3Tensor map)
    { GlobalEmbedding = global.Values; NormalizedEmbedding = normalized.Values; PatchTokens = patches; FeatureMap = map; }
}

/// <summary>
/// DINOv3 TensorRT image feature inference (ViT/ConvNeXt). Each concurrent
/// inference stream should use an independent instance. Outputs are owned
/// managed arrays and remain valid after further inference or model disposal.
/// </summary>
public sealed class DinoV3FeatureExtractor : IDisposable
{
    private IntPtr handle;
    public int InputWidth { get; }
    public int InputHeight { get; }
    public DinoV3FeatureExtractor(string enginePath)
    {
        if (string.IsNullOrWhiteSpace(enginePath)) throw new ArgumentException("DINOv3 engine path is required", nameof(enginePath));
        DinoV3Native.EnsureLoaded();
        var path = NativeMethods.StringToCoTaskMemUtf8(Path.GetFullPath(enginePath));
        try
        {
            DinoV3Native.Check(DinoV3Native.dinov3_model_create(path, out handle));
            DinoV3Native.Check(DinoV3Native.dinov3_model_input_size(handle, out var width, out var height));
            InputWidth = width; InputHeight = height;
        }
        catch { Dispose(); throw; }
        finally { Marshal.FreeCoTaskMem(path); }
    }
    public DinoV3Features InferBgr(byte[] bgr, int width, int height)
    {
        if (bgr is null) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || bgr.Length < checked(width * height * 3)) throw new ArgumentException("DINOv3 BGR buffer is too small");
        var pinned = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try { return InferBgr(pinned.AddrOfPinnedObject(), width, height); }
        finally { pinned.Free(); }
    }
    public DinoV3Features InferBgr(IntPtr bgr, int width, int height, int channels = 3)
    {
        if (bgr == IntPtr.Zero || width <= 0 || height <= 0 || channels != 3) throw new ArgumentException("DINOv3 requires a contiguous 3-channel BGR image");
        DinoV3Native.Check(DinoV3Native.dinov3_model_infer_bgr(Handle, bgr, width, height, channels, out var result));
        return ReadResult(result);
    }
    public DinoV3Features InferChw(float[] values)
    {
        if (values is null || values.Length != checked(3 * InputWidth * InputHeight)) throw new ArgumentException("DINOv3 CHW input size does not match the engine");
        DinoV3Native.Check(DinoV3Native.dinov3_model_infer_chw(Handle, values, new UIntPtr((uint)values.Length), out var result));
        return ReadResult(result);
    }
    private static DinoV3Features ReadResult(IntPtr result)
    {
        try
        {
            DinoV3Tensor Read(string name)
            {
                DinoV3Native.Check(DinoV3Native.dinov3_result_tensor_info(result, name, out var info));
                if (info.Rank < 1 || info.Rank > 8) throw new InvalidDataException("Invalid DINOv3 tensor rank");
                var shape = new long[info.Rank];
                long elements = 1;
                for (var i = 0; i < info.Rank; ++i)
                {
                    shape[i] = info.Shape[i];
                    if (shape[i] <= 0) throw new InvalidDataException("Invalid DINOv3 tensor dimension");
                    elements = checked(elements * shape[i]);
                }
                if ((ulong)elements != info.Elements.ToUInt64()) throw new InvalidDataException("DINOv3 tensor size mismatch");
                var values = new float[checked((int)elements)];
                DinoV3Native.Check(DinoV3Native.dinov3_result_copy_tensor(result, name, values, new UIntPtr((uint)values.Length), out var copied));
                if (copied.ToUInt64() != (ulong)values.Length) throw new InvalidDataException("DINOv3 tensor copy length mismatch");
                return new DinoV3Tensor(shape, values);
            }
            return new DinoV3Features(Read("global_embedding"), Read("normalized_embedding"), Read("patch_tokens"), Read("feature_map"));
        }
        finally { DinoV3Native.dinov3_result_destroy(result); }
    }
    public static double CosineSimilarity(float[] first, float[] second)
    {
        if (first is null || second is null || first.Length == 0 || first.Length != second.Length)
            throw new ArgumentException("Feature vectors must have the same nonzero length");
        double dot = 0, aa = 0, bb = 0;
        for (var i = 0; i < first.Length; ++i)
        {
            if (float.IsNaN(first[i]) || float.IsInfinity(first[i]) || float.IsNaN(second[i]) || float.IsInfinity(second[i]))
                throw new ArgumentException("Feature vectors must contain finite values");
            dot += (double)first[i] * second[i]; aa += (double)first[i] * first[i]; bb += (double)second[i] * second[i];
        }
        if (aa == 0 || bb == 0) throw new ArgumentException("Feature vectors must be nonzero");
        return Math.Max(-1, Math.Min(1, dot / Math.Sqrt(aa * bb)));
    }
    private IntPtr Handle => handle != IntPtr.Zero ? handle : throw new ObjectDisposedException(nameof(DinoV3FeatureExtractor));
    public void Dispose() { if (handle != IntPtr.Zero) { DinoV3Native.dinov3_model_destroy(handle); handle = IntPtr.Zero; } GC.SuppressFinalize(this); }
    ~DinoV3FeatureExtractor() { if (handle != IntPtr.Zero) DinoV3Native.dinov3_model_destroy(handle); }
}

internal static class DinoV3Native
{
    private const string Library = "alanxinn_cvinfer_dinov3.dll";
    private static readonly object LoadLock = new();
    private static IntPtr module;
    [StructLayout(LayoutKind.Sequential)]
    internal struct TensorInfo
    {
        public int Rank;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)] public long[] Shape;
        public UIntPtr Elements;
    }
    internal static void EnsureLoaded()
    {
        lock (LoadLock)
        {
            if (module != IntPtr.Zero) return;
            foreach (var root in new[] { Environment.GetEnvironmentVariable("ALANXINN_DINOV3_NATIVE_DIR"), AppContext.BaseDirectory,
                                         Path.GetDirectoryName(typeof(DinoV3Native).Assembly.Location) })
            {
                if (string.IsNullOrWhiteSpace(root)) continue;
                foreach (var directory in new[] { root!, Path.Combine(root!, "runtimes", "win-x64", "native") })
                {
                    var path = Path.Combine(directory, Library);
                    if (!File.Exists(path)) continue;
                    var siblingRuntime = Path.GetFullPath(Path.Combine(directory, "..", "bintrt"));
                    SetDllDirectoryW(Directory.Exists(siblingRuntime) ? siblingRuntime : directory);
                    module = LoadLibraryW(path);
                    if (module != IntPtr.Zero) return;
                }
            }
            throw new DllNotFoundException("DINOv3 native DLL/runtime is missing; build scripts/Build-Trt10.ps1 and deploy alanxinn_cvinfer_dinov3.dll with bintrt DLLs.");
        }
    }
    internal static void Check(int status)
    {
        if (status == 0) return;
        var text = dinov3_last_error(); var length = 0;
        while (Marshal.ReadByte(text, length) != 0) ++length;
        var bytes = new byte[length]; Marshal.Copy(text, bytes, 0, length);
        throw new CvinferException(status, Encoding.UTF8.GetString(bytes));
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool SetDllDirectoryW(string directory);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] private static extern IntPtr dinov3_last_error();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int dinov3_model_create(IntPtr path, out IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void dinov3_model_destroy(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int dinov3_model_input_size(IntPtr model, out int width, out int height);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int dinov3_model_infer_bgr(IntPtr model, IntPtr data, int width, int height, int channels, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int dinov3_model_infer_chw(IntPtr model, [In] float[] data, UIntPtr elements, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void dinov3_result_destroy(IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int dinov3_result_tensor_info(IntPtr result, [MarshalAs(UnmanagedType.LPStr)] string name, out TensorInfo info);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int dinov3_result_copy_tensor(IntPtr result, [MarshalAs(UnmanagedType.LPStr)] string name, [Out] float[] data, UIntPtr capacity, out UIntPtr copied);
}
