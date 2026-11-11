using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

public enum WeVisOneTask
{
    Detection, Referring, Point, CountBox, CountPoint, Ocr, OcrGrounding,
    Segmentation, Matting, VisualPrompt, Freeform
}

[StructLayout(LayoutKind.Sequential)]
public readonly struct WeVisOneBox
{
    public readonly float X1, Y1, X2, Y2;
    public WeVisOneBox(float x1, float y1, float x2, float y2) { X1 = x1; Y1 = y1; X2 = x2; Y2 = y2; }
}

public sealed class WeVisOneOptions
{
    public int MaxNewTokens { get; set; } = 1024;
    public float Temperature { get; set; }
    public float TopP { get; set; } = .9f;
    public int TopK { get; set; } = 50;
    public float RepetitionPenalty { get; set; } = 1;
    public float MaskThreshold { get; set; } = .5f;
    public ulong Seed { get; set; }
    public bool DecodePixels { get; set; } = true;
}

public sealed class WeVisOneAnnotation
{
    /// <summary>UTF-8 category, referring phrase or OCR text.</summary>
    public string Label { get; }
    public WeVisOneBox Box { get; }
    public float X { get; }
    public float Y { get; }
    public bool IsPoint { get; }
    /// <summary>-1 if this annotation has no decoded pixel output.</summary>
    public int MaskIndex { get; }
    internal WeVisOneAnnotation(string label, WeVisOneNative.Annotation value)
    { Label = label; Box = value.Box; X = value.X; Y = value.Y; IsPoint = value.IsPoint != 0; MaskIndex = value.MaskIndex; }
}

public sealed class WeVisOneMask
{
    public WeVisOneBox Box { get; }
    public int Width { get; }
    public int Height { get; }
    /// <summary>Owned row-major original-image continuous alpha in [0,1].</summary>
    public float[] Alpha { get; }
    /// <summary>0/255 for Segmentation; empty for Matting.</summary>
    public byte[] Binary { get; }
    internal WeVisOneMask(WeVisOneNative.MaskInfo info, float[] alpha, byte[] binary)
    { Box = info.Box; Width = info.Width; Height = info.Height; Alpha = alpha; Binary = binary; }
}

public sealed class WeVisOneResult
{
    public string Prompt { get; }
    public string Response { get; }
    public IReadOnlyList<WeVisOneAnnotation> Annotations { get; }
    public IReadOnlyList<WeVisOneMask> Masks { get; }
    public IReadOnlyList<string> AbsentLabels { get; }
    /// <summary>-1 for non-count tasks and incomplete or malformed count output.</summary>
    public int Count { get; }
    public bool Rejected { get; }
    public bool Truncated { get; }
    public double LatencyMilliseconds { get; }
    internal WeVisOneResult(string prompt, string response, WeVisOneAnnotation[] annotations,
        WeVisOneMask[] masks, string[] absent, WeVisOneNative.Summary summary)
    {
        Prompt = prompt; Response = response; Annotations = annotations; Masks = masks; AbsentLabels = absent;
        Count = summary.Count; Rejected = summary.Rejected != 0; Truncated = summary.Truncated != 0;
        LatencyMilliseconds = summary.LatencyMs;
    }
}

/// <summary>
/// Native TensorRT WeVisOne-VL (2B/4B), without a Python runtime. SetImage caches
/// visual features across queries. Use one instance per concurrent caller.
/// Results own their arrays and remain valid after model disposal.
/// </summary>
public sealed class WeVisOneModel : IDisposable
{
    private IntPtr handle;
    private static readonly string[] TaskNames = { "detection", "referring", "point", "count_box", "count_point",
        "ocr", "ocr_grounding", "segmentation", "matting", "visual_prompt", "freeform" };

    public WeVisOneModel(string engineDirectory)
    {
        if (string.IsNullOrWhiteSpace(engineDirectory)) throw new ArgumentException("WeVisOne engine directory is required", nameof(engineDirectory));
        WeVisOneNative.EnsureLoaded();
        var path = NativeMethods.StringToCoTaskMemUtf8(Path.GetFullPath(engineDirectory));
        try { WeVisOneNative.Check(WeVisOneNative.wevisone_model_create(path, out handle)); }
        finally { Marshal.FreeCoTaskMem(path); }
    }

    public void SetImage(byte[] bgr, int width, int height)
    {
        if (bgr is null) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || bgr.Length < checked(width * height * 3)) throw new ArgumentException("WeVisOne BGR buffer is too small");
        var pinned = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try { SetImage(pinned.AddrOfPinnedObject(), width, height); }
        finally { pinned.Free(); }
    }

    public void SetImage(IntPtr bgr, int width, int height, int channels = 3)
    {
        if (bgr == IntPtr.Zero || width <= 0 || height <= 0 || channels != 3)
            throw new ArgumentException("WeVisOne requires contiguous 3-channel BGR data");
        WeVisOneNative.Check(WeVisOneNative.wevisone_model_set_image(Handle, bgr, width, height, channels));
    }

    public WeVisOneResult Predict(WeVisOneTask task, string query = "", WeVisOneOptions? options = null,
        WeVisOneBox[]? referenceBoxes = null, string? prompt = null)
    {
        var index = (int)task;
        if (index < 0 || index >= TaskNames.Length) throw new ArgumentOutOfRangeException(nameof(task));
        var config = Configuration(options);
        var q = NativeMethods.StringToCoTaskMemUtf8(query ?? "");
        var p = IntPtr.Zero;
        try
        {
            p = NativeMethods.StringToCoTaskMemUtf8(prompt ?? "");
            WeVisOneNative.Check(WeVisOneNative.wevisone_model_predict(Handle, TaskNames[index], q, p,
                referenceBoxes, Size(referenceBoxes?.Length ?? 0), ref config, out var result));
            return ReadResult(result);
        }
        finally { Marshal.FreeCoTaskMem(q); if (p != IntPtr.Zero) Marshal.FreeCoTaskMem(p); }
    }

    /// <summary>Pass a category list without comma-splitting referring expressions.</summary>
    public WeVisOneResult Detect(string[] categories, WeVisOneOptions? options = null)
    {
        if (categories is null || categories.Length == 0) throw new ArgumentException("Detection categories are required", nameof(categories));
        return Predict(WeVisOneTask.Detection, string.Join("\n", categories), options);
    }

    public WeVisOneResult PredictBgr(byte[] bgr, int width, int height, WeVisOneTask task,
        string query = "", WeVisOneOptions? options = null, WeVisOneBox[]? referenceBoxes = null, string? prompt = null)
    { SetImage(bgr, width, height); return Predict(task, query, options, referenceBoxes, prompt); }

    public WeVisOneResult MatteBoxes(WeVisOneBox[] boxes, WeVisOneOptions? options = null)
    {
        if (boxes is null || boxes.Length == 0) throw new ArgumentException("MatteBoxes requires nonempty boxes", nameof(boxes));
        var config = Configuration(options);
        WeVisOneNative.Check(WeVisOneNative.wevisone_model_matte_boxes(Handle, boxes, Size(boxes.Length), ref config, out var result));
        return ReadResult(result);
    }

    private static WeVisOneNative.Options Configuration(WeVisOneOptions? value)
    {
        var o = value ?? new WeVisOneOptions();
        return new WeVisOneNative.Options { StructSize = checked((uint)Marshal.SizeOf<WeVisOneNative.Options>()),
            MaxNewTokens = o.MaxNewTokens, Temperature = o.Temperature, TopP = o.TopP, TopK = o.TopK,
            RepetitionPenalty = o.RepetitionPenalty, MaskThreshold = o.MaskThreshold, Seed = o.Seed,
            DecodePixels = o.DecodePixels ? 1 : 0 };
    }
    private static UIntPtr Size(int n) => new UIntPtr(checked((uint)n));

    private static WeVisOneResult ReadResult(IntPtr result)
    {
        try
        {
            WeVisOneNative.Check(WeVisOneNative.wevisone_result_summary(result, out var summary));
            string Text(int field, int index = 0)
            {
                WeVisOneNative.Check(WeVisOneNative.wevisone_result_copy_text(result, field, Size(index), null, UIntPtr.Zero, out var needed));
                var count = checked((int)needed.ToUInt64());
                if (count < 1) throw new InvalidDataException("Invalid WeVisOne string size");
                var bytes = new byte[count];
                WeVisOneNative.Check(WeVisOneNative.wevisone_result_copy_text(result, field, Size(index), bytes, Size(count), out var copied));
                if (copied != needed || bytes[count - 1] != 0) throw new InvalidDataException("WeVisOne string copy mismatch");
                return Encoding.UTF8.GetString(bytes, 0, count - 1);
            }
            var annotations = new WeVisOneAnnotation[checked((int)summary.Annotations.ToUInt64())];
            var masks = new WeVisOneMask[checked((int)summary.Masks.ToUInt64())];
            var absent = new string[checked((int)summary.AbsentLabels.ToUInt64())];
            for (var i = 0; i < annotations.Length; ++i)
            {
                WeVisOneNative.Check(WeVisOneNative.wevisone_result_annotation(result, Size(i), out var info));
                if (info.MaskIndex < -1 || info.MaskIndex >= masks.Length) throw new InvalidDataException("Invalid WeVisOne mask index");
                annotations[i] = new WeVisOneAnnotation(Text(2, i), info);
            }
            for (var i = 0; i < masks.Length; ++i)
            {
                WeVisOneNative.Check(WeVisOneNative.wevisone_result_mask_info(result, Size(i), out var info));
                var n = checked(info.Width * info.Height);
                if (info.Width <= 0 || info.Height <= 0 || info.AlphaElements.ToUInt64() != (ulong)n ||
                    (info.BinaryBytes != UIntPtr.Zero && info.BinaryBytes.ToUInt64() != (ulong)n))
                    throw new InvalidDataException("Invalid WeVisOne pixel dimensions");
                var alpha = new float[n];
                var binary = new byte[checked((int)info.BinaryBytes.ToUInt64())];
                WeVisOneNative.Check(WeVisOneNative.wevisone_result_copy_alpha(result, Size(i), alpha, Size(n), out var copied));
                if (copied != info.AlphaElements) throw new InvalidDataException("WeVisOne alpha copy mismatch");
                if (binary.Length > 0)
                {
                    WeVisOneNative.Check(WeVisOneNative.wevisone_result_copy_binary(result, Size(i), binary, Size(binary.Length), out copied));
                    if (copied != info.BinaryBytes) throw new InvalidDataException("WeVisOne binary copy mismatch");
                }
                masks[i] = new WeVisOneMask(info, alpha, binary);
            }
            for (var i = 0; i < absent.Length; ++i) absent[i] = Text(3, i);
            return new WeVisOneResult(Text(1), Text(0), annotations, masks, absent, summary);
        }
        finally { WeVisOneNative.wevisone_result_destroy(result); }
    }
    private IntPtr Handle => handle != IntPtr.Zero ? handle : throw new ObjectDisposedException(nameof(WeVisOneModel));
    public void Dispose() { if (handle != IntPtr.Zero) { WeVisOneNative.wevisone_model_destroy(handle); handle = IntPtr.Zero; } GC.SuppressFinalize(this); }
    ~WeVisOneModel() { if (handle != IntPtr.Zero) WeVisOneNative.wevisone_model_destroy(handle); }
}

internal static class WeVisOneNative
{
    private const string Library = "alanxinn_cvinfer_wevisone.dll";
    private static readonly object LoadLock = new();
    private static IntPtr module;
    [StructLayout(LayoutKind.Sequential)]
    internal struct Options
    {
        public uint StructSize;
        public int MaxNewTokens;
        public float Temperature, TopP;
        public int TopK;
        public float RepetitionPenalty, MaskThreshold;
        public ulong Seed;
        public int DecodePixels;
    }
    [StructLayout(LayoutKind.Sequential)]
    internal struct Annotation { public WeVisOneBox Box; public float X, Y; public int IsPoint, MaskIndex; }
    [StructLayout(LayoutKind.Sequential)]
    internal struct MaskInfo { public WeVisOneBox Box; public int Width, Height; public UIntPtr AlphaElements, BinaryBytes; }
    [StructLayout(LayoutKind.Sequential)]
    internal struct Summary
    { public int Count, Rejected, Truncated; public double LatencyMs; public UIntPtr Annotations, Masks, AbsentLabels; }
    internal static void EnsureLoaded()
    {
        lock (LoadLock)
        {
            if (module != IntPtr.Zero) return;
            foreach (var root in new[] { Environment.GetEnvironmentVariable("ALANXINN_WEVISONE_NATIVE_DIR"),
                AppContext.BaseDirectory, Path.GetDirectoryName(typeof(WeVisOneNative).Assembly.Location) })
            {
                if (string.IsNullOrWhiteSpace(root)) continue;
                foreach (var directory in new[] { root!, Path.Combine(root!, "runtimes", "win-x64", "native") })
                {
                    var path = Path.Combine(directory, Library);
                    if (!File.Exists(path)) continue;
                    var sibling = Path.GetFullPath(Path.Combine(directory, "..", "bintrt"));
                    SetDllDirectoryW(Directory.Exists(sibling) ? sibling : directory);
                    module = LoadLibraryW(path);
                    if (module != IntPtr.Zero) return;
                }
            }
            throw new DllNotFoundException("WeVisOne native DLL/runtime is missing; deploy alanxinn_cvinfer_wevisone.dll with the TensorRT10 runtime.");
        }
    }
    internal static void Check(int status)
    {
        if (status == 0) return;
        var text = wevisone_last_error(); var length = 0;
        while (Marshal.ReadByte(text, length) != 0) ++length;
        var bytes = new byte[length]; Marshal.Copy(text, bytes, 0, length);
        throw new CvinferException(status, Encoding.UTF8.GetString(bytes));
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] private static extern bool SetDllDirectoryW(string directory);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] private static extern IntPtr wevisone_last_error();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_model_create(IntPtr path, out IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void wevisone_model_destroy(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_model_set_image(IntPtr model, IntPtr data, int width, int height, int channels);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_model_predict(IntPtr model,
        [MarshalAs(UnmanagedType.LPStr)] string task, IntPtr query, IntPtr prompt, [In] WeVisOneBox[]? boxes, UIntPtr n, ref Options options, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_model_matte_boxes(IntPtr model,
        [In] WeVisOneBox[] boxes, UIntPtr n, ref Options options, out IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void wevisone_result_destroy(IntPtr result);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_result_summary(IntPtr result, out Summary summary);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_result_copy_text(IntPtr result,
        int field, UIntPtr index, [Out] byte[]? data, UIntPtr capacity, out UIntPtr needed);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_result_annotation(IntPtr result, UIntPtr index, out Annotation annotation);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_result_mask_info(IntPtr result, UIntPtr index, out MaskInfo info);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_result_copy_alpha(IntPtr result,
        UIntPtr index, [Out] float[] data, UIntPtr capacity, out UIntPtr needed);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int wevisone_result_copy_binary(IntPtr result,
        UIntPtr index, [Out] byte[] data, UIntPtr capacity, out UIntPtr needed);
}
