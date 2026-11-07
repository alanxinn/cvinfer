using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

/// <summary>Owned hand result in MediaPipe's 21-landmark order.</summary>
public sealed class MediaPipeHand
{
    public float[] Box { get; }
    public float PalmScore { get; }
    public float Confidence { get; }
    /// <summary>Model probability: 0=Left, 1=Right. Input is not mirrored automatically.</summary>
    public float Handedness { get; }
    public string Label => Handedness >= .5f ? "Right" : "Left";
    /// <summary>21 xyz triples; xy in image pixels, wrist-relative z in the x scale.</summary>
    public float[] Landmarks { get; }
    /// <summary>21 xyz triples in metres, rotated into the original image orientation.</summary>
    public float[] WorldLandmarks { get; }
    internal MediaPipeHand(MediaPipeHandsNative.Hand hand)
    {
        Box = hand.Box; PalmScore = hand.PalmScore; Confidence = hand.Confidence;
        Handedness = hand.Handedness; Landmarks = hand.Landmarks; WorldLandmarks = hand.WorldLandmarks;
    }
}

/// <summary>
/// MediaPipe palm detection and hand landmarks on TensorRT 10. Prepare both engines
/// with scripts/prepare_mediapipe_hands.py. Use one instance per concurrent stream.
/// </summary>
public sealed class MediaPipeHandDetector : IDisposable
{
    private IntPtr handle;
    public int MaxHands { get; }
    public MediaPipeHandDetector(string palmEngine, string handEngine, float palmThreshold = .5f,
        float handThreshold = .5f, float nmsThreshold = .3f, int maxHands = 2)
    {
        if (string.IsNullOrWhiteSpace(palmEngine) || string.IsNullOrWhiteSpace(handEngine))
            throw new ArgumentException("Both MediaPipe engine paths are required");
        foreach (var value in new[] { palmThreshold, handThreshold, nmsThreshold })
            if (float.IsNaN(value) || float.IsInfinity(value) || value < 0 || value > 1)
                throw new ArgumentOutOfRangeException(nameof(palmThreshold), "Thresholds must be finite and in [0,1]");
        if (maxHands < 1 || maxHands > 64) throw new ArgumentOutOfRangeException(nameof(maxHands));
        MediaPipeHandsNative.EnsureLoaded();
        var palm = NativeMethods.StringToCoTaskMemUtf8(Path.GetFullPath(palmEngine));
        var hand = IntPtr.Zero;
        try
        {
            hand = NativeMethods.StringToCoTaskMemUtf8(Path.GetFullPath(handEngine));
            var config = new MediaPipeHandsNative.Config
            { PalmThreshold = palmThreshold, HandThreshold = handThreshold, NmsThreshold = nmsThreshold, MaxHands = maxHands };
            MediaPipeHandsNative.Check(MediaPipeHandsNative.hands_model_create(palm, hand, ref config, out handle));
            MaxHands = maxHands;
        }
        catch { Dispose(); throw; }
        finally { Marshal.FreeCoTaskMem(palm); Marshal.FreeCoTaskMem(hand); }
    }
    public MediaPipeHand[] InferBgr(byte[] bgr, int width, int height)
    {
        if (bgr is null) throw new ArgumentNullException(nameof(bgr));
        if (width <= 0 || height <= 0 || bgr.Length < checked(width * height * 3))
            throw new ArgumentException("MediaPipe BGR buffer is too small");
        var pinned = GCHandle.Alloc(bgr, GCHandleType.Pinned);
        try { return InferBgr(pinned.AddrOfPinnedObject(), width, height); }
        finally { pinned.Free(); }
    }
    public MediaPipeHand[] InferBgr(IntPtr bgr, int width, int height, int channels = 3)
    {
        if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(MediaPipeHandDetector));
        if (bgr == IntPtr.Zero || width <= 0 || height <= 0 || channels != 3)
            throw new ArgumentException("MediaPipe requires a contiguous 3-channel BGR image");
        var stride = Marshal.SizeOf<MediaPipeHandsNative.Hand>();
        var buffer = Marshal.AllocHGlobal(checked(stride * MaxHands));
        try
        {
            MediaPipeHandsNative.Check(MediaPipeHandsNative.hands_model_infer_bgr(handle, bgr, width, height,
                channels, buffer, new UIntPtr((uint)MaxHands), out var count));
            if (count.ToUInt64() > (ulong)MaxHands) throw new InvalidDataException("Invalid MediaPipe hand count");
            var result = new MediaPipeHand[(int)count.ToUInt64()];
            for (var i = 0; i < result.Length; ++i)
                result[i] = new MediaPipeHand(Marshal.PtrToStructure<MediaPipeHandsNative.Hand>(IntPtr.Add(buffer, i * stride)));
            return result;
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }
    public void Dispose()
    {
        if (handle != IntPtr.Zero) { MediaPipeHandsNative.hands_model_destroy(handle); handle = IntPtr.Zero; }
        GC.SuppressFinalize(this);
    }
    ~MediaPipeHandDetector() { if (handle != IntPtr.Zero) MediaPipeHandsNative.hands_model_destroy(handle); }
}

internal static class MediaPipeHandsNative
{
    private const string Library = "alanxinn_cvinfer_mediapipe.dll";
    private static readonly object LoadLock = new();
    private static IntPtr module;
    [StructLayout(LayoutKind.Sequential)]
    internal struct Config
    {
        public float PalmThreshold, HandThreshold, NmsThreshold;
        public int MaxHands;
    }
    [StructLayout(LayoutKind.Sequential)]
    internal struct Hand
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)] public float[] Box;
        public float PalmScore, Confidence, Handedness;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 63)] public float[] Landmarks;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 63)] public float[] WorldLandmarks;
    }
    internal static void EnsureLoaded()
    {
        lock (LoadLock)
        {
            if (module != IntPtr.Zero) return;
            foreach (var root in new[] { Environment.GetEnvironmentVariable("ALANXINN_MEDIAPIPE_NATIVE_DIR"),
                AppContext.BaseDirectory, Path.GetDirectoryName(typeof(MediaPipeHandsNative).Assembly.Location) })
            {
                if (string.IsNullOrWhiteSpace(root)) continue;
                foreach (var directory in new[] { root!, Path.Combine(root!, "runtimes", "win-x64", "native") })
                {
                    var path = Path.Combine(directory, Library);
                    if (!File.Exists(path)) continue;
                    var runtime = Path.GetFullPath(Path.Combine(directory, "..", "bintrt"));
                    SetDllDirectoryW(Directory.Exists(runtime) ? runtime : directory);
                    module = LoadLibraryW(path);
                    if (module != IntPtr.Zero) return;
                }
            }
            throw new DllNotFoundException("Deploy alanxinn_cvinfer_mediapipe.dll and the TensorRT bintrt runtime DLLs.");
        }
    }
    internal static void Check(int status)
    {
        if (status == 0) return;
        var pointer = hands_last_error();
        var length = 0;
        while (Marshal.ReadByte(pointer, length) != 0) ++length;
        var bytes = new byte[length]; Marshal.Copy(pointer, bytes, 0, length);
        throw new CvinferException(status, Encoding.UTF8.GetString(bytes));
    }
    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)] private static extern IntPtr LoadLibraryW(string path);
    [DllImport("kernel32", CharSet = CharSet.Unicode)] private static extern bool SetDllDirectoryW(string directory);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr hands_last_error();
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int hands_model_create(
        IntPtr palm, IntPtr hand, ref Config config, out IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern void hands_model_destroy(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)] internal static extern int hands_model_infer_bgr(
        IntPtr model, IntPtr image, int width, int height, int channels, IntPtr output, UIntPtr capacity, out UIntPtr count);
}
