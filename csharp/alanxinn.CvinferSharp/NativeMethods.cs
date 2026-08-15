using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

internal static class NativeMethods
{
    private const string TensorRtLibrary = "alanxinn_cvinfer_trt.dll";
    private const string DenglinLibrary = "alanxinn_cvinfer_denglin.dll";
    private static readonly object CacheLock = new();
    private static Api? _tensorRt;
    private static Api? _denglin;

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeConfig
    {
        public int InputWidth;
        public int InputHeight;
        public int ClassCount;
        public float ConfidenceThreshold;
        public float NmsThreshold;
        public float MaskThreshold;
        public int ClassAgnosticNms;
        public IntPtr InputName;
        public IntPtr OutputName;
        public IntPtr PrototypeName;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeConfigV2
    {
        public int InputWidth;
        public int InputHeight;
        public int ClassCount;
        public float ConfidenceThreshold;
        public float NmsThreshold;
        public float MaskThreshold;
        public int ClassAgnosticNms;
        public IntPtr InputName;
        public IntPtr OutputName;
        public IntPtr PrototypeName;
        public int PreprocessBackend;
        public int PostprocessBackend;
        public int KeypointCount;
        public int KeypointDimensions;
        public int InferenceBackend;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeBox
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;
        public float Score;
        public int ClassId;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeDetection
    {
        public NativeBox Box;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeSegmentationInfo
    {
        public NativeBox Box;
        public int MaskWidth;
        public int MaskHeight;
        public UIntPtr MaskBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeKeypoint
    {
        public float X;
        public float Y;
        public float Score;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativePoseInfo
    {
        public NativeBox Box;
        public int KeypointCount;
        public int KeypointDimensions;
        public UIntPtr KeypointBytes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeTiming
    {
        public double PreprocessMs;
        public double InferenceMs;
        public double PostprocessMs;
        public double TotalMs;
    }

    internal sealed class Api
    {
        private readonly LastErrorDelegate _lastError;
        private readonly ModelConfigV2InitDelegate _modelConfigV2Init;
        private readonly ModelCreateV2Delegate _modelCreateV2;
        private readonly ModelDestroyDelegate _modelDestroy;
        private readonly ModelSetTimingDelegate _modelSetTiming;
        private readonly ModelInferBgrDelegate _modelInferBgr;
        private readonly ResultDestroyDelegate _resultDestroy;
        private readonly ResultTypeDelegate _resultType;
        private readonly ResultCountDelegate _resultCount;
        private readonly ResultGetDetectionDelegate _resultGetDetection;
        private readonly ResultGetSegmentationDelegate _resultGetSegmentation;
        private readonly ResultCopyMaskDelegate _resultCopyMask;
        private readonly ResultGetPoseDelegate _resultGetPose;
        private readonly ResultCopyKeypointsDelegate _resultCopyKeypoints;
        private readonly ResultGetTimingDelegate _resultGetTiming;

        internal Api(IntPtr module)
        {
            _lastError = Bind<LastErrorDelegate>(module, "cvinfer_last_error");
            _modelConfigV2Init = Bind<ModelConfigV2InitDelegate>(
                module, "cvinfer_model_config_v2_init");
            _modelCreateV2 = Bind<ModelCreateV2Delegate>(
                module, "cvinfer_model_create_v2");
            _modelDestroy = Bind<ModelDestroyDelegate>(module, "cvinfer_model_destroy");
            _modelSetTiming = Bind<ModelSetTimingDelegate>(
                module, "cvinfer_model_set_timing_enabled");
            _modelInferBgr = Bind<ModelInferBgrDelegate>(
                module, "cvinfer_model_infer_bgr");
            _resultDestroy = Bind<ResultDestroyDelegate>(
                module, "cvinfer_result_destroy");
            _resultType = Bind<ResultTypeDelegate>(module, "cvinfer_result_type");
            _resultCount = Bind<ResultCountDelegate>(module, "cvinfer_result_count");
            _resultGetDetection = Bind<ResultGetDetectionDelegate>(
                module, "cvinfer_result_get_detection");
            _resultGetSegmentation = Bind<ResultGetSegmentationDelegate>(
                module, "cvinfer_result_get_segmentation");
            _resultCopyMask = Bind<ResultCopyMaskDelegate>(
                module, "cvinfer_result_copy_mask");
            _resultGetPose = Bind<ResultGetPoseDelegate>(
                module, "cvinfer_result_get_pose");
            _resultCopyKeypoints = Bind<ResultCopyKeypointsDelegate>(
                module, "cvinfer_result_copy_keypoints");
            _resultGetTiming = Bind<ResultGetTimingDelegate>(
                module, "cvinfer_result_get_timing");
        }

        internal void ModelConfigV2Init(ref NativeConfigV2 config) =>
            _modelConfigV2Init(ref config);

        internal int ModelCreateV2(
            IntPtr enginePathUtf8,
            int modelType,
            ref NativeConfigV2 config,
            out IntPtr model) =>
            _modelCreateV2(enginePathUtf8, modelType, ref config, out model);

        internal void ModelDestroy(IntPtr model) => _modelDestroy(model);

        internal int ModelSetTimingEnabled(IntPtr model, int enabled) =>
            _modelSetTiming(model, enabled);

        internal int ModelInferBgr(
            IntPtr model,
            IntPtr bgr,
            int width,
            int height,
            int channels,
            out IntPtr result) =>
            _modelInferBgr(model, bgr, width, height, channels, out result);

        internal void ResultDestroy(IntPtr result) => _resultDestroy(result);
        internal int ResultType(IntPtr result) => _resultType(result);
        internal UIntPtr ResultCount(IntPtr result) => _resultCount(result);

        internal int ResultGetDetection(
            IntPtr result,
            UIntPtr index,
            out NativeDetection detection) =>
            _resultGetDetection(result, index, out detection);

        internal int ResultGetSegmentation(
            IntPtr result,
            UIntPtr index,
            out NativeSegmentationInfo segmentation) =>
            _resultGetSegmentation(result, index, out segmentation);

        internal int ResultCopyMask(
            IntPtr result,
            UIntPtr index,
            IntPtr destination,
            UIntPtr destinationBytes,
            out UIntPtr copiedBytes) =>
            _resultCopyMask(result, index, destination, destinationBytes, out copiedBytes);

        internal int ResultGetPose(
            IntPtr result,
            UIntPtr index,
            out NativePoseInfo pose) =>
            _resultGetPose(result, index, out pose);

        internal int ResultCopyKeypoints(
            IntPtr result,
            UIntPtr index,
            IntPtr destination,
            UIntPtr destinationBytes,
            out UIntPtr copiedBytes) =>
            _resultCopyKeypoints(result, index, destination, destinationBytes, out copiedBytes);

        internal int ResultGetTiming(IntPtr result, out NativeTiming timing) =>
            _resultGetTiming(result, out timing);

        internal string LastError()
        {
            var pointer = _lastError();
            return PtrToStringUtf8(pointer) ?? "Unknown cvinfer error";
        }

        internal void ThrowIfFailed(int status)
        {
            if (status != 0) throw new CvinferException(status, LastError());
        }
    }

    internal static Api Load(CvinferInferenceBackend backend)
    {
        lock (CacheLock)
        {
            if (backend == CvinferInferenceBackend.TensorRt)
                return _tensorRt ??= LoadApi(TensorRtLibrary);
            if (backend == CvinferInferenceBackend.Denglin)
                return _denglin ??= LoadApi(DenglinLibrary);
            throw new ArgumentOutOfRangeException(nameof(backend), backend,
                "Unsupported inference backend");
        }
    }

    private static Api LoadApi(string libraryName)
    {
        var attempted = new List<string>();
        var loadErrors = new List<string>();
        foreach (var candidate in CandidatePaths(libraryName))
        {
            attempted.Add(candidate);
            var nativeDirectory = Path.GetDirectoryName(candidate);
            if (!string.IsNullOrWhiteSpace(nativeDirectory))
                SetDllDirectory(nativeDirectory);
            var module = LoadLibrary(candidate);
            if (module == IntPtr.Zero)
            {
                var error = Marshal.GetLastWin32Error();
                if (error != 0)
                    loadErrors.Add($"{candidate}: {new Win32Exception(error).Message} ({error})");
                continue;
            }
            try
            {
                return new Api(module);
            }
            catch
            {
                FreeLibrary(module);
                throw;
            }
        }

        throw new DllNotFoundException(
            $"Unable to load {libraryName}. Searched: {string.Join("; ", attempted)}" +
            (loadErrors.Count == 0
                ? string.Empty
                : $". Native loader errors: {string.Join(" | ", loadErrors)}"));
    }

    private static IEnumerable<string> CandidatePaths(string libraryName)
    {
        var roots = new List<string>();
        AddRoot(roots, Path.GetDirectoryName(typeof(NativeMethods).Assembly.Location));
        AddRoot(roots, AppContext.BaseDirectory);
        AddRoot(roots, AppDomain.CurrentDomain.BaseDirectory);
        AddRoot(roots, Environment.CurrentDirectory);

        foreach (var root in roots)
        {
            yield return Path.Combine(root, "runtimes", "win-x64", "native", libraryName);
            yield return Path.Combine(root, libraryName);
        }

        // Let Windows apply its normal search path as a final compatibility
        // fallback for applications that deploy native DLLs beside the exe.
        yield return libraryName;
    }

    private static void AddRoot(List<string> roots, string? root)
    {
        if (string.IsNullOrWhiteSpace(root)) return;
        var full = Path.GetFullPath(root);
        if (!roots.Exists(value =>
                string.Equals(value, full, StringComparison.OrdinalIgnoreCase)))
            roots.Add(full);
    }

    private static T Bind<T>(IntPtr module, string name) where T : Delegate
    {
        var address = GetProcAddress(module, name);
        if (address == IntPtr.Zero)
            throw new EntryPointNotFoundException($"Native export not found: {name}");
        return (T)Marshal.GetDelegateForFunctionPointer(address, typeof(T));
    }

    internal static IntPtr StringToCoTaskMemUtf8(string? value)
    {
        if (string.IsNullOrEmpty(value)) return IntPtr.Zero;
        var bytes = Encoding.UTF8.GetBytes(value);
        var pointer = Marshal.AllocCoTaskMem(bytes.Length + 1);
        Marshal.Copy(bytes, 0, pointer, bytes.Length);
        Marshal.WriteByte(pointer, bytes.Length, 0);
        return pointer;
    }

    private static string? PtrToStringUtf8(IntPtr pointer)
    {
        if (pointer == IntPtr.Zero) return null;
        var length = 0;
        while (Marshal.ReadByte(pointer, length) != 0) length++;
        if (length == 0) return string.Empty;
        var bytes = new byte[length];
        Marshal.Copy(pointer, bytes, 0, length);
        return Encoding.UTF8.GetString(bytes);
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate IntPtr LastErrorDelegate();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void ModelConfigV2InitDelegate(ref NativeConfigV2 config);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ModelCreateV2Delegate(
        IntPtr enginePathUtf8,
        int modelType,
        ref NativeConfigV2 config,
        out IntPtr model);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void ModelDestroyDelegate(IntPtr model);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ModelSetTimingDelegate(IntPtr model, int enabled);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ModelInferBgrDelegate(
        IntPtr model,
        IntPtr bgr,
        int width,
        int height,
        int channels,
        out IntPtr result);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void ResultDestroyDelegate(IntPtr result);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultTypeDelegate(IntPtr result);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate UIntPtr ResultCountDelegate(IntPtr result);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultGetDetectionDelegate(
        IntPtr result,
        UIntPtr index,
        out NativeDetection detection);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultGetSegmentationDelegate(
        IntPtr result,
        UIntPtr index,
        out NativeSegmentationInfo segmentation);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultCopyMaskDelegate(
        IntPtr result,
        UIntPtr index,
        IntPtr destination,
        UIntPtr destinationBytes,
        out UIntPtr copiedBytes);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultGetPoseDelegate(
        IntPtr result,
        UIntPtr index,
        out NativePoseInfo pose);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultCopyKeypointsDelegate(
        IntPtr result,
        UIntPtr index,
        IntPtr destination,
        UIntPtr destinationBytes,
        out UIntPtr copiedBytes);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ResultGetTimingDelegate(
        IntPtr result,
        out NativeTiming timing);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibrary(string fileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string? pathName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr module, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr module);
}
