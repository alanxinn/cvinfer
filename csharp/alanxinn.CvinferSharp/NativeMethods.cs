using System;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn;

internal static class NativeMethods
{
    private const string Library = "alanxinn_cvinfer.dll";

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
    internal struct NativeTiming
    {
        public double PreprocessMs;
        public double InferenceMs;
        public double PostprocessMs;
        public double TotalMs;
    }

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr cvinfer_last_error();

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void cvinfer_model_config_init(ref NativeConfig config);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void cvinfer_model_config_v2_init(ref NativeConfigV2 config);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_model_create(
        IntPtr enginePathUtf8, int modelType, ref NativeConfig config, out IntPtr model);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void cvinfer_model_destroy(IntPtr model);
    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_model_create_v2(
        IntPtr enginePathUtf8, int modelType, ref NativeConfigV2 config, out IntPtr model);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_model_infer_bgr(
        IntPtr model, IntPtr bgr, int width, int height, int channels, out IntPtr result);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void cvinfer_result_destroy(IntPtr result);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_result_type(IntPtr result);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern UIntPtr cvinfer_result_count(IntPtr result);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_result_get_detection(
        IntPtr result, UIntPtr index, out NativeDetection detection);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_result_get_segmentation(
        IntPtr result, UIntPtr index, out NativeSegmentationInfo segmentation);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_result_copy_mask(
        IntPtr result, UIntPtr index, IntPtr destination, UIntPtr destinationBytes, out UIntPtr copiedBytes);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int cvinfer_result_get_timing(IntPtr result, out NativeTiming timing);

    internal static string LastError()
    {
        var pointer = cvinfer_last_error();
        return PtrToStringUtf8(pointer) ?? "Unknown cvinfer error";
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

    internal static void ThrowIfFailed(int status)
    {
        if (status != 0) throw new CvinferException(status, LastError());
    }
}
