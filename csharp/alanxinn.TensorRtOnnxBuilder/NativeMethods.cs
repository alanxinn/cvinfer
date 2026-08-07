using System;
using System.Runtime.InteropServices;
using System.Text;

namespace alanxinn.TensorRt;

internal static class NativeMethods
{
    private const string Library = "alanxinn_tensorrt_builder.dll";

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeConfig
    {
        public int Precision;
        public long WorkspaceMiB;
        public int DeviceId;
        public int EnableTf32;
        public int EnableSparseWeights;
        public int StrictTypes;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct NativeProfile
    {
        public IntPtr MinShapes;
        public IntPtr OptShapes;
        public IntPtr MaxShapes;
    }

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr trt_builder_last_error();

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void trt_builder_config_init(ref NativeConfig config);

    [DllImport(Library, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int trt_builder_build_onnx(
        IntPtr onnxPathUtf8,
        IntPtr enginePathUtf8,
        ref NativeConfig config,
        IntPtr profiles,
        UIntPtr profileCount);

    internal static string LastError()
    {
        var pointer = trt_builder_last_error();
        return PtrToStringUtf8(pointer) ?? "Unknown TensorRT builder error";
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

    internal static void Free(IntPtr pointer)
    {
        if (pointer != IntPtr.Zero) Marshal.FreeCoTaskMem(pointer);
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
}
