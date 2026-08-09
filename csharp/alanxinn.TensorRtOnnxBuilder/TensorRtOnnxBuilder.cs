using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace alanxinn.TensorRt;

public static class TensorRtOnnxBuilder
{
    public static TensorRtBuildResult Build(
        string onnxPath,
        string enginePath,
        TensorRtBuildOptions? options = null)
    {
        options ??= new TensorRtBuildOptions();
        var sourcePath = NormalizeExistingFile(onnxPath, "ONNX");
        var destinationPath = NormalizeOutputPath(enginePath);
        if (string.Equals(sourcePath, destinationPath, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("ONNX and engine paths must be different", nameof(enginePath));
        if (!Enum.IsDefined(typeof(TensorRtPrecision), options.Precision))
            throw new ArgumentOutOfRangeException(nameof(options.Precision));
        if (options.WorkspaceMiB <= 0 || options.WorkspaceMiB > long.MaxValue / (1024L * 1024L))
            throw new ArgumentOutOfRangeException(nameof(options.WorkspaceMiB), "WorkspaceMiB must be positive");
        if (options.DeviceId < -1)
            throw new ArgumentOutOfRangeException(nameof(options.DeviceId), "DeviceId must be -1 or greater");

        var profiles = options.OptimizationProfiles ?? Array.Empty<TensorRtOptimizationProfile>();
        foreach (var profile in profiles)
        {
            if (profile is null) throw new ArgumentException("Optimization profile cannot be null", nameof(options));
            ValidateProfile(profile);
        }

        // Provision the external TensorRT/CUDA runtime before the first P/Invoke.
        // The NuGet package contains only the small builder DLL; large dependencies
        // are resolved locally or downloaded from the configured GitHub Release.
        TensorRtRuntime.Initialize();

        Directory.CreateDirectory(Path.GetDirectoryName(destinationPath) ?? Environment.CurrentDirectory);
        var startedAt = DateTime.UtcNow;
        var progressContext = IntPtr.Zero;
        NativeMethods.ProgressCallback? progressCallback = null;
        var onnxPointer = IntPtr.Zero;
        var enginePointer = IntPtr.Zero;
        var nativeProfilePointers = new List<IntPtr>();
        var profileArray = IntPtr.Zero;
        try
        {
            if (options.Progress is not null)
            {
                var context = new ProgressContext(options.Progress, startedAt);
                progressContext = GCHandle.ToIntPtr(GCHandle.Alloc(context));
                progressCallback = ReportProgress;
            }
            onnxPointer = NativeMethods.StringToCoTaskMemUtf8(sourcePath);
            enginePointer = NativeMethods.StringToCoTaskMemUtf8(destinationPath);
            var nativeConfig = new NativeMethods.NativeConfig();
            NativeMethods.trt_builder_config_init(ref nativeConfig);
            nativeConfig.Precision = (int)options.Precision;
            nativeConfig.WorkspaceMiB = options.WorkspaceMiB;
            nativeConfig.DeviceId = options.DeviceId;
            nativeConfig.EnableTf32 = options.EnableTf32 ? 1 : 0;
            nativeConfig.EnableSparseWeights = options.EnableSparseWeights ? 1 : 0;
            nativeConfig.StrictTypes = options.StrictTypes ? 1 : 0;

            if (profiles.Count > 0)
            {
                var nativeProfileSize = Marshal.SizeOf(typeof(NativeMethods.NativeProfile));
                profileArray = Marshal.AllocCoTaskMem(checked(nativeProfileSize * profiles.Count));
                for (var index = 0; index < profiles.Count; index++)
                {
                    var profile = profiles[index];
                    var nativeProfile = new NativeMethods.NativeProfile
                    {
                        MinShapes = AllocateShapeString(profile.MinShapes, nativeProfilePointers),
                        OptShapes = AllocateShapeString(profile.OptShapes, nativeProfilePointers),
                        MaxShapes = AllocateShapeString(profile.MaxShapes, nativeProfilePointers)
                    };
                    Marshal.StructureToPtr(
                        nativeProfile,
                        IntPtr.Add(profileArray, checked(index * nativeProfileSize)),
                        false);
                }
            }

            int status;
            if (progressCallback is null)
            {
                // Keep compatibility with an older native builder when no
                // progress reporting was requested.
                status = NativeMethods.trt_builder_build_onnx(
                    onnxPointer,
                    enginePointer,
                    ref nativeConfig,
                    profileArray,
                    new UIntPtr((ulong)profiles.Count));
            }
            else
            {
                status = NativeMethods.trt_builder_build_onnx_with_progress(
                    onnxPointer,
                    enginePointer,
                    ref nativeConfig,
                    profileArray,
                    new UIntPtr((ulong)profiles.Count),
                    progressCallback,
                    progressContext);
            }
            if (status != 0)
                throw new TensorRtBuilderException(status, NativeMethods.LastError());
        }
        finally
        {
            foreach (var pointer in nativeProfilePointers) NativeMethods.Free(pointer);
            NativeMethods.Free(profileArray);
            NativeMethods.Free(onnxPointer);
            NativeMethods.Free(enginePointer);
            if (progressContext != IntPtr.Zero)
            {
                GCHandle.FromIntPtr(progressContext).Free();
                progressContext = IntPtr.Zero;
            }
        }

        if (!File.Exists(destinationPath))
            throw new IOException("TensorRT builder returned successfully but did not create the engine file");
        var engineSize = new FileInfo(destinationPath).Length;
        if (engineSize <= 0)
            throw new IOException("TensorRT builder created an empty engine file");
        return new TensorRtBuildResult(
            sourcePath,
            destinationPath,
            options.Precision,
            DateTime.UtcNow - startedAt,
            engineSize);
    }

    private sealed class ProgressContext
    {
        public ProgressContext(IProgress<TensorRtBuildProgress> progress, DateTime startedAt)
        {
            Progress = progress;
            StartedAt = startedAt;
        }

        public IProgress<TensorRtBuildProgress> Progress { get; }
        public DateTime StartedAt { get; }
    }

    private static void ReportProgress(
        int stage,
        int percent,
        int isHeartbeat,
        IntPtr messageUtf8,
        IntPtr userData)
    {
        try
        {
            if (userData == IntPtr.Zero) return;
            var context = (ProgressContext?)GCHandle.FromIntPtr(userData).Target;
            if (context is null) return;
            context.Progress.Report(new TensorRtBuildProgress(
                stage,
                percent,
                isHeartbeat != 0,
                NativeMethods.PtrToStringUtf8(messageUtf8) ?? string.Empty,
                DateTime.UtcNow - context.StartedAt));
        }
        catch
        {
            // A progress handler must never escape through the unmanaged callback.
        }
    }

    private static IntPtr AllocateShapeString(
        IReadOnlyDictionary<string, TensorRtShape> shapes,
        ICollection<IntPtr> allocatedPointers)
    {
        var text = string.Join(",", shapes
            .OrderBy(item => item.Key, StringComparer.Ordinal)
            .Select(item => item.Key + ":" + item.Value));
        var pointer = NativeMethods.StringToCoTaskMemUtf8(text);
        allocatedPointers.Add(pointer);
        return pointer;
    }

    private static void ValidateProfile(TensorRtOptimizationProfile profile)
    {
        var min = profile.MinShapes ?? throw new ArgumentException("minShapes cannot be null", nameof(profile));
        var opt = profile.OptShapes ?? throw new ArgumentException("optShapes cannot be null", nameof(profile));
        var max = profile.MaxShapes ?? throw new ArgumentException("maxShapes cannot be null", nameof(profile));
        if (min.Count == 0 || opt.Count == 0 || max.Count == 0)
            throw new ArgumentException("Each optimization profile must contain min, opt and max shapes", nameof(profile));
        var names = new HashSet<string>(min.Keys, StringComparer.Ordinal);
        if (!names.SetEquals(opt.Keys) || !names.SetEquals(max.Keys))
            throw new ArgumentException("min, opt and max shapes must contain the same tensor names", nameof(profile));
        foreach (var name in names)
        {
            var minShape = min[name] ?? throw new ArgumentException("min shape cannot be null", nameof(profile));
            var optShape = opt[name] ?? throw new ArgumentException("opt shape cannot be null", nameof(profile));
            var maxShape = max[name] ?? throw new ArgumentException("max shape cannot be null", nameof(profile));
            if (minShape.Dimensions.Count != optShape.Dimensions.Count ||
                minShape.Dimensions.Count != maxShape.Dimensions.Count)
                throw new ArgumentException("min, opt and max shapes must have the same rank", nameof(profile));
            for (var index = 0; index < minShape.Dimensions.Count; index++)
            {
                if (minShape.Dimensions[index] > optShape.Dimensions[index] ||
                    optShape.Dimensions[index] > maxShape.Dimensions[index])
                    throw new ArgumentException("Shape dimensions must satisfy min <= opt <= max", nameof(profile));
            }
        }
    }

    private static string NormalizeExistingFile(string path, string description)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException(description + " path is required", nameof(path));
        var fullPath = Path.GetFullPath(path);
        if (!File.Exists(fullPath))
            throw new FileNotFoundException(description + " file was not found", fullPath);
        return fullPath;
    }

    private static string NormalizeOutputPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("Engine path is required", nameof(path));
        return Path.GetFullPath(path);
    }
}
