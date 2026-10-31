using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace alanxinn.TensorRt;

/// <summary>
/// Locates or downloads the large TensorRT/CUDA runtime used by the native builder.
/// </summary>
public static partial class TensorRtRuntime
{
    private const string PackageVersion = "0.2.1-trt10";
    private const string RuntimeArchiveName = "tensorrt10.16.1.11-cuda12.9-runtime-win-x64.zip";
    // No TRT 10 release archive is published by this branch. Never fall back
    // to the incompatible TRT 8 archive. A deployment can configure a mirror.
    private const string DefaultDownloadUrl = "";

    private static readonly string[] RequiredFiles =
    {
        "nvinfer_10.dll",
        "nvinfer_plugin_10.dll",
        "nvonnxparser_10.dll",
        "nvinfer_builder_resource_ptx_10.dll",
        "cudart64_12.dll",
        "nvinfer_builder_resource_sm86_10.dll",
        "nvinfer_builder_resource_sm89_10.dll",
        "nvinfer_builder_resource_sm120_10.dll"
    };
    private static readonly string[] OptionalFiles =
    {
        "cudnn64_9.dll", "cudnn_graph64_9.dll", "cublas64_12.dll", "cublasLt64_12.dll",
        "nvrtc64_120_0.dll", "nvrtc-builtins64_129.dll"
    };

    private static readonly HttpClient HttpClient = CreateHttpClient();
    private static readonly SemaphoreSlim InitializeLock = new SemaphoreSlim(1, 1);
    private static string? initializedDirectory;

    /// <summary>Optional TensorRT 10 runtime ZIP URL for a deployment mirror.</summary>
    public static string DownloadUrl { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_RUNTIME_URL") ?? DefaultDownloadUrl;

    /// <summary>Optional SHA256 of the ZIP archive, written as hexadecimal text.</summary>
    public static string? ExpectedSha256 { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_RUNTIME_SHA256");

    /// <summary>Optional cache directory. The default is under the current user's LocalApplicationData.</summary>
    public static string? CacheDirectory { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_RUNTIME_CACHE");

    /// <summary>Set true to require the runtime to be provisioned by the application.</summary>
    public static bool DisableAutoDownload { get; set; } =
        string.Equals(Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_NO_DOWNLOAD"), "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_NO_DOWNLOAD"), "true", StringComparison.OrdinalIgnoreCase);

    /// <summary>Include cuDNN/cuBLAS/NVRTC when a custom plugin or deprecated layer needs them.</summary>
    public static bool IncludeOptionalLibraries { get; set; } =
        string.Equals(Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_OPTIONAL_LIBRARIES"), "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_OPTIONAL_LIBRARIES"), "true", StringComparison.OrdinalIgnoreCase);

    public static bool IsInitialized => initializedDirectory != null;

    public static string? InitializedDirectory => initializedDirectory;

    /// <summary>
    /// Ensures all TensorRT/CUDA DLLs exist and configures the Windows DLL search path.
    /// </summary>
    public static string Initialize()
    {
        return InitializeAsync(CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>
    /// Ensures all TensorRT/CUDA DLLs exist and configures the Windows DLL search path.
    /// </summary>
    public static async Task<string> InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            throw new PlatformNotSupportedException("TensorRT ONNX builder requires Windows x64.");
        if (!Environment.Is64BitProcess)
            throw new PlatformNotSupportedException("TensorRT ONNX builder requires a 64-bit process.");
        if (initializedDirectory != null)
            return initializedDirectory;

        await InitializeLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (initializedDirectory != null)
                return initializedDirectory;

            var localDirectory = StageBundledRuntime();
            if (localDirectory == null && Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_REQUIRE_BUNDLED") == "1")
                throw new DllNotFoundException("TensorRT runtime must be restored from the NuGet dependency packages.");
            localDirectory = localDirectory ?? FindLocalRuntimeDirectory() ?? StageInstalledRuntime();
            if (localDirectory == null)
            {
                if (DisableAutoDownload)
                    throw new DllNotFoundException(
                        "TensorRT runtime DLLs are missing and automatic download is disabled. " +
                        "Provision the runtime DLLs manually or set TensorRtRuntime.DisableAutoDownload to false.");
                localDirectory = await DownloadRuntimeAsync(cancellationToken).ConfigureAwait(false);
            }

            SetDllDirectory(localDirectory);
            // The inference wrapper also calls SetDllDirectory. Keep builder
            // resource/optional-library discovery valid when both APIs share
            // one process and the most recently selected DLL directory changes.
            var processPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            if (!processPath.Split(Path.PathSeparator).Contains(localDirectory, StringComparer.OrdinalIgnoreCase))
                Environment.SetEnvironmentVariable("PATH", localDirectory + Path.PathSeparator + processPath);
            initializedDirectory = localDirectory;
            return localDirectory;
        }
        finally
        {
            InitializeLock.Release();
        }
    }

    private static string? FindLocalRuntimeDirectory()
    {
        var candidates = new List<string>();
        AddCandidate(candidates, Environment.GetEnvironmentVariable("ALANXINN_TENSORRT_RUNTIME_DIR"));
        // Also discover the runtime staged by a local CMake build.
        var parent = new DirectoryInfo(AppContext.BaseDirectory);
        for (var depth = 0; parent != null && depth < 8; ++depth, parent = parent.Parent) {
            AddCandidate(candidates, Path.Combine(parent.FullName, "builder_runtime"));
            AddCandidate(candidates, Path.Combine(parent.FullName, "build", "trt10-release", "builder_runtime"));
        }
        AddCandidate(candidates, Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native"));
        AddCandidate(candidates, AppContext.BaseDirectory);
        var assemblyDirectory = Path.GetDirectoryName(typeof(TensorRtRuntime).GetTypeInfo().Assembly.Location);
        if (!string.IsNullOrWhiteSpace(assemblyDirectory))
            AddCandidate(candidates, Path.Combine(assemblyDirectory, "runtimes", "win-x64", "native"));

        var configuredCache = GetCacheDirectory();
        AddCandidate(candidates, configuredCache);

        return candidates.FirstOrDefault(ContainsRequiredRuntime);
    }

    private static string? StageInstalledRuntime()
    {
        var trt = Environment.GetEnvironmentVariable("TENSORRT_ROOT") ??
            @"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.16.1.11-cuda-12";
        var cuda = Environment.GetEnvironmentVariable("CUDA_PATH") ??
            @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9";
        var cudnn = Environment.GetEnvironmentVariable("CUDNN_ROOT") ?? @"C:\Program Files\NVIDIA\CUDNN\v9.27";
        var trtBin = Path.Combine(trt, "bin");
        var cudaBin = Path.Combine(cuda, "bin");
        if (!File.Exists(Path.Combine(trtBin, "nvinfer_10.dll")) || !File.Exists(Path.Combine(cudaBin, "cudart64_12.dll")))
            return null;
        var cache = GetCacheDirectory();
        Directory.CreateDirectory(cache);
        foreach (var pattern in new[] { "nvinfer_10.dll", "nvinfer_plugin_10.dll", "nvonnxparser_10.dll", "nvinfer_builder_resource*_10.dll" })
            foreach (var source in Directory.GetFiles(trtBin, pattern))
                File.Copy(source, Path.Combine(cache, Path.GetFileName(source)), true);
        var cudaPatterns = IncludeOptionalLibraries
            ? new[] { "cudart64_12.dll", "cublas*64_12.dll", "nvrtc*.dll" }
            : new[] { "cudart64_12.dll" };
        foreach (var pattern in cudaPatterns)
            foreach (var source in Directory.GetFiles(cudaBin, pattern))
                File.Copy(source, Path.Combine(cache, Path.GetFileName(source)), true);
        var cudnnBin = Path.Combine(cudnn, "bin", "12.9", "x64");
        if (IncludeOptionalLibraries && Directory.Exists(cudnnBin))
            foreach (var source in Directory.GetFiles(cudnnBin, "cudnn*64_9.dll"))
                File.Copy(source, Path.Combine(cache, Path.GetFileName(source)), true);
        return ContainsRequiredRuntime(cache) ? cache : null;
    }

    private static async Task<string> DownloadRuntimeAsync(CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(DownloadUrl))
            throw new InvalidOperationException("No TensorRT 10 runtime was found. Build with scripts/Build-Trt10.ps1, set ALANXINN_TENSORRT_RUNTIME_DIR, TENSORRT_ROOT/CUDA_PATH, or configure TensorRtRuntime.DownloadUrl for your TRT 10 runtime ZIP.");

        var cacheDirectory = GetCacheDirectory();
        if (ContainsRequiredRuntime(cacheDirectory))
            return cacheDirectory;

        Directory.CreateDirectory(cacheDirectory);
        var archivePath = Path.Combine(cacheDirectory, RuntimeArchiveName + ".download");
        var extractionDirectory = cacheDirectory + ".partial-" + Guid.NewGuid().ToString("N");
        try
        {
            using (var response = await HttpClient.GetAsync(
                       DownloadUrl,
                       HttpCompletionOption.ResponseHeadersRead,
                       cancellationToken).ConfigureAwait(false))
            {
                response.EnsureSuccessStatusCode();
                using (var input = await response.Content.ReadAsStreamAsync().ConfigureAwait(false))
                using (var output = new FileStream(archivePath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    await input.CopyToAsync(output).ConfigureAwait(false);
                }
            }

            if (!string.IsNullOrWhiteSpace(ExpectedSha256))
            {
                var actualSha256 = ComputeSha256(archivePath);
                if (!string.Equals(actualSha256, NormalizeHash(ExpectedSha256!), StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("TensorRT runtime ZIP SHA256 does not match ExpectedSha256.");
            }

            Directory.CreateDirectory(extractionDirectory);
            ZipFile.ExtractToDirectory(archivePath, extractionDirectory);
            var extractedFiles = Directory.GetFiles(extractionDirectory, "*.dll", SearchOption.AllDirectories);
            foreach (var fileName in RequiredFiles)
                if (!extractedFiles.Any(path => string.Equals(Path.GetFileName(path), fileName, StringComparison.OrdinalIgnoreCase)))
                    throw new InvalidDataException("TensorRT runtime ZIP is missing " + fileName + ".");
            // Preserve resources; optional plugin libraries are not baseline dependencies.
            foreach (var source in extractedFiles.Where(path => IncludeOptionalLibraries || !IsOptionalLibrary(path)))
                File.Copy(source, Path.Combine(cacheDirectory, Path.GetFileName(source)), true);

            if (!ContainsRequiredRuntime(cacheDirectory))
                throw new InvalidDataException("TensorRT runtime ZIP did not produce a complete runtime directory.");
            return cacheDirectory;
        }
        catch (Exception exception) when (!(exception is OperationCanceledException))
        {
            throw new InvalidOperationException(
                "Unable to download TensorRT runtime from the configured URL. Configure TensorRtRuntime.DownloadUrl " +
                "or provision the TensorRT 10 runtime DLLs manually.", exception);
        }
        finally
        {
            TryDeleteFile(archivePath);
            TryDeleteDirectory(extractionDirectory);
        }
    }

    private static bool ContainsRequiredRuntime(string directory)
    {
        return Directory.Exists(directory) && RequiredFiles.All(fileName => File.Exists(Path.Combine(directory, fileName))) &&
               (!IncludeOptionalLibraries || OptionalFiles.All(fileName => File.Exists(Path.Combine(directory, fileName))));
    }

    private static bool IsOptionalLibrary(string path)
    {
        var name = Path.GetFileName(path);
        return name.StartsWith("cudnn", StringComparison.OrdinalIgnoreCase) ||
               name.StartsWith("cublas", StringComparison.OrdinalIgnoreCase) ||
               name.StartsWith("nvrtc", StringComparison.OrdinalIgnoreCase);
    }

    private static string GetCacheDirectory()
    {
        if (!string.IsNullOrWhiteSpace(CacheDirectory))
            return Path.GetFullPath(CacheDirectory);
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppData))
            localAppData = Path.GetTempPath();
        return Path.Combine(localAppData, "alanxinn", "TensorRtOnnxBuilder", PackageVersion,
                            IncludeOptionalLibraries ? "optional" : "minimal", "win-x64");
    }

    private static void AddCandidate(ICollection<string> candidates, string? path)
    {
        if (!string.IsNullOrWhiteSpace(path) && !candidates.Contains(path, StringComparer.OrdinalIgnoreCase))
            candidates.Add(path!);
    }

    private static string ComputeSha256(string path)
    {
        using (var stream = File.OpenRead(path))
        using (var sha256 = SHA256.Create())
            return BitConverter.ToString(sha256.ComputeHash(stream)).Replace("-", string.Empty);
    }

    private static string NormalizeHash(string value)
    {
        return new string(value.Where(character => !char.IsWhiteSpace(character)).ToArray());
    }

    private static HttpClient CreateHttpClient()
    {
        // The runtime archive can be hundreds of MiB; do not apply HttpClient's
        // default 100-second timeout to a legitimate slow download.
        var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("alanxinn.TensorRtOnnxBuilder/" + PackageVersion);
        return client;
    }

    private static void SetDllDirectory(string directory)
    {
        if (!SetDllDirectoryW(directory))
            throw new InvalidOperationException("Unable to configure the TensorRT runtime DLL search path.");
    }

    private static void TryDeleteFile(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch { }
    }

    private static void TryDeleteDirectory(string path)
    {
        try { if (Directory.Exists(path)) Directory.Delete(path, true); } catch { }
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectoryW(string? lpPathName);
}
