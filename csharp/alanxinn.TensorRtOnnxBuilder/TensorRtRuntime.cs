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
public static class TensorRtRuntime
{
    private const string PackageVersion = "0.1.3";
    private const string RuntimeArchiveName = "tensorrt8.6.1.6-cuda11.8-runtime-win-x64-0.1.3.zip";
    private const string DefaultDownloadUrl =
        "https://github.com/alanxinn/cvinfer/releases/download/v0.1.3/" +
        RuntimeArchiveName;

    private static readonly string[] RequiredFiles =
    {
        "nvinfer.dll",
        "nvinfer_plugin.dll",
        "nvonnxparser.dll",
        "cudart64_110.dll",
        "cublas64_11.dll",
        "cublasLt64_11.dll",
        "cudnn64_8.dll",
        "zlibwapi.dll"
    };

    private static readonly HttpClient HttpClient = CreateHttpClient();
    private static readonly SemaphoreSlim InitializeLock = new SemaphoreSlim(1, 1);
    private static string? initializedDirectory;

    /// <summary>Versioned GitHub Release ZIP URL. Override this for a private mirror.</summary>
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

            var localDirectory = FindLocalRuntimeDirectory();
            if (localDirectory == null)
            {
                if (DisableAutoDownload)
                    throw new DllNotFoundException(
                        "TensorRT runtime DLLs are missing and automatic download is disabled. " +
                        "Provision the runtime DLLs manually or set TensorRtRuntime.DisableAutoDownload to false.");
                localDirectory = await DownloadRuntimeAsync(cancellationToken).ConfigureAwait(false);
            }

            SetDllDirectory(localDirectory);
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
        AddCandidate(candidates, Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native"));
        AddCandidate(candidates, AppContext.BaseDirectory);
        var assemblyDirectory = Path.GetDirectoryName(typeof(TensorRtRuntime).GetTypeInfo().Assembly.Location);
        if (!string.IsNullOrWhiteSpace(assemblyDirectory))
            AddCandidate(candidates, Path.Combine(assemblyDirectory, "runtimes", "win-x64", "native"));

        var configuredCache = GetCacheDirectory();
        AddCandidate(candidates, configuredCache);

        return candidates.FirstOrDefault(ContainsRequiredRuntime);
    }

    private static async Task<string> DownloadRuntimeAsync(CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(DownloadUrl))
            throw new InvalidOperationException("TensorRtRuntime.DownloadUrl must be configured before first use.");

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
                if (!string.Equals(actualSha256, NormalizeHash(ExpectedSha256), StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("TensorRT runtime ZIP SHA256 does not match ExpectedSha256.");
            }

            Directory.CreateDirectory(extractionDirectory);
            ZipFile.ExtractToDirectory(archivePath, extractionDirectory);
            var extractedFiles = Directory.GetFiles(extractionDirectory, "*.dll", SearchOption.AllDirectories);
            foreach (var fileName in RequiredFiles)
            {
                var source = extractedFiles.FirstOrDefault(path =>
                    string.Equals(Path.GetFileName(path), fileName, StringComparison.OrdinalIgnoreCase));
                if (source == null)
                    throw new InvalidDataException("TensorRT runtime ZIP is missing " + fileName + ".");
                File.Copy(source, Path.Combine(cacheDirectory, fileName), true);
            }

            if (!ContainsRequiredRuntime(cacheDirectory))
                throw new InvalidDataException("TensorRT runtime ZIP did not produce a complete runtime directory.");
            return cacheDirectory;
        }
        catch (Exception exception) when (!(exception is OperationCanceledException))
        {
            throw new InvalidOperationException(
                "Unable to download TensorRT runtime from GitHub. Configure TensorRtRuntime.DownloadUrl " +
                "or provision the eight runtime DLLs manually.", exception);
        }
        finally
        {
            TryDeleteFile(archivePath);
            TryDeleteDirectory(extractionDirectory);
        }
    }

    private static bool ContainsRequiredRuntime(string directory)
    {
        return Directory.Exists(directory) && RequiredFiles.All(fileName => File.Exists(Path.Combine(directory, fileName)));
    }

    private static string GetCacheDirectory()
    {
        if (!string.IsNullOrWhiteSpace(CacheDirectory))
            return Path.GetFullPath(CacheDirectory);
        var localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(localAppData))
            localAppData = Path.GetTempPath();
        return Path.Combine(localAppData, "alanxinn", "TensorRtOnnxBuilder", PackageVersion, "win-x64");
    }

    private static void AddCandidate(ICollection<string> candidates, string? path)
    {
        if (!string.IsNullOrWhiteSpace(path) && !candidates.Contains(path, StringComparer.OrdinalIgnoreCase))
            candidates.Add(path);
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
