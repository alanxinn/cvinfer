using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace alanxinn.Denglin;

/// <summary>
/// Locates the Denglin SDK runtime used by nnexec, including its Python dependencies.
/// </summary>
public static class DenglinRuntime
{
    private const string PackageVersion = "0.1.4";
    private const string RuntimeArchiveName = "alanxinn-denglin-runtime-win-x64-0.1.4.zip";
    private const string DefaultDownloadUrl =
        "https://github.com/alanxinn/cvinfer/releases/download/v0.1.4/" + RuntimeArchiveName;
    private const string CompleteMarker = ".complete";

    private static readonly object Sync = new object();
    private static readonly HttpClient HttpClient = CreateHttpClient();
    private static string? initializedNativeDirectory;

    /// <summary>Optional SDK override. Environment variables are checked when this is null.</summary>
    public static string? SdkRoot { get; set; } = FirstEnvironment("DENGLIN_SDK_ROOT", "ALANXINN_DENGLIN_SDK_ROOT");
    /// <summary>Optional Python directory override.</summary>
    public static string? PythonDirectory { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_PYTHON");
    /// <summary>Versioned GitHub Release ZIP URL; override for an internal mirror.</summary>
    public static string DownloadUrl { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_RUNTIME_URL") ?? DefaultDownloadUrl;
    /// <summary>Optional SHA256 for the runtime ZIP.</summary>
    public static string? ExpectedSha256 { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_RUNTIME_SHA256");
    /// <summary>Optional runtime cache directory.</summary>
    public static string? CacheDirectory { get; set; } =
        Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_RUNTIME_CACHE");
    /// <summary>Set with ALANXINN_DENGLIN_NO_DOWNLOAD=1 to require a preinstalled runtime.</summary>
    public static bool DisableAutoDownload { get; set; } = IsTrue(
        Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_NO_DOWNLOAD"));

    /// <summary>
    /// Uses only the versioned downloaded/bundled runtime cache. This is enabled
    /// by default so a Conda or system Python cannot accidentally be selected.
    /// Set false, or set ALANXINN_DENGLIN_ALLOW_LOCAL_RUNTIME=1, for development.
    /// </summary>
    public static bool UseDownloadedRuntimeOnly { get; set; } = !IsTrue(
        Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_ALLOW_LOCAL_RUNTIME"));

    /// <summary>Receives progress while the SDK/Python runtime ZIP is downloaded.</summary>
    public static IProgress<DenglinRuntimeDownloadProgress>? DownloadProgress { get; set; }

    public static bool IsInitialized => initializedNativeDirectory != null;
    public static string? InitializedNativeDirectory => initializedNativeDirectory;

    public static string Initialize(string? sdkRoot = null, string? pythonDirectory = null,
        string? testDataDirectory = null)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows) || !Environment.Is64BitProcess)
            throw new PlatformNotSupportedException("Denglin ONNX builder requires Windows x64.");

        lock (Sync)
        {
            if (initializedNativeDirectory != null) return initializedNativeDirectory;

            RuntimeLayout? layout;
            var configuredSdk = FirstNonEmpty(sdkRoot, SdkRoot,
                FirstEnvironment("DENGLIN_SDK_ROOT", "ALANXINN_DENGLIN_SDK_ROOT"));
            var configuredPython = FirstNonEmpty(pythonDirectory, PythonDirectory,
                Environment.GetEnvironmentVariable("ALANXINN_DENGLIN_PYTHON"));
            if (UseDownloadedRuntimeOnly && string.IsNullOrWhiteSpace(configuredSdk) &&
                string.IsNullOrWhiteSpace(configuredPython))
            {
                if (DisableAutoDownload)
                    throw MissingRuntime("download-only mode is enabled but automatic download is disabled");
                var downloadedRoot = DownloadRuntimeAsync(CancellationToken.None).GetAwaiter().GetResult();
                layout = FindBundleInstallation(downloadedRoot);
            }
            else
            {
                layout = FindInstallation(configuredSdk, configuredPython);
                if (layout == null)
                {
                    if (DisableAutoDownload)
                        throw MissingRuntime("automatic download is disabled");
                    var downloadedRoot = DownloadRuntimeAsync(CancellationToken.None).GetAwaiter().GetResult();
                    layout = FindBundleInstallation(downloadedRoot);
                }
            }
            if (layout == null)
                throw MissingRuntime("the downloaded archive did not contain a valid sdk/ and python/ layout");

            var root = ResolveSdkRoot(layout.SdkRoot);
            var python = ResolvePythonDirectory(layout.PythonDirectory);
            var nativeDirectory = Directory.Exists(Path.Combine(layout.Root, "native"))
                ? Path.Combine(layout.Root, "native") : null;
            EnsurePython3Alias(python);
            var pythonCompatibility = EnsurePythonCompatibility();
            PrependPath(python, Path.Combine(root, "bin"), Path.Combine(root, "lib"),
                nativeDirectory);
            SetEnvironmentPath("PYTHONPATH", new[]
            {
                pythonCompatibility,
                Path.Combine(root, "lib"),
                Path.Combine(root, "python"),
                Path.Combine(root, "python", "nne"),
                Path.Combine(root, "lib"),
                Path.Combine(root, "python", "tvm", "python"),
                Path.Combine(root, "python", "tvm", "dl", "python")
            });
            Environment.SetEnvironmentVariable("PYTHONNOUSERSITE", "1");
            ValidatePythonEnvironment(python);
            var dataDirectory = string.IsNullOrWhiteSpace(testDataDirectory)
                ? Path.Combine(GetCacheDirectory(), "test-data")
                : Path.GetFullPath(testDataDirectory);
            Directory.CreateDirectory(dataDirectory);
            Environment.SetEnvironmentVariable("TEST_DATA_ROOT_PATH", dataDirectory);
            Environment.SetEnvironmentVariable("DENGLIN_SDK_ROOT", root);
            Environment.SetEnvironmentVariable("ALANXINN_DENGLIN_SDK_ROOT", root);
            Environment.SetEnvironmentVariable("ALANXINN_DENGLIN_PYTHON", python);
            SdkRoot = root;
            PythonDirectory = python;

            if (!string.IsNullOrWhiteSpace(nativeDirectory) && !SetDllDirectoryW(nativeDirectory))
                throw new InvalidOperationException("Unable to configure the Denglin runtime DLL search path.");
            initializedNativeDirectory = nativeDirectory ?? root;
            return initializedNativeDirectory;
        }
    }

    private sealed class RuntimeLayout
    {
        public RuntimeLayout(string root, string sdkRoot, string pythonDirectory)
        {
            Root = root;
            SdkRoot = sdkRoot;
            PythonDirectory = pythonDirectory;
        }
        public string Root { get; }
        public string SdkRoot { get; }
        public string PythonDirectory { get; }
    }

    private static RuntimeLayout? FindInstallation(string? configuredSdk, string? configuredPython,
        string? additionalRoot = null)
    {
        var roots = new List<string>();
        AddCandidate(roots, additionalRoot);
        AddCandidate(roots, Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "denglin"));
        AddCandidate(roots, Path.Combine(AppContext.BaseDirectory, "denglin"));
        var assemblyDirectory = Path.GetDirectoryName(typeof(DenglinRuntime).GetTypeInfo().Assembly.Location);
        if (!string.IsNullOrWhiteSpace(assemblyDirectory))
        {
            AddCandidate(roots, Path.Combine(assemblyDirectory, "runtimes", "win-x64", "denglin"));
            AddCandidate(roots, Path.Combine(assemblyDirectory, "denglin"));
        }
        AddCandidate(roots, GetCacheDirectory());

        var sdk = TrySdkRoot(configuredSdk) ?? DiscoverSdkRoot();
        var python = TryPythonDirectory(configuredPython);
        if (sdk != null)
        {
            python ??= TryPythonDirectory(null);
            if (python != null)
                return new RuntimeLayout(Path.GetDirectoryName(sdk) ?? sdk, sdk, python);
        }

        foreach (var root in roots)
        {
            var bundled = FindBundleInstallation(root);
            if (bundled != null) return bundled;
        }

        if (python != null)
        {
            foreach (var root in roots)
            {
                var bundledSdk = TrySdkRoot(Path.Combine(root, "sdk")) ?? TrySdkRoot(root);
                if (bundledSdk != null)
                    return new RuntimeLayout(root, bundledSdk, python);
            }
        }
        return null;
    }

    private static string? DiscoverSdkRoot()
    {
        var candidates = new List<string>
        {
            Path.Combine(AppContext.BaseDirectory, "sdk"),
            Path.Combine(AppContext.BaseDirectory, "denglin", "sdk"),
            Path.Combine(Environment.CurrentDirectory, "sdk"),
            Path.Combine(Environment.CurrentDirectory, "denglin", "sdk")
        };
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        if (Directory.Exists(programFiles))
        {
            try
            {
                candidates.AddRange(Directory.GetDirectories(programFiles, "Denglin*", SearchOption.TopDirectoryOnly)
                    .Select(directory => Path.Combine(directory, "sdk")));
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
        try
        {
            foreach (var drive in DriveInfo.GetDrives().Where(value => value.IsReady))
                candidates.Add(Path.Combine(drive.RootDirectory.FullName, "dlgpu", "sdk"));
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
        return candidates.Select(TrySdkRoot).FirstOrDefault(value => value != null);
    }

    private static RuntimeLayout? FindBundleInstallation(string root, bool searchNested = false)
    {
        var sdk = TrySdkRoot(Path.Combine(root, "sdk")) ?? TrySdkRoot(root);
        var python = TryPythonDirectory(Path.Combine(root, "python"));
        if (sdk != null && python != null && HasConverterDependencies(python))
            return new RuntimeLayout(root, sdk, python);
        if (!searchNested || !Directory.Exists(root)) return null;
        foreach (var child in Directory.GetDirectories(root))
        {
            var nested = FindBundleInstallation(child, false);
            if (nested != null) return nested;
        }
        return null;
    }

    private static string ResolveSdkRoot(string configured)
    {
        var root = TrySdkRoot(configured);
        if (root == null)
            throw new DirectoryNotFoundException("The Denglin SDK root is missing nnexec, its Python module, or native runtime: " + configured);
        return root;
    }

    private static string? TrySdkRoot(string? configured)
    {
        if (string.IsNullOrWhiteSpace(configured)) return null;
        var root = Path.GetFullPath(configured);
        var module = Path.Combine(root, "python", "tvm", "dl", "python", "dl", "__init__.py");
        var topiCuda = Path.Combine(root, "python", "tvm", "dl", "python", "dl", "topi", "cuda", "__init__.py");
        var nnexec = Path.Combine(root, "bin", "nnexec.exe");
        return Directory.Exists(root) && File.Exists(module) && File.Exists(topiCuda) && File.Exists(nnexec) &&
            (File.Exists(Path.Combine(root, "lib", "dlnne.dll")) ||
             File.Exists(Path.Combine(root, "bin", "dlnne.dll"))) ? root : null;
    }

    private static string ResolvePythonDirectory(string configured)
    {
        var directory = TryPythonDirectory(configured);
        if (directory == null)
            throw new FileNotFoundException("python.exe was not found in: " + configured);
        if (!HasConverterDependencies(directory))
            throw new FileNotFoundException(
                "NumPy, ONNX, SciPy, setuptools and dataclasses are required in the selected Python " +
                "environment: " + directory + ". Install the Denglin converter dependencies or set " +
                "DenglinRuntime.PythonDirectory to a complete Python environment.");
        return directory;
    }

    private static string? TryPythonDirectory(string? configured)
    {
        if (!string.IsNullOrWhiteSpace(configured))
        {
            var directory = Path.GetFullPath(configured);
            return File.Exists(Path.Combine(directory, "python.exe")) ? directory : null;
        }
        var candidates = new List<string>();
        var path = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (var item in path.Split(Path.PathSeparator))
        {
            var directory = item.Trim().Trim('"');
            if (directory.Length > 0 && File.Exists(Path.Combine(directory, "python.exe")) &&
                !candidates.Contains(directory, StringComparer.OrdinalIgnoreCase))
                candidates.Add(directory);
        }

        var commonRoots = new[]
        {
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python"),
            Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData)
        };
        foreach (var root in commonRoots.Where(Directory.Exists))
        {
            string[] directories;
            try
            {
                directories = Directory.GetDirectories(root, "Python*", SearchOption.TopDirectoryOnly);
            }
            catch (IOException) { continue; }
            catch (UnauthorizedAccessException) { continue; }
            foreach (var directory in directories.OrderByDescending(value => value, StringComparer.OrdinalIgnoreCase))
            {
                if (File.Exists(Path.Combine(directory, "python.exe")) &&
                    !candidates.Contains(directory, StringComparer.OrdinalIgnoreCase))
                    candidates.Add(directory);
            }
        }
        return candidates.FirstOrDefault(HasConverterDependencies) ??
            candidates.FirstOrDefault();
    }

    private static bool HasConverterDependencies(string directory) =>
        HasPythonPackage(directory, "numpy") &&
        HasPythonPackage(directory, "onnx") &&
        HasPythonPackage(directory, "scipy") &&
        HasPythonPackage(directory, "setuptools") &&
        HasPythonModule(directory, "dataclasses");

    private static bool HasPythonModule(string directory, string moduleName)
    {
        var roots = new[]
        {
            Path.Combine(directory, "Lib"),
            Path.Combine(directory, "Lib", "site-packages")
        };
        return roots.Any(root => File.Exists(Path.Combine(root, moduleName + ".py")) ||
            Directory.Exists(Path.Combine(root, moduleName)));
    }

    private static void EnsurePython3Alias(string directory)
    {
        var python = Path.Combine(directory, "python.exe");
        var python3 = Path.Combine(directory, "python3.exe");
        if (File.Exists(python3)) return;
        try
        {
            File.Copy(python, python3);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            throw new InvalidOperationException(
                "nnexec requires python3.exe next to the selected Python. Unable to create it in: " +
                directory, error);
        }
    }

    private static string EnsurePythonCompatibility()
    {
        var directory = Path.Combine(GetCacheDirectory(), "python-compat");
        Directory.CreateDirectory(directory);
        var siteCustomisation = Path.Combine(directory, "sitecustomize.py");
        const string content = "try:\n    import setuptools\nexcept ImportError:\n    pass\n";
        if (!File.Exists(siteCustomisation) ||
            !string.Equals(File.ReadAllText(siteCustomisation), content, StringComparison.Ordinal))
            File.WriteAllText(siteCustomisation, content);
        return directory;
    }

    private static void ValidatePythonEnvironment(string directory)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = Path.Combine(directory, "python3.exe"),
            Arguments = "-c \"import numpy, onnx, scipy, setuptools, dataclasses; import dl\"",
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        using var process = new Process { StartInfo = startInfo };
        if (!process.Start())
            throw new InvalidOperationException("Unable to validate the Denglin Python environment.");
        var output = process.StandardOutput.ReadToEndAsync();
        var error = process.StandardError.ReadToEndAsync();
        process.WaitForExit();
        Task.WaitAll(output, error);
        if (process.ExitCode == 0) return;
        var detail = string.IsNullOrWhiteSpace(error.Result) ? output.Result : error.Result;
        throw new InvalidOperationException(
            "Denglin Python dependency validation failed: " + detail.Trim());
    }

    private static bool HasPythonPackage(string directory, string packageName)
    {
        var sitePackages = new List<string>
        {
            Path.Combine(directory, "Lib", "site-packages")
        };
        try
        {
            var pythonLib = Path.Combine(directory, "lib");
            if (Directory.Exists(pythonLib))
            {
                sitePackages.AddRange(Directory.GetDirectories(pythonLib, "python*", SearchOption.TopDirectoryOnly)
                    .Select(value => Path.Combine(value, "site-packages")));
            }
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }

        foreach (var site in sitePackages.Where(Directory.Exists))
        {
            if (Directory.Exists(Path.Combine(site, packageName))) return true;
            try
            {
                if (Directory.GetDirectories(site, packageName + "-*", SearchOption.TopDirectoryOnly).Length > 0)
                    return true;
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
        return false;
    }

    private static async Task<string> DownloadRuntimeAsync(CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(DownloadUrl))
            throw new InvalidOperationException("DenglinRuntime.DownloadUrl must be configured before first use.");

        var cache = GetCacheDirectory();
        if (FindBundleInstallation(cache) != null) return cache;
        Directory.CreateDirectory(cache);
        var archivePath = Path.Combine(cache, RuntimeArchiveName + ".download");
        var extraction = cache + ".partial-" + Guid.NewGuid().ToString("N");
        try
        {
            using (var response = await HttpClient.GetAsync(DownloadUrl,
                       HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false))
            {
                response.EnsureSuccessStatusCode();
                using (var input = await response.Content.ReadAsStreamAsync().ConfigureAwait(false))
                using (var output = new FileStream(archivePath, FileMode.Create, FileAccess.Write, FileShare.None))
                {
                    var totalBytes = response.Content.Headers.ContentLength ?? -1L;
                    var downloadedBytes = 0L;
                    var lastReportBytes = 0L;
                    var stopwatch = Stopwatch.StartNew();
                    var buffer = new byte[64 * 1024];
                    ReportDownloadProgress(0, totalBytes, stopwatch.Elapsed, false);
                    int read;
                    while ((read = await input.ReadAsync(buffer, 0, buffer.Length, cancellationToken)
                               .ConfigureAwait(false)) > 0)
                    {
                        await output.WriteAsync(buffer, 0, read, cancellationToken).ConfigureAwait(false);
                        downloadedBytes += read;
                        if (downloadedBytes - lastReportBytes >= 256 * 1024 ||
                            (totalBytes > 0 && downloadedBytes >= totalBytes))
                        {
                            ReportDownloadProgress(downloadedBytes, totalBytes,
                                stopwatch.Elapsed, false);
                            lastReportBytes = downloadedBytes;
                        }
                    }
                    ReportDownloadProgress(downloadedBytes, totalBytes, stopwatch.Elapsed, true);
                }
            }
            if (!string.IsNullOrWhiteSpace(ExpectedSha256))
            {
                var actual = ComputeSha256(archivePath);
                if (!string.Equals(actual, NormalizeHash(ExpectedSha256), StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("Denglin runtime ZIP SHA256 does not match ExpectedSha256.");
            }

            Directory.CreateDirectory(extraction);
            ExtractZipSafely(archivePath, extraction);
            var layout = FindBundleInstallation(extraction, true);
            if (layout == null)
                throw new InvalidDataException("Denglin runtime ZIP must contain sdk/ and python/ directories.");
            CopyDirectory(layout.SdkRoot, Path.Combine(cache, "sdk"));
            CopyDirectory(layout.PythonDirectory, Path.Combine(cache, "python"));
            var native = Path.Combine(layout.Root, "native");
            if (Directory.Exists(native)) CopyDirectory(native, Path.Combine(cache, "native"));
            File.WriteAllText(Path.Combine(cache, CompleteMarker), PackageVersion);
            return cache;
        }
        catch (Exception error) when (!(error is OperationCanceledException))
        {
            throw new InvalidOperationException(
                "Unable to download Denglin SDK/Python runtime. Configure DenglinRuntime.DownloadUrl " +
                "or provision sdk/ and python/ under the runtime cache.", error);
        }
        finally
        {
            TryDeleteFile(archivePath);
            TryDeleteDirectory(extraction);
        }
    }

    private static void ExtractZipSafely(string archivePath, string destination)
    {
        var root = Path.GetFullPath(destination) + Path.DirectorySeparatorChar;
        using (var archive = ZipFile.OpenRead(archivePath))
        {
            foreach (var entry in archive.Entries)
            {
                var target = Path.GetFullPath(Path.Combine(destination, entry.FullName));
                if (!target.StartsWith(root, StringComparison.OrdinalIgnoreCase))
                    throw new InvalidDataException("Runtime ZIP contains an unsafe path: " + entry.FullName);
                if (string.IsNullOrEmpty(entry.Name))
                {
                    Directory.CreateDirectory(target);
                    continue;
                }
                Directory.CreateDirectory(Path.GetDirectoryName(target) ?? destination);
                entry.ExtractToFile(target, true);
            }
        }
    }

    private static void CopyDirectory(string source, string destination)
    {
        Directory.CreateDirectory(destination);
        foreach (var file in Directory.GetFiles(source))
            File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), true);
        foreach (var directory in Directory.GetDirectories(source))
            CopyDirectory(directory, Path.Combine(destination, Path.GetFileName(directory)));
    }

    private static void AddCandidate(ICollection<string> values, string? value)
    {
        if (!string.IsNullOrWhiteSpace(value) && !values.Contains(value, StringComparer.OrdinalIgnoreCase))
            values.Add(Path.GetFullPath(value));
    }

    private static void PrependPath(params string[] directories) => SetEnvironmentPath("PATH", directories);

    private static void SetEnvironmentPath(string variable, IEnumerable<string> directories)
    {
        var existing = Environment.GetEnvironmentVariable(variable) ?? string.Empty;
        var values = directories.Where(Directory.Exists).Concat(existing.Split(Path.PathSeparator))
            .Where(value => !string.IsNullOrWhiteSpace(value)).Distinct(StringComparer.OrdinalIgnoreCase);
        Environment.SetEnvironmentVariable(variable, string.Join(Path.PathSeparator.ToString(), values));
    }

    private static string GetCacheDirectory()
    {
        if (!string.IsNullOrWhiteSpace(CacheDirectory)) return Path.GetFullPath(CacheDirectory);
        var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        if (string.IsNullOrWhiteSpace(local)) local = Path.GetTempPath();
        return Path.Combine(local, "alanxinn", "DenglinOnnxBuilder", PackageVersion, "win-x64");
    }

    private static string ComputeSha256(string path)
    {
        using (var stream = File.OpenRead(path))
        using (var sha = SHA256.Create())
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty);
    }

    private static string NormalizeHash(string value) =>
        new string(value.Where(character => !char.IsWhiteSpace(character)).ToArray());

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("alanxinn.DenglinOnnxBuilder/" + PackageVersion);
        return client;
    }

    private static void ReportDownloadProgress(long downloadedBytes, long totalBytes,
        TimeSpan elapsed, bool completed)
    {
        var progress = DownloadProgress;
        if (progress == null) return;
        var percent = totalBytes > 0
            ? (int)Math.Min(100, downloadedBytes * 100L / totalBytes)
            : -1;
        var speed = elapsed.TotalSeconds > 0 ? downloadedBytes / elapsed.TotalSeconds : 0;
        try
        {
            progress.Report(new DenglinRuntimeDownloadProgress(
                downloadedBytes, totalBytes, percent, speed, completed));
        }
        catch { }
    }

    private static InvalidOperationException MissingRuntime(string reason) => new InvalidOperationException(
        "Denglin SDK/Python runtime was not found (" + reason + "). Set DenglinRuntime.SdkRoot and " +
        "DenglinRuntime.PythonDirectory, provision the runtime cache, or configure " +
        "ALANXINN_DENGLIN_RUNTIME_URL.");

    private static string? FirstNonEmpty(params string?[] values) =>
        values.FirstOrDefault(value => !string.IsNullOrWhiteSpace(value));

    private static string? FirstEnvironment(params string[] names) =>
        names.Select(Environment.GetEnvironmentVariable).FirstOrDefault(value => !string.IsNullOrWhiteSpace(value));

    private static bool IsTrue(string? value) =>
        string.Equals(value, "1", StringComparison.OrdinalIgnoreCase) ||
        string.Equals(value, "true", StringComparison.OrdinalIgnoreCase);

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

/// <summary>Progress information for a Denglin SDK/Python runtime download.</summary>
public sealed class DenglinRuntimeDownloadProgress
{
    internal DenglinRuntimeDownloadProgress(long bytesDownloaded, long totalBytes,
        int percent, double bytesPerSecond, bool isCompleted)
    {
        BytesDownloaded = bytesDownloaded;
        TotalBytes = totalBytes;
        Percent = percent;
        BytesPerSecond = bytesPerSecond;
        IsCompleted = isCompleted;
    }

    public long BytesDownloaded { get; }
    public long TotalBytes { get; }
    /// <summary>0-100, or -1 when the server did not send Content-Length.</summary>
    public int Percent { get; }
    public double BytesPerSecond { get; }
    public bool IsCompleted { get; }
}
