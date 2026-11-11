using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace alanxinn.TensorRt;

public static partial class TensorRtRuntime
{
    // Runtime assets are transitive NuGet dependencies, not a second network
    // download. Split resources keep every package below NuGet.org's limit.
    private static string? StageBundledRuntime()
    {
        var assembly = Path.GetDirectoryName(typeof(TensorRtRuntime).GetTypeInfo().Assembly.Location);
        var roots = new[] { Path.Combine(AppContext.BaseDirectory, "runtimes", "win-x64", "native"),
                            AppContext.BaseDirectory, Path.Combine(assembly ?? AppContext.BaseDirectory, "runtimes", "win-x64", "native") };
        foreach (var root in roots.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var manifest = Path.Combine(root, "tensorrt10-runtime.tsv");
            if (!File.Exists(manifest)) continue;
            var lines = File.ReadAllLines(manifest, Encoding.UTF8);
            if (lines.Length != 3 || lines[0] != "cvinfer-trt10-parts-v1")
                throw new InvalidDataException("Invalid bundled TensorRT runtime manifest.");
            var records = lines.Skip(1).Select(line => line.Split('\t')).ToArray();
            foreach (var record in records)
            {
                if (record.Length < 3 || !RequiredFiles.Contains(record[0]) || record[1].Length != 64 ||
                    !record[1].All(Uri.IsHexDigit) || record.Skip(2).Any(part => Path.GetFileName(part) != part))
                    throw new InvalidDataException("Invalid bundled TensorRT resource record.");
            }
            var splitNames = records.Select(record => record[0]).ToArray();
            if (splitNames.Distinct().Count() != 2 || !splitNames.Contains("nvinfer_builder_resource_ptx_10.dll") ||
                !splitNames.Contains("nvinfer_builder_resource_sm120_10.dll"))
                throw new InvalidDataException("The bundled runtime must contain PTX and SM120 resource parts.");
            var direct = RequiredFiles.Except(splitNames).ToArray();
            if (direct.Any(name => !File.Exists(Path.Combine(root, name))) ||
                !File.Exists(Path.Combine(root, "alanxinn_tensorrt_builder.dll")) ||
                records.Any(record => record.Skip(2).Any(part => !File.Exists(Path.Combine(root, part)))))
                throw new DllNotFoundException("Incomplete TensorRT runtime dependencies; restore all transitive NuGet packages.");
            if (IncludeOptionalLibraries && OptionalFiles.Any(name => !File.Exists(Path.Combine(root, name))))
                continue;
            var hash = ComputeSha256(manifest);
            var apiHash = ComputeSha256(Path.Combine(root, "alanxinn_tensorrt_builder.dll"));
            // The Core/SM86/SM89 DLLs can be updated without changing the PTX
            // split manifest or managed package. Include their content in the
            // cache identity so an independent runtime update is actually used.
            var payloadNames = direct.Concat(IncludeOptionalLibraries ? OptionalFiles : Array.Empty<string>())
                .Concat(Directory.GetFiles(root, "msvcp140*.dll").Select(path => Path.GetFileName(path)!))
                .Concat(Directory.GetFiles(root, "vcruntime140*.dll").Select(path => Path.GetFileName(path)!))
                .Concat(new[] { "concrt140.dll" }).Distinct(StringComparer.OrdinalIgnoreCase)
                .Where(name => File.Exists(Path.Combine(root, name))).OrderBy(name => name, StringComparer.OrdinalIgnoreCase);
            string payloadHash;
            using (var sha = SHA256.Create())
            {
                var identity = hash + apiHash + string.Concat(payloadNames.Select(name => name + ComputeSha256(Path.Combine(root, name))));
                payloadHash = BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(identity))).Replace("-", "").ToLowerInvariant();
            }
            var cache = Path.Combine(GetCacheDirectory(), "bundled-" + PackageVersion + "-" + payloadHash.Substring(0, 32));
            using (var mutex = new Mutex(false, "Local\\alanxinn.TensorRT10." + hash.Substring(0, 16)))
            {
                try { mutex.WaitOne(); } catch (AbandonedMutexException) { /* Lock is acquired. */ }
                try
                {
                    if (ContainsRequiredRuntime(cache)) return cache;
                    Directory.CreateDirectory(cache);
                    var copy = direct.Concat(IncludeOptionalLibraries ? OptionalFiles : Array.Empty<string>())
                                     .Concat(new[] { "alanxinn_tensorrt_builder.dll" })
                                     .Concat(Directory.GetFiles(root, "msvcp140*.dll").Select(path => Path.GetFileName(path)!))
                                     .Concat(Directory.GetFiles(root, "vcruntime140*.dll").Select(path => Path.GetFileName(path)!))
                                     .Concat(Directory.GetFiles(root, "concrt140.dll").Select(path => Path.GetFileName(path)!));
                    foreach (var name in copy)
                    {
                        var source = Path.Combine(root, name);
                        if (File.Exists(source)) File.Copy(source, Path.Combine(cache, name), true);
                    }
                    foreach (var record in records)
                    {
                        var destination = Path.Combine(cache, record[0]);
                        if (File.Exists(destination) && string.Equals(ComputeSha256(destination), record[1], StringComparison.OrdinalIgnoreCase))
                            continue;
                        var temporary = destination + "." + Guid.NewGuid().ToString("N") + ".partial";
                        try
                        {
                            using (var output = File.Create(temporary))
                                foreach (var part in record.Skip(2))
                                    using (var input = File.OpenRead(Path.Combine(root, part))) input.CopyTo(output);
                            if (!string.Equals(ComputeSha256(temporary), record[1], StringComparison.OrdinalIgnoreCase))
                                throw new InvalidDataException("Bundled TensorRT resource checksum mismatch: " + record[0]);
                            if (File.Exists(destination)) File.Delete(destination);
                            File.Move(temporary, destination);
                        }
                        finally { TryDeleteFile(temporary); }
                    }
                    if (!ContainsRequiredRuntime(cache)) throw new DllNotFoundException("Bundled TensorRT runtime is incomplete.");
                    return cache;
                }
                finally { mutex.ReleaseMutex(); }
            }
        }
        return null;
    }
}
