# alanxinn.CvinferSharp.Denglin

This package contains the managed `alanxinn.CvinferSharp` API and the native
Denglin inference runtime. Native files are declared under
`runtimes/win-x64/native` and are resolved by NuGet/.NET at application load
time; applications do not need to copy DLLs manually.

```csharp
var config = new CvinferConfig
{
    InferenceBackend = CvinferInferenceBackend.Denglin
};

using var model = new CvinferModel(
    "yolo11n.denglin.cache.raw.engine",
    CvinferModelType.Yolo11Detection,
    config);
```

The package is Windows x64 and requires a Denglin engine/cache produced for the
same Denglin SDK/device runtime. It also carries the SDK device runtime loader
under `runtimes/win-x64/hc2` and `runtimes/win-x64/hc3`; the package build
targets copy these directories beside the native runtime automatically, so no
manual DLL copying is required.
