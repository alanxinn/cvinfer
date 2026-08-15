#include "yolo/yolo.h"

#if defined(_WIN32)
#include <windows.h>
#include <string>
#else
#include <dlfcn.h>
#endif

namespace cvinfer::yolo {

bool cuda_backend_available() noexcept {
#if !defined(CVINFER_HAS_CUDA_BACKEND)
    return false;
#else
#if defined(_WIN32)
    constexpr const wchar_t* kLibraryName = L"alanxinn_cvinfer_cuda.dll";
    if (GetModuleHandleW(kLibraryName)) return true;

    // When alanxinn_cvinfer_trt.dll is loaded from a NuGet native asset directory,
    // the CUDA backend is usually beside it under:
    //   runtimes/win-x64/native/
    // LoadLibraryW("alanxinn_cvinfer_cuda.dll") only searches the process
    // search path and does not reliably search that directory. Resolve the
    // current module path and load the sibling DLL explicitly first.
    HMODULE current_module = nullptr;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&cuda_backend_available),
            &current_module)) {
        wchar_t module_path[MAX_PATH] = {};
        const DWORD length = GetModuleFileNameW(
            current_module, module_path,
            static_cast<DWORD>(sizeof(module_path) / sizeof(module_path[0])));
        if (length > 0 &&
            length < static_cast<DWORD>(sizeof(module_path) / sizeof(module_path[0]))) {
            std::wstring sibling_path(module_path, length);
            const std::wstring::size_type separator = sibling_path.find_last_of(L"\\/");
            if (separator != std::wstring::npos) {
                sibling_path.resize(separator + 1);
                sibling_path += kLibraryName;
                if (LoadLibraryW(sibling_path.c_str()) != nullptr) return true;

                // CMake deployments keep the optional CUDA backend in the
                // sibling bintrt directory. NuGet deployments keep it beside
                // alanxinn_cvinfer_trt.dll, which was attempted above.
                sibling_path.resize(separator + 1);
                sibling_path += L"bintrt\\";
                sibling_path += kLibraryName;
                if (LoadLibraryW(sibling_path.c_str()) != nullptr) return true;
            }
        }
    }

    // Keep the normal search-path fallback for applications that deploy all
    // native DLLs beside the executable.
    return LoadLibraryW(kLibraryName) != nullptr;
#else
    return dlopen("alanxinn_cvinfer_cuda.so", RTLD_LAZY | RTLD_LOCAL) != nullptr;
#endif
#endif
}

} // namespace cvinfer::yolo
