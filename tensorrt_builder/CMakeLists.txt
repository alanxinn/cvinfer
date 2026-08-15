add_library(alanxinn_tensorrt_builder SHARED
  src/tensorrt_builder.cpp)

# Keep the ONNX-to-engine builder and its TensorRT/CUDA runtime DLLs isolated
# from the regular cvinfer binaries in `${CMAKE_BINARY_DIR}/bin`.
set_target_properties(alanxinn_tensorrt_builder PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tensorrt_builder"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tensorrt_builder")

target_compile_definitions(alanxinn_tensorrt_builder PRIVATE
  TENSORRT_BUILDER_BUILD_DLL)

target_include_directories(alanxinn_tensorrt_builder PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CUDA_ROOT}/include"
  "${TENSORRT_ROOT}/include")

target_link_directories(alanxinn_tensorrt_builder PRIVATE
  "${CUDA_ROOT}/lib/x64"
  "${CUDA_ROOT}/lib64"
  "${TENSORRT_ROOT}/lib")

target_link_libraries(alanxinn_tensorrt_builder PRIVATE
  nvinfer
  nvonnxparser
  nvinfer_plugin
  cudart)

if(WIN32)
  # The Builder NuGet package contains only alanxinn_tensorrt_builder.dll;
  # TensorRT/CUDA dependencies are provisioned by TensorRtRuntime separately.
  # Remove dependencies left by older builds from this output directory.
  file(GLOB TENSORRT_BUILDER_STALE_RUNTIME_DLLS
    "${CMAKE_BINARY_DIR}/tensorrt_builder/nvinfer*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/nvonnxparser*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/cudart64_*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/cublas64_*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/cublasLt64_*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/cudnn*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/nvrtc64_*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/nvrtc-builtins64_*.dll"
    "${CMAKE_BINARY_DIR}/tensorrt_builder/zlibwapi.dll")
  if(TENSORRT_BUILDER_STALE_RUNTIME_DLLS)
    add_custom_command(TARGET alanxinn_tensorrt_builder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm -f
              ${TENSORRT_BUILDER_STALE_RUNTIME_DLLS}
      COMMAND_EXPAND_LISTS)
  endif()
endif()
