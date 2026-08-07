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
  # Keep the builder deployment deterministic. Only these runtime DLLs are
  # copied; the C++ builder DLL itself is the target output.
  set(TENSORRT_BUILDER_RUNTIME_DLLS)
  foreach(_dll IN ITEMS
      "${TENSORRT_ROOT}/lib/nvinfer.dll"
      "${TENSORRT_ROOT}/bin/nvinfer.dll"
      "${TENSORRT_ROOT}/lib/nvinfer_plugin.dll"
      "${TENSORRT_ROOT}/bin/nvinfer_plugin.dll"
      "${TENSORRT_ROOT}/lib/nvonnxparser.dll"
      "${TENSORRT_ROOT}/bin/nvonnxparser.dll"
      "${CUDA_ROOT}/bin/cudart64_110.dll"
      "${CUDA_ROOT}/bin/cublas64_11.dll"
      "${CUDA_ROOT}/bin/cublasLt64_11.dll"
      "${CUDA_ROOT}/bin/cudnn64_8.dll"
      "${CUDA_ROOT}/bin/zlibwapi.dll"
      "${TENSORRT_ROOT}/lib/zlibwapi.dll"
      "${TENSORRT_ROOT}/bin/zlibwapi.dll")
    if(EXISTS "${_dll}")
      list(APPEND TENSORRT_BUILDER_RUNTIME_DLLS "${_dll}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES TENSORRT_BUILDER_RUNTIME_DLLS)

  if(TENSORRT_BUILDER_RUNTIME_DLLS)
    add_custom_command(TARGET alanxinn_tensorrt_builder POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${TENSORRT_BUILDER_RUNTIME_DLLS}
              "$<TARGET_FILE_DIR:alanxinn_tensorrt_builder>"
      COMMAND_EXPAND_LISTS)
  endif()
endif()
