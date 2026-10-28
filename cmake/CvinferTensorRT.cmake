# TensorRT 10 Windows packages use versioned import libraries and runtime DLLs.
set(CUDAToolkit_ROOT "${CUDA_ROOT}")
find_package(CUDAToolkit 12.9 REQUIRED)
find_path(CVINFER_TENSORRT_INCLUDE NvInfer.h HINTS "${TENSORRT_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(CVINFER_NVINFER_LIBRARY NAMES nvinfer_10 nvinfer
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
find_library(CVINFER_PLUGIN_LIBRARY NAMES nvinfer_plugin_10 nvinfer_plugin
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)
find_library(CVINFER_ONNXPARSER_LIBRARY NAMES nvonnxparser_10 nvonnxparser
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" REQUIRED NO_DEFAULT_PATH)

file(READ "${CVINFER_TENSORRT_INCLUDE}/NvInferVersion.h" _trt_version_header)
if(NOT _trt_version_header MATCHES "#define (NV_TENSORRT_MAJOR|TRT_MAJOR_ENTERPRISE) +10([ \r\n]|$)")
  message(FATAL_ERROR "This branch requires TensorRT 10: ${TENSORRT_ROOT}")
endif()

find_path(CVINFER_CUDNN_INCLUDE cudnn_version.h
  HINTS "${CUDNN_ROOT}/include/12.9" "${CUDNN_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
file(STRINGS "${CVINFER_CUDNN_INCLUDE}/cudnn_version.h" _cudnn_version
  REGEX "^#define CUDNN_(MAJOR|MINOR|PATCHLEVEL) ")
message(STATUS "cuDNN headers: ${CVINFER_CUDNN_INCLUDE}; ${_cudnn_version}")
# TensorRT 10 does not link cuDNN. Retain the explicitly selected cuDNN 9
# runtime for applications that use it alongside inference, in builder_runtime.
file(GLOB CVINFER_CUDNN_RUNTIME_DLLS CONFIGURE_DEPENDS
  "${CUDNN_ROOT}/bin/12.9/x64/cudnn*64_9.dll" "${CUDNN_ROOT}/bin/cudnn*64_9.dll")
set(CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY "${CMAKE_BINARY_DIR}/builder_runtime")
file(GLOB CVINFER_TRT_BUILDER_RUNTIME_DLLS CONFIGURE_DEPENDS
  "${TENSORRT_ROOT}/bin/nvinfer_10.dll"
  "${TENSORRT_ROOT}/bin/nvinfer_plugin_10.dll"
  "${TENSORRT_ROOT}/bin/nvonnxparser_10.dll"
  "${TENSORRT_ROOT}/bin/nvinfer_builder_resource*_10.dll"
  "${CUDA_ROOT}/bin/cudart64_12.dll"
  "${CUDA_ROOT}/bin/cublas64_12.dll"
  "${CUDA_ROOT}/bin/cublasLt64_12.dll"
  "${CUDA_ROOT}/bin/nvrtc64_120_0.dll"
  "${CUDA_ROOT}/bin/nvrtc-builtins64_129.dll")
list(APPEND CVINFER_TRT_BUILDER_RUNTIME_DLLS ${CVINFER_CUDNN_RUNTIME_DLLS})

function(cvinfer_copy_builder_runtime TARGET_NAME)
  if(WIN32)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CVINFER_TRT_BUILDER_RUNTIME_DLLS} "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}"
      COMMAND_EXPAND_LISTS VERBATIM)
  endif()
endfunction()
