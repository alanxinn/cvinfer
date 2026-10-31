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

option(CVINFER_TRT_INCLUDE_OPTIONAL_LIBRARIES "Stage cuDNN/cuBLAS/NVRTC for plugins that explicitly require them" OFF)
set(CVINFER_TRT_BUILDER_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "Builder resource SMs, or ALL")
set(CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY "${CMAKE_BINARY_DIR}/builder_runtime")
file(GLOB CVINFER_TRT_BUILDER_RUNTIME_DLLS CONFIGURE_DEPENDS
  "${TENSORRT_ROOT}/bin/nvinfer_10.dll"
  "${TENSORRT_ROOT}/bin/nvinfer_plugin_10.dll"
  "${TENSORRT_ROOT}/bin/nvonnxparser_10.dll"
  "${TENSORRT_ROOT}/bin/nvinfer_builder_resource_ptx_10.dll"
  "${CUDA_ROOT}/bin/cudart64_12.dll")
if(CVINFER_TRT_BUILDER_ARCHITECTURES STREQUAL "ALL")
  file(GLOB _builder_resources CONFIGURE_DEPENDS "${TENSORRT_ROOT}/bin/nvinfer_builder_resource_sm*_10.dll")
  list(APPEND CVINFER_TRT_BUILDER_RUNTIME_DLLS ${_builder_resources})
else()
  foreach(_arch IN LISTS CVINFER_TRT_BUILDER_ARCHITECTURES)
    string(REGEX REPLACE "-(real|virtual)$" "" _arch "${_arch}")
    if(NOT _arch MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Set explicit builder SMs (e.g. 89), or ALL")
    endif()
    set(_resource "${TENSORRT_ROOT}/bin/nvinfer_builder_resource_sm${_arch}_10.dll")
    if(NOT EXISTS "${_resource}")
      message(FATAL_ERROR "Missing TensorRT builder resource: ${_resource}")
    endif()
    list(APPEND CVINFER_TRT_BUILDER_RUNTIME_DLLS "${_resource}")
  endforeach()
endif()
if(CVINFER_TRT_INCLUDE_OPTIONAL_LIBRARIES)
  file(GLOB _optional_runtime CONFIGURE_DEPENDS
    "${CUDNN_ROOT}/bin/12.9/x64/cudnn*64_9.dll" "${CUDNN_ROOT}/bin/cudnn*64_9.dll"
    "${CUDA_ROOT}/bin/cublas*64_12.dll" "${CUDA_ROOT}/bin/nvrtc*.dll")
  list(APPEND CVINFER_TRT_BUILDER_RUNTIME_DLLS ${_optional_runtime})
endif()
list(REMOVE_DUPLICATES CVINFER_TRT_BUILDER_RUNTIME_DLLS)
# Remove only obsolete SDK DLLs from this build's own staging directory.
file(GLOB _old_builder_runtime
  "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/nvinfer*.dll"
  "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/nvonnxparser*.dll"
  "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/cudart64_*.dll"
  "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/cublas*64_*.dll"
  "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/cudnn*.dll"
  "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/nvrtc*.dll")
foreach(_selected IN LISTS CVINFER_TRT_BUILDER_RUNTIME_DLLS)
  get_filename_component(_name "${_selected}" NAME)
  list(REMOVE_ITEM _old_builder_runtime "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}/${_name}")
endforeach()

function(cvinfer_copy_builder_runtime TARGET_NAME)
  if(WIN32)
    if(_old_builder_runtime)
      add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -f ${_old_builder_runtime}
        COMMAND_EXPAND_LISTS VERBATIM)
    endif()
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CVINFER_TRT_BUILDER_RUNTIME_DLLS} "${CVINFER_TRT_BUILDER_RUNTIME_DIRECTORY}"
      COMMAND_EXPAND_LISTS VERBATIM)
  endif()
endfunction()
