cmake_minimum_required(VERSION 3.23)

project(alanxinn_cvinfer LANGUAGES CXX)

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(BUILD_CUDA_BACKEND
  "Build the optional CUDA preprocessing/postprocessing backend"
  OFF)
option(BUILD_DENGLIN_ACCELERATOR "Build Denglin GPU preprocessing/postprocessing" OFF)
set(ALANXINN_DENGLIN_ACCELERATOR_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/cvinfer_private/denglin_backend"
  CACHE PATH "Private Denglin accelerator source directory")

option(BUILD_DENGLIN_BACKEND
  "Build the optional Denglin inference backend"
  OFF)
set(ALANXINN_CUDA_BACKEND_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/private/cuda_backend"
  CACHE PATH "Private CUDA backend source directory")
set(ALANXINN_PUBLIC_YOLO_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/yolo/include"
  CACHE PATH "Public YOLO11/CUDA interface include directory")
set(ALANXINN_PUBLIC_CORE_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/core/include"
  CACHE PATH "Public core interface include directory")

if(BUILD_CUDA_BACKEND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Keep backend-specific public native APIs in bin and stage their runtime DLLs
# separately.  TensorRT and Denglin intentionally use different CUDA runtime
# implementations, so they must never be linked into the same native DLL.
set(CVINFER_TRT_RUNTIME_DIRECTORY "${CMAKE_BINARY_DIR}/bintrt")
set(CVINFER_DENGLIN_RUNTIME_DIRECTORY "${CMAKE_BINARY_DIR}/bindl")

if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>)

if(MSVC)
  add_compile_options(
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=/O2>)
else()
  add_compile_options(
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>)
endif()

set(CUDA_ROOT "${CMAKE_CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "CUDA root directory")
set(TENSORRT_ROOT "" CACHE PATH "TensorRT root directory")
set(DENGLIN_SDK_ROOT "" CACHE PATH "Denglin Hamming SDK root")
set(OPENCV_ROOT "" CACHE PATH "OpenCV root directory or install prefix")

if(BUILD_DENGLIN_BACKEND)
  set(_denglin_sdk_candidates
    "${DENGLIN_SDK_ROOT}"
    "$ENV{DENGLIN_SDK_ROOT}"
    "$ENV{DENGLIN_ROOT}"
    "$ENV{DLGPU_ROOT}"
    "$ENV{DLGPU_HOME}"
    "$ENV{DLICC_PATH}/.."
    "$ENV{DENGLIN_ROOT}/sdk"
    "$ENV{DLGPU_ROOT}/sdk"
    "$ENV{DLGPU_HOME}/sdk"
    "$ENV{ProgramFiles}/Denglin"
    "$ENV{ProgramFiles}/Denglin/sdk"
    "C:/Program Files/dlgpu/sdk"
    "C:/dlgpu/sdk"
    "D:/dlgpu/sdk"
    "E:/dlgpu/sdk"
    "C:/denglin-hamming/sdk"
    "D:/denglin-hamming/sdk"
    "E:/denglin-hamming/sdk")

  set(_denglin_sdk_found "")
  foreach(_candidate IN LISTS _denglin_sdk_candidates)
    if(_candidate AND EXISTS "${_candidate}/include/dlnne/dlnne.h")
      get_filename_component(_denglin_sdk_found "${_candidate}" ABSOLUTE)
      break()
    endif()
  endforeach()

  if(_denglin_sdk_found)
    set(DENGLIN_SDK_ROOT "${_denglin_sdk_found}" CACHE PATH
      "Denglin Hamming SDK root" FORCE)
    message(STATUS "Denglin SDK found: ${DENGLIN_SDK_ROOT}")
  else()
    set(DENGLIN_SDK_ROOT "" CACHE PATH
      "Denglin Hamming SDK root" FORCE)
    message(STATUS
      "Denglin SDK was not found in configured paths or common local locations")
  endif()
  unset(_denglin_sdk_candidates)
  unset(_denglin_sdk_found)
endif()

if(NOT CUDA_ROOT)
  if(CMAKE_CUDA_COMPILER)
    get_filename_component(_cuda_bin_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    get_filename_component(CUDA_ROOT "${_cuda_bin_dir}/.." ABSOLUTE)
  endif()
endif()

if(NOT OpenCV_DIR)
  foreach(_opencv_config_dir
      "${OPENCV_ROOT}"
      "${OPENCV_ROOT}/build"
      "${OPENCV_ROOT}/x64/vc16/lib"
      "${OPENCV_ROOT}/lib/cmake/opencv4")
    if(EXISTS "${_opencv_config_dir}/OpenCVConfig.cmake")
      set(OpenCV_DIR "${_opencv_config_dir}" CACHE PATH "Directory containing OpenCVConfig.cmake")
      break()
    endif()
  endforeach()
endif()

find_package(OpenCV REQUIRED CONFIG)

add_library(alanxinn_cvinfer_common INTERFACE)
target_include_directories(alanxinn_cvinfer_common INTERFACE
  "${CUDA_ROOT}/include"
  "${TENSORRT_ROOT}/include"
  ${OpenCV_INCLUDE_DIRS}
)
target_link_directories(alanxinn_cvinfer_common INTERFACE
  "${CUDA_ROOT}/lib/x64"
  "${CUDA_ROOT}/lib64"
  "${TENSORRT_ROOT}/lib"
)
target_link_libraries(alanxinn_cvinfer_common INTERFACE
  nvinfer
  cudart
  ${OpenCV_LIBS}
)

function(cvinfer_find_existing_dir OUT_VAR)
  set(_result "")
  foreach(_candidate IN LISTS ARGN)
    if(_candidate AND EXISTS "${_candidate}")
      set(_result "${_candidate}")
      break()
    endif()
  endforeach()
  set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()

cvinfer_find_existing_dir(CVINFER_OPENCV_BIN_DIR
  "${OpenCV_DIR}/../bin"
  "${OpenCV_DIR}/x64/vc16/bin"
  "${OpenCV_DIR}/bin"
  "${OPENCV_ROOT}/bin")

file(GLOB CVINFER_OPENCV_RELEASE_DLLS CONFIGURE_DEPENDS
  "${CVINFER_OPENCV_BIN_DIR}/opencv_world*.dll")
list(FILTER CVINFER_OPENCV_RELEASE_DLLS EXCLUDE REGEX "d\\.dll$")

# Keep only the runtime set required by the public TensorRT inference DLL in
# bintrt.  The ONNX builder has a separate deployment directory below, so its
# parser/plugin/cuBLAS/cuDNN dependencies must not leak into the inference
# runtime or the CvinferSharp package.
file(GLOB CVINFER_TRT_RUNTIME_DLLS CONFIGURE_DEPENDS
  ${CVINFER_OPENCV_RELEASE_DLLS}
  "${TENSORRT_ROOT}/lib/nvinfer.dll"
  "${TENSORRT_ROOT}/bin/nvinfer.dll"
  "${TENSORRT_ROOT}/lib/zlibwapi.dll"
  "${TENSORRT_ROOT}/bin/zlibwapi.dll"
  "${CUDA_ROOT}/bin/cudart64_*.dll"
  "${CUDA_ROOT}/bin/zlibwapi.dll")
list(FILTER CVINFER_TRT_RUNTIME_DLLS EXCLUDE REGEX "d\\.dll$")
list(REMOVE_DUPLICATES CVINFER_TRT_RUNTIME_DLLS)

# Remove files that may have been staged by older builds.  This keeps a
# reconfigured build tree deterministic without deleting the whole output
# directory (which may contain the optional CUDA backend DLL).
file(GLOB CVINFER_TRT_STALE_RUNTIME_DLLS
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/nvinfer_plugin.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/nvonnxparser.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/cublas64_*.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/cublasLt64_*.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/cudnn*.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/nvrtc64_*.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/nvrtc-builtins64_*.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/cudart64_*d.dll"
  "${CVINFER_TRT_RUNTIME_DIRECTORY}/opencv_world*d.dll")

set(CVINFER_DENGLIN_RUNTIME_DLL_NAMES
  dlnne.dll
  dlnne_plugin.dll
  curt.dll
  Executable_tc.dll
  dleol.dll
  hcrt.dll
  dlhal2.dll
  dlutil.dll
  utils_common.dll
  utils_log.dll
  device_properties.dll
  dltu.dll)
set(CVINFER_DENGLIN_RUNTIME_DLLS)
set(CVINFER_DENGLIN_HC2_RUNTIME_DLL "")
set(CVINFER_DENGLIN_HC3_RUNTIME_DLL "")
if(BUILD_DENGLIN_BACKEND AND DENGLIN_SDK_ROOT)
  foreach(_denglin_dll IN LISTS CVINFER_DENGLIN_RUNTIME_DLL_NAMES)
    if(EXISTS "${DENGLIN_SDK_ROOT}/lib/${_denglin_dll}")
      list(APPEND CVINFER_DENGLIN_RUNTIME_DLLS
        "${DENGLIN_SDK_ROOT}/lib/${_denglin_dll}")
    endif()
  endforeach()
  if(EXISTS "${DENGLIN_SDK_ROOT}/lib/dlhc2.dll")
    set(CVINFER_DENGLIN_HC2_RUNTIME_DLL
      "${DENGLIN_SDK_ROOT}/lib/dlhc2.dll")
  endif()
  if(EXISTS "${DENGLIN_SDK_ROOT}/lib/dlhc3.dll")
    set(CVINFER_DENGLIN_HC3_RUNTIME_DLL
      "${DENGLIN_SDK_ROOT}/lib/dlhc3.dll")
  endif()
  list(REMOVE_DUPLICATES CVINFER_DENGLIN_RUNTIME_DLLS)
endif()

file(GLOB CVINFER_DELAY_LOAD_DLLS CONFIGURE_DEPENDS
  ${CVINFER_OPENCV_RELEASE_DLLS}
  "${TENSORRT_ROOT}/lib/nvinfer.dll"
  "${CUDA_ROOT}/bin/cudart64_*.dll")
list(FILTER CVINFER_DELAY_LOAD_DLLS EXCLUDE REGEX "d\\.dll$")

function(cvinfer_copy_runtime TARGET_NAME)
  if(NOT WIN32)
    return()
  endif()

  set(_backend "BOTH")
  if(ARGC GREATER 1)
    string(TOUPPER "${ARGV1}" _backend)
  endif()
  if(NOT _backend STREQUAL "BOTH" AND
     NOT _backend STREQUAL "TENSORRT" AND
     NOT _backend STREQUAL "DENGLIN")
    message(FATAL_ERROR
      "cvinfer_copy_runtime(${TARGET_NAME}): backend must be BOTH, TENSORRT or DENGLIN")
  endif()

  set(_runtime_commands)
  if(_backend STREQUAL "BOTH" OR _backend STREQUAL "TENSORRT")
    if(CVINFER_TRT_STALE_RUNTIME_DLLS)
      list(APPEND _runtime_commands
        COMMAND ${CMAKE_COMMAND} -E rm -f
                ${CVINFER_TRT_STALE_RUNTIME_DLLS})
    endif()
    list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E make_directory
              "${CVINFER_TRT_RUNTIME_DIRECTORY}")
    if(CVINFER_TRT_RUNTIME_DLLS)
      list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CVINFER_TRT_RUNTIME_DLLS}
              "${CVINFER_TRT_RUNTIME_DIRECTORY}")
    endif()
  endif()
  if(_backend STREQUAL "BOTH" OR _backend STREQUAL "DENGLIN")
    list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E make_directory
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}")
    if(CVINFER_DENGLIN_RUNTIME_DLLS)
      list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CVINFER_DENGLIN_RUNTIME_DLLS}
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}")
    endif()
  endif()
  if((_backend STREQUAL "BOTH" OR _backend STREQUAL "DENGLIN") AND
     CVINFER_DENGLIN_HC2_RUNTIME_DLL)
    list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}/hc2"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CVINFER_DENGLIN_HC2_RUNTIME_DLL}"
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}/hc2/dlhc2.dll")
  endif()
  if((_backend STREQUAL "BOTH" OR _backend STREQUAL "DENGLIN") AND
     CVINFER_DENGLIN_HC3_RUNTIME_DLL)
    list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E make_directory
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}/hc3"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CVINFER_DENGLIN_HC3_RUNTIME_DLL}"
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}/hc3/dlhc3.dll")
  elseif((_backend STREQUAL "BOTH" OR _backend STREQUAL "DENGLIN") AND
         CVINFER_DENGLIN_HC2_RUNTIME_DLL)
    # SDK versions that only ship dlhc2.dll use the same runtime image for
    # the hc3 lookup performed by hcrt.dll.
    list(APPEND _runtime_commands
      COMMAND ${CMAKE_COMMAND} -E make_directory
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}/hc3"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CVINFER_DENGLIN_HC2_RUNTIME_DLL}"
              "${CVINFER_DENGLIN_RUNTIME_DIRECTORY}/hc3/dlhc2.dll")
  endif()
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    ${_runtime_commands}
    COMMAND_EXPAND_LISTS
    VERBATIM)
endfunction()

function(cvinfer_enable_delay_load TARGET_NAME)
  if(MSVC)
    target_link_libraries(${TARGET_NAME} PRIVATE delayimp)
    foreach(_dll IN LISTS CVINFER_DELAY_LOAD_DLLS)
      get_filename_component(_dll_name "${_dll}" NAME)
      target_link_options(${TARGET_NAME} PRIVATE "/DELAYLOAD:${_dll_name}")
    endforeach()
  endif()
endfunction()


option(BUILD_CVINFER_DLL "Build C# compatible cvinfer DLL" OFF)
option(BUILD_TRT_ONNX_BUILDER_DLL
  "Build the TensorRT C++ ONNX-to-engine builder DLL"
  OFF)
add_subdirectory(core)
if(BUILD_TRT_ONNX_BUILDER_DLL)
  add_subdirectory(tensorrt_builder)
endif()
if(BUILD_CUDA_BACKEND)
  if(NOT EXISTS "${ALANXINN_CUDA_BACKEND_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "CUDA backend source was not found: ${ALANXINN_CUDA_BACKEND_DIR}")
  endif()
  add_subdirectory("${ALANXINN_CUDA_BACKEND_DIR}"
                   "${CMAKE_BINARY_DIR}/cuda_backend")
endif()
if(BUILD_DENGLIN_ACCELERATOR)
  enable_testing()
  if(NOT BUILD_DENGLIN_BACKEND)
    message(FATAL_ERROR "BUILD_DENGLIN_ACCELERATOR requires BUILD_DENGLIN_BACKEND=ON")
  endif()
  add_subdirectory("${ALANXINN_DENGLIN_ACCELERATOR_DIR}" "${CMAKE_BINARY_DIR}/denglin_accelerator")
endif()
add_subdirectory(yolo)
add_subdirectory(rf_detr)
if(BUILD_CVINFER_DLL)
  add_subdirectory(api)
endif()
