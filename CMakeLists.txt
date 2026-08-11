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
set(ALANXINN_CUDA_BACKEND_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/private/cuda_backend"
  CACHE PATH "Private CUDA backend source directory")
set(ALANXINN_PUBLIC_YOLO_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/yolo/include"
  CACHE PATH "Public YOLO11/YOLO26/CUDA interface include directory")
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
set(OPENCV_ROOT "" CACHE PATH "OpenCV root directory or install prefix")

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

file(GLOB CVINFER_RUNTIME_DLLS CONFIGURE_DEPENDS
  "${CVINFER_OPENCV_BIN_DIR}/opencv_world*.dll"
  "${TENSORRT_ROOT}/lib/nvinfer.dll"
  "${CUDA_ROOT}/bin/cudart64_*.dll"
  "${CUDA_ROOT}/bin/zlibwapi.dll")
list(FILTER CVINFER_RUNTIME_DLLS EXCLUDE REGEX "d\\.dll$")

file(GLOB CVINFER_DELAY_LOAD_DLLS CONFIGURE_DEPENDS
  "${CVINFER_OPENCV_BIN_DIR}/opencv_world*.dll"
  "${TENSORRT_ROOT}/lib/nvinfer.dll"
  "${CUDA_ROOT}/bin/cudart64_*.dll")
list(FILTER CVINFER_DELAY_LOAD_DLLS EXCLUDE REGEX "d\\.dll$")

function(cvinfer_copy_runtime TARGET_NAME)
  if(WIN32 AND CVINFER_RUNTIME_DLLS)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${CVINFER_RUNTIME_DLLS}
              $<TARGET_FILE_DIR:${TARGET_NAME}>
      COMMAND_EXPAND_LISTS)
  endif()
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


option(BUILD_CVINFER_DLL "Build C# compatible cvinfer DLL" ON)
option(BUILD_TRT_ONNX_BUILDER_DLL
  "Build the TensorRT C++ ONNX-to-engine builder DLL"
  ON)
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
add_subdirectory(yolo)
add_subdirectory(rf_detr)
if(BUILD_CVINFER_DLL)
  add_subdirectory(api)
endif()
