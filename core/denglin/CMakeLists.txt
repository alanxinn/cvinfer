if(NOT BUILD_DENGLIN_BACKEND)
  # Keep the target name available to downstream projects without requiring
  # the proprietary SDK on TensorRT-only machines.
  add_library(alanxinn_cvinfer_core_denglin INTERFACE)
  return()
endif()

if(NOT DENGLIN_SDK_ROOT OR
   NOT EXISTS "${DENGLIN_SDK_ROOT}/include/dlnne/dlnne.h")
  message(FATAL_ERROR
    "Denglin SDK was not found. Automatic detection checks DENGLIN_SDK_ROOT, "
    "DENGLIN_ROOT, DLGPU_ROOT, DLGPU_HOME and DLICC_PATH, plus common "
    "install locations. Set DENGLIN_SDK_ROOT to the directory containing "
    "include/dlnne/dlnne.h.")
endif()

add_library(alanxinn_cvinfer_core_denglin STATIC
  src/denglin_core.cpp)

target_include_directories(alanxinn_cvinfer_core_denglin PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  "${DENGLIN_SDK_ROOT}/include"
  "${DENGLIN_SDK_ROOT}/include/dlnne")

target_link_directories(alanxinn_cvinfer_core_denglin PUBLIC
  "${DENGLIN_SDK_ROOT}/lib")

target_link_libraries(alanxinn_cvinfer_core_denglin PUBLIC
  alanxinn_cvinfer_core_common
  curt
  dlnne)

set(CVINFER_DENGLIN_SDK_VERSION "unknown")
if(EXISTS "${DENGLIN_SDK_ROOT}/version.txt")
  file(READ "${DENGLIN_SDK_ROOT}/version.txt" CVINFER_DENGLIN_SDK_VERSION)
  string(STRIP "${CVINFER_DENGLIN_SDK_VERSION}" CVINFER_DENGLIN_SDK_VERSION)
endif()
target_compile_definitions(alanxinn_cvinfer_core_denglin PRIVATE
  CVINFER_DENGLIN_SDK_VERSION="${CVINFER_DENGLIN_SDK_VERSION}")

option(BUILD_DENGLIN_SMOKE_TEST
  "Build the Denglin Core ONNX/Host/Device inference smoke test"
  OFF)
if(BUILD_DENGLIN_SMOKE_TEST)
  add_executable(alanxinn_cvinfer_denglin_smoke
    tests/denglin_core_smoke.cpp)
  target_link_libraries(alanxinn_cvinfer_denglin_smoke PRIVATE
    alanxinn_cvinfer_core_denglin)
endif()
