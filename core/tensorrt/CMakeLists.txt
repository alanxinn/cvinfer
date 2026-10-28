add_library(alanxinn_cvinfer_core_tensorrt STATIC src/trt_core.cpp)
target_include_directories(alanxinn_cvinfer_core_tensorrt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(alanxinn_cvinfer_core_tensorrt PUBLIC
  alanxinn_cvinfer_core_common
  alanxinn_cvinfer_common)

# Keep the historical target name available to downstream CMake projects.
add_library(alanxinn_cvinfer_core ALIAS alanxinn_cvinfer_core_tensorrt)

option(BUILD_TRT_VALIDATION_TOOLS "Build TensorRT raw output validation tools" ON)
if(BUILD_TRT_VALIDATION_TOOLS)
  add_executable(trt_raw_outputs tests/trt_raw_outputs.cpp)
  target_link_libraries(trt_raw_outputs PRIVATE alanxinn_cvinfer_core_tensorrt)
  cvinfer_copy_runtime(trt_raw_outputs TENSORRT)
  add_executable(trt_session_contract tests/trt_session_contract.cpp)
  target_link_libraries(trt_session_contract PRIVATE alanxinn_cvinfer_core_tensorrt)
  cvinfer_copy_runtime(trt_session_contract TENSORRT)
endif()
