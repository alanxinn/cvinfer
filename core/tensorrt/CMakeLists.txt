add_library(alanxinn_cvinfer_core_tensorrt STATIC src/trt_core.cpp)
target_include_directories(alanxinn_cvinfer_core_tensorrt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(alanxinn_cvinfer_core_tensorrt PUBLIC
  alanxinn_cvinfer_core_common
  alanxinn_cvinfer_common)

# Keep the historical target name available to downstream CMake projects.
add_library(alanxinn_cvinfer_core ALIAS alanxinn_cvinfer_core_tensorrt)
