add_library(alanxinn_cvinfer SHARED src/cvinfer_api.cpp)
set_target_properties(alanxinn_cvinfer PROPERTIES
  OUTPUT_NAME "alanxinn_cvinfer"
  ARCHIVE_OUTPUT_NAME "alanxinn_cvinfer"
  RUNTIME_OUTPUT_NAME "alanxinn_cvinfer")
target_compile_definitions(alanxinn_cvinfer PRIVATE CVINFER_BUILD_DLL)
target_include_directories(alanxinn_cvinfer
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/yolo/include
          ${CMAKE_SOURCE_DIR}/rf_detr/include
)
target_link_libraries(alanxinn_cvinfer PRIVATE alanxinn_cvinfer_yolo alanxinn_cvinfer_rf_detr)

cvinfer_enable_delay_load(alanxinn_cvinfer)
cvinfer_copy_runtime(alanxinn_cvinfer)
