function(cvinfer_add_native_api TARGET_NAME OUTPUT_NAME BACKEND)
  add_library(${TARGET_NAME} SHARED src/cvinfer_api.cpp)
  set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${OUTPUT_NAME}"
    ARCHIVE_OUTPUT_NAME "${OUTPUT_NAME}"
    RUNTIME_OUTPUT_NAME "${OUTPUT_NAME}")

  target_compile_definitions(${TARGET_NAME} PRIVATE
    CVINFER_BUILD_DLL)
  target_include_directories(${TARGET_NAME}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/core/include
            ${CMAKE_SOURCE_DIR}/yolo/include
            ${CMAKE_SOURCE_DIR}/rf_detr/include)

  if(BACKEND STREQUAL "TENSORRT")
    target_compile_definitions(${TARGET_NAME} PRIVATE
      CVINFER_NATIVE_BACKEND_TENSORRT=1
      CVINFER_HAS_TENSORRT_BACKEND=1
      CVINFER_HAS_DENGLIN_BACKEND=0)
    target_link_libraries(${TARGET_NAME} PRIVATE
      alanxinn_cvinfer_yolo_tensorrt
      alanxinn_cvinfer_rf_detr_tensorrt)
    cvinfer_enable_delay_load(${TARGET_NAME})
    cvinfer_copy_runtime(${TARGET_NAME} TENSORRT)
  elseif(BACKEND STREQUAL "DENGLIN")
    if(NOT BUILD_DENGLIN_BACKEND)
      message(FATAL_ERROR
        "Denglin native API requested while BUILD_DENGLIN_BACKEND=OFF")
    endif()
    target_compile_definitions(${TARGET_NAME} PRIVATE
      CVINFER_NATIVE_BACKEND_DENGLIN=1
      CVINFER_HAS_TENSORRT_BACKEND=0
      CVINFER_HAS_DENGLIN_BACKEND=1)
    target_link_libraries(${TARGET_NAME} PRIVATE
      alanxinn_cvinfer_yolo_denglin
      alanxinn_cvinfer_rf_detr_denglin)
    # Do not apply the TensorRT delay-load list here. This DLL must not import
    # nvinfer/cudart; Denglin owns its device runtime through curt.dll.
    cvinfer_copy_runtime(${TARGET_NAME} DENGLIN)
  else()
    message(FATAL_ERROR
      "cvinfer_add_native_api(${TARGET_NAME}): unsupported backend ${BACKEND}")
  endif()
endfunction()

cvinfer_add_native_api(
  alanxinn_cvinfer_tensorrt
  alanxinn_cvinfer_trt
  TENSORRT)

if(BUILD_DENGLIN_BACKEND)
  cvinfer_add_native_api(
    alanxinn_cvinfer_denglin
    alanxinn_cvinfer_denglin
    DENGLIN)
endif()

# Historical CMake target name: TensorRT is the default public native API.
add_library(alanxinn_cvinfer ALIAS alanxinn_cvinfer_tensorrt)
