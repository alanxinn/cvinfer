if(TARGET alanxinn_cvinfer_cuda)
  set(CVINFER_YOLO_HAS_CUDA_BACKEND ON)
else()
  set(CVINFER_YOLO_HAS_CUDA_BACKEND OFF)
endif()

function(cvinfer_add_yolo_target TARGET_NAME SESSION_TARGET ENABLE_CUDA)
  set(_sources src/yolo.cpp src/yolo_cuda_loader.cpp)
  if(NOT ENABLE_CUDA)
    list(APPEND _sources src/yolo_cuda_stub.cpp)
  endif()

  add_library(${TARGET_NAME} STATIC ${_sources})
  target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${TARGET_NAME} PUBLIC ${SESSION_TARGET})

  if(ENABLE_CUDA)
    target_compile_definitions(${TARGET_NAME} PUBLIC
      CVINFER_HAS_CUDA_BACKEND=1)
    target_link_libraries(${TARGET_NAME} PUBLIC
      alanxinn_cvinfer_cuda)
    if(MSVC)
      target_link_libraries(${TARGET_NAME} INTERFACE delayimp)
      set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "/DELAYLOAD:alanxinn_cvinfer_cuda.dll")
    endif()
  else()
    # The Denglin native DLL must not link the NVIDIA CUDA preprocessing
    # backend. Its device pointers belong to the Denglin SDK runtime.
    target_compile_definitions(${TARGET_NAME} PRIVATE CVINFER_CUDA_STATIC)
  endif()
endfunction()

cvinfer_add_yolo_target(
  alanxinn_cvinfer_yolo_tensorrt
  alanxinn_cvinfer_core_session_tensorrt
  ${CVINFER_YOLO_HAS_CUDA_BACKEND})

if(BUILD_DENGLIN_BACKEND)
  cvinfer_add_yolo_target(
    alanxinn_cvinfer_yolo_denglin
    alanxinn_cvinfer_core_session_denglin
    OFF)
endif()

# Historical target name: the default model adapter is TensorRT.
add_library(alanxinn_cvinfer_yolo ALIAS alanxinn_cvinfer_yolo_tensorrt)

option(BUILD_YOLO_UNIFIED_DEMOS "Build demos for the unified YOLO11 adapters" ON)
if(BUILD_YOLO_UNIFIED_DEMOS)
  add_executable(alanxinn_yolo11_det_core_demo examples/yolo11_det_core_demo.cpp)
  target_link_libraries(alanxinn_yolo11_det_core_demo PRIVATE
    alanxinn_cvinfer_yolo_tensorrt alanxinn_cvinfer_common)
  cvinfer_enable_delay_load(alanxinn_yolo11_det_core_demo)
  cvinfer_copy_runtime(alanxinn_yolo11_det_core_demo TENSORRT)

  add_executable(alanxinn_yolo11_seg_core_demo examples/yolo11_seg_core_demo.cpp)
  target_link_libraries(alanxinn_yolo11_seg_core_demo PRIVATE
    alanxinn_cvinfer_yolo_tensorrt alanxinn_cvinfer_common)
  cvinfer_enable_delay_load(alanxinn_yolo11_seg_core_demo)
  cvinfer_copy_runtime(alanxinn_yolo11_seg_core_demo TENSORRT)
endif()
