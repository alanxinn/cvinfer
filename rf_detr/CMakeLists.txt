if(TARGET alanxinn_cvinfer_cuda)
  set(CVINFER_RF_DETR_HAS_CUDA_BACKEND ON)
else()
  set(CVINFER_RF_DETR_HAS_CUDA_BACKEND OFF)
endif()

function(cvinfer_add_rf_detr_target TARGET_NAME SESSION_TARGET ENABLE_CUDA)
  set(_accelerator alanxinn_cvinfer_cuda)
  if(ARGC GREATER 3)
    set(_accelerator "${ARGV3}")
  endif()
  set(_sources src/rf_detr.cpp)
  if(NOT ENABLE_CUDA)
    list(APPEND _sources src/rf_detr_cuda_stub.cpp)
  endif()

  add_library(${TARGET_NAME} STATIC ${_sources})
  target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${TARGET_NAME} PUBLIC ${SESSION_TARGET})

  if(ENABLE_CUDA)
    if(_accelerator STREQUAL "alanxinn_cvinfer_denglin_accel")
      target_compile_definitions(${TARGET_NAME} PRIVATE CVINFER_HAS_DENGLIN_ACCEL=1)
    endif()
    target_compile_definitions(${TARGET_NAME} PUBLIC
      CVINFER_HAS_CUDA_BACKEND=1)
    target_link_libraries(${TARGET_NAME} PUBLIC
      ${_accelerator})
    if(MSVC)
      target_link_libraries(${TARGET_NAME} INTERFACE delayimp)
      set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "/DELAYLOAD:${_accelerator}.dll")
    endif()
  else()
    # Keep Denglin completely independent from the NVIDIA CUDA backend.
    target_compile_definitions(${TARGET_NAME} PRIVATE
      CVINFER_RF_DETR_CUDA_STATIC
      CVINFER_HAS_CUDA_BACKEND=0)
  endif()
endfunction()

cvinfer_add_rf_detr_target(
  alanxinn_cvinfer_rf_detr_tensorrt
  alanxinn_cvinfer_core_session_tensorrt
  ${CVINFER_RF_DETR_HAS_CUDA_BACKEND})

if(BUILD_DENGLIN_BACKEND)
  set(_denglin_accel OFF)
  if(TARGET alanxinn_cvinfer_denglin_accel)
    set(_denglin_accel ON)
  endif()
  cvinfer_add_rf_detr_target(
    alanxinn_cvinfer_rf_detr_denglin
    alanxinn_cvinfer_core_session_denglin
    ${_denglin_accel} alanxinn_cvinfer_denglin_accel)
endif()

# Historical target name: the default model adapter is TensorRT.
add_library(alanxinn_cvinfer_rf_detr ALIAS alanxinn_cvinfer_rf_detr_tensorrt)
