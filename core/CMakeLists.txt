add_library(alanxinn_cvinfer_core_common INTERFACE)
target_include_directories(alanxinn_cvinfer_core_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(tensorrt)
add_subdirectory(denglin)

function(cvinfer_add_session_target TARGET_NAME BACKEND)
  add_library(${TARGET_NAME} STATIC
    src/inference_session.cpp)
  target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tensorrt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/denglin/include)

  if(BACKEND STREQUAL "TENSORRT")
    target_link_libraries(${TARGET_NAME} PUBLIC
      alanxinn_cvinfer_core_tensorrt)
    target_compile_definitions(${TARGET_NAME} PUBLIC
      CVINFER_HAS_TENSORRT_BACKEND=1
      CVINFER_HAS_DENGLIN_BACKEND=0)
  elseif(BACKEND STREQUAL "DENGLIN")
    if(NOT BUILD_DENGLIN_BACKEND)
      message(FATAL_ERROR
        "Denglin session target requested while BUILD_DENGLIN_BACKEND=OFF")
    endif()
    target_link_libraries(${TARGET_NAME} PUBLIC
      alanxinn_cvinfer_core_denglin)
    target_compile_definitions(${TARGET_NAME} PUBLIC
      CVINFER_HAS_TENSORRT_BACKEND=0
      CVINFER_HAS_DENGLIN_BACKEND=1)
  else()
    message(FATAL_ERROR
      "cvinfer_add_session_target(${TARGET_NAME}): unsupported backend ${BACKEND}")
  endif()
endfunction()

cvinfer_add_session_target(
  alanxinn_cvinfer_core_session_tensorrt TENSORRT)

if(BUILD_DENGLIN_BACKEND)
  cvinfer_add_session_target(
    alanxinn_cvinfer_core_session_denglin DENGLIN)
else()
  # Keep the historical target name available to downstream projects without
  # requiring the proprietary SDK on TensorRT-only machines.
  add_library(alanxinn_cvinfer_core_session_denglin INTERFACE)
endif()

# Historical target name: TensorRT is the default public C++ session.
add_library(alanxinn_cvinfer_core_session ALIAS
  alanxinn_cvinfer_core_session_tensorrt)
