option(BUILD_WEVISONE "Build the TensorRT WeVisOne-VL adapter" ON)
if(NOT BUILD_WEVISONE)
  return()
endif()

# Both dependencies are statically linked; inference adds no Python dependency.
# FETCHCONTENT_SOURCE_DIR_* can point to offline source checkouts.
include(FetchContent)
enable_language(C)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(wevisone_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.12.0 GIT_SHALLOW TRUE)
set(PCRE2_BUILD_PCRE2_8 ON CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_16 OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2_32 OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_STATIC_PIC ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
FetchContent_Declare(wevisone_pcre2
  GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git GIT_TAG pcre2-10.46 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(wevisone_json wevisone_pcre2)

add_library(alanxinn_cvinfer_wevisone SHARED
  src/wevisone.cpp src/tokenizer.cpp src/wevisone_api.cpp)
target_include_directories(alanxinn_cvinfer_wevisone PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(alanxinn_cvinfer_wevisone PRIVATE
  alanxinn_cvinfer_core_tensorrt alanxinn_cvinfer_common nlohmann_json::nlohmann_json pcre2-8-static)
target_compile_definitions(alanxinn_cvinfer_wevisone PRIVATE CVINFER_WEVISONE_BUILD PCRE2_STATIC)
if(WIN32)
  target_link_libraries(alanxinn_cvinfer_wevisone PRIVATE normaliz)
endif()
cvinfer_copy_runtime(alanxinn_cvinfer_wevisone TENSORRT)

option(BUILD_WEVISONE_DEMO "Build the WeVisOne-VL C++ demo" ON)
if(BUILD_WEVISONE_DEMO)
  add_executable(alanxinn_wevisone_demo examples/wevisone_image_demo.cpp)
  target_link_libraries(alanxinn_wevisone_demo PRIVATE alanxinn_cvinfer_wevisone alanxinn_cvinfer_common)
  cvinfer_copy_runtime(alanxinn_wevisone_demo TENSORRT)
endif()

option(BUILD_WEVISONE_TESTS "Build official-tokenizer conformance checks" OFF)
if(BUILD_WEVISONE_TESTS)
  add_executable(wevisone_tokenizer_contract tests/tokenizer_contract.cpp src/tokenizer.cpp)
  target_include_directories(wevisone_tokenizer_contract PRIVATE src)
  target_link_libraries(wevisone_tokenizer_contract PRIVATE nlohmann_json::nlohmann_json pcre2-8-static)
  if(WIN32)
    target_link_libraries(wevisone_tokenizer_contract PRIVATE normaliz)
  endif()
endif()
