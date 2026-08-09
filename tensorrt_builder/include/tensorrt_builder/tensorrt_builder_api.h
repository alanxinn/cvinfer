#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(TENSORRT_BUILDER_BUILD_DLL)
    #define TENSORRT_BUILDER_API __declspec(dllexport)
  #else
    #define TENSORRT_BUILDER_API __declspec(dllimport)
  #endif
  #define TENSORRT_BUILDER_CALL __cdecl
#else
  #define TENSORRT_BUILDER_API
  #define TENSORRT_BUILDER_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum trt_builder_status {
    TRT_BUILDER_STATUS_OK = 0,
    TRT_BUILDER_STATUS_INVALID_ARGUMENT = 1,
    TRT_BUILDER_STATUS_IO_ERROR = 2,
    TRT_BUILDER_STATUS_BUILD_ERROR = 3,
    TRT_BUILDER_STATUS_UNSUPPORTED = 4
} trt_builder_status_t;

typedef enum trt_builder_precision {
    TRT_BUILDER_PRECISION_FP32 = 0,
    TRT_BUILDER_PRECISION_FP16 = 1,
    TRT_BUILDER_PRECISION_INT8 = 2
} trt_builder_precision_t;

typedef struct trt_builder_config {
    int32_t precision;
    int64_t workspace_mib;
    int32_t device_id;
    int32_t enable_tf32;
    int32_t enable_sparse_weights;
    int32_t strict_types;
} trt_builder_config_t;

/*
 * Each shape string uses TensorRT's compact syntax:
 *   input:1x3x640x640,mask:1x32x160x160
 * The three strings in one profile must contain the same tensor names.
 */
typedef struct trt_builder_profile {
    const char* min_shapes;
    const char* opt_shapes;
    const char* max_shapes;
} trt_builder_profile_t;

/*
 * Progress is reported as coarse build stages. TensorRT does not expose a
 * reliable percentage for buildEngineWithConfig(), so that stage reports a
 * fixed percentage plus heartbeat notifications while work is ongoing.
 */
typedef void (TENSORRT_BUILDER_CALL *trt_builder_progress_callback_t)(
    int32_t stage,
    int32_t percent,
    int32_t is_heartbeat,
    const char* message,
    void* user_data);

TENSORRT_BUILDER_API const char* TENSORRT_BUILDER_CALL trt_builder_last_error(void);

TENSORRT_BUILDER_API void TENSORRT_BUILDER_CALL trt_builder_config_init(
    trt_builder_config_t* config);

TENSORRT_BUILDER_API trt_builder_status_t TENSORRT_BUILDER_CALL trt_builder_build_onnx(
    const char* onnx_path,
    const char* engine_path,
    const trt_builder_config_t* config,
    const trt_builder_profile_t* profiles,
    size_t profile_count);

TENSORRT_BUILDER_API trt_builder_status_t TENSORRT_BUILDER_CALL
trt_builder_build_onnx_with_progress(
    const char* onnx_path,
    const char* engine_path,
    const trt_builder_config_t* config,
    const trt_builder_profile_t* profiles,
    size_t profile_count,
    trt_builder_progress_callback_t progress_callback,
    void* progress_user_data);

#ifdef __cplusplus
}
#endif
