#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(CVINFER_BUILD_DLL)
    #define CVINFER_API __declspec(dllexport)
  #else
    #define CVINFER_API __declspec(dllimport)
  #endif
  #define CVINFER_CALL __cdecl
#else
  #define CVINFER_API
  #define CVINFER_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cvinfer_model cvinfer_model_t;
typedef struct cvinfer_result cvinfer_result_t;

typedef enum cvinfer_status {
    CVINFER_STATUS_OK = 0,
    CVINFER_STATUS_INVALID_ARGUMENT = 1,
    CVINFER_STATUS_RUNTIME_ERROR = 2,
    CVINFER_STATUS_UNSUPPORTED = 3,
    CVINFER_STATUS_BUFFER_TOO_SMALL = 4
} cvinfer_status_t;

typedef enum cvinfer_model_type {
    CVINFER_MODEL_YOLO_DET = 1,
    CVINFER_MODEL_YOLO_SEG = 2,
    CVINFER_MODEL_RF_DETR_DET = 3,
    CVINFER_MODEL_RF_DETR_SEG = 4
} cvinfer_model_type_t;

typedef enum cvinfer_compute_backend {
    CVINFER_BACKEND_AUTO = 0,
    CVINFER_BACKEND_CPU = 1,
    CVINFER_BACKEND_CUDA = 2
} cvinfer_compute_backend_t;
typedef struct cvinfer_model_config {
    /* <= 0: read input dimensions from the TensorRT engine. */
    int32_t input_width;
    int32_t input_height;
    int32_t class_count;
    float confidence_threshold;
    float nms_threshold;
    float mask_threshold;
    int32_t class_agnostic_nms;
    const char* input_name;
    const char* output_name;
    const char* prototype_name;
} cvinfer_model_config_t;
typedef struct cvinfer_model_config_v2 {
    /* <= 0: read input dimensions from the TensorRT engine. */
    int32_t input_width;
    int32_t input_height;
    int32_t class_count;
    float confidence_threshold;
    float nms_threshold;
    float mask_threshold;
    int32_t class_agnostic_nms;
    const char* input_name;
    const char* output_name;
    const char* prototype_name;
    int32_t preprocess_backend;
    int32_t postprocess_backend;
} cvinfer_model_config_v2_t;

typedef struct cvinfer_box {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int32_t class_id;
} cvinfer_box_t;

typedef struct cvinfer_detection {
    cvinfer_box_t box;
} cvinfer_detection_t;

typedef struct cvinfer_segmentation_info {
    cvinfer_box_t box;
    int32_t mask_width;
    int32_t mask_height;
    size_t mask_bytes;
} cvinfer_segmentation_info_t;

typedef struct cvinfer_timing {
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double total_ms;
} cvinfer_timing_t;

CVINFER_API const char* CVINFER_CALL cvinfer_last_error(void);
CVINFER_API void CVINFER_CALL cvinfer_model_config_init(cvinfer_model_config_t* config);
CVINFER_API void CVINFER_CALL cvinfer_model_config_v2_init(cvinfer_model_config_v2_t* config);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_create(
    const char* engine_path,
    cvinfer_model_type_t model_type,
    const cvinfer_model_config_t* config,
    cvinfer_model_t** model);

CVINFER_API void CVINFER_CALL cvinfer_model_destroy(cvinfer_model_t* model);
CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_create_v2(
    const char* engine_path,
    cvinfer_model_type_t model_type,
    const cvinfer_model_config_v2_t* config,
    cvinfer_model_t** model);


CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_infer_bgr(
    cvinfer_model_t* model,
    const uint8_t* bgr,
    int32_t width,
    int32_t height,
    int32_t channels,
    cvinfer_result_t** result);

CVINFER_API void CVINFER_CALL cvinfer_result_destroy(cvinfer_result_t* result);
CVINFER_API cvinfer_model_type_t CVINFER_CALL cvinfer_result_type(const cvinfer_result_t* result);
CVINFER_API size_t CVINFER_CALL cvinfer_result_count(const cvinfer_result_t* result);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_detection(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_detection_t* detection);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_segmentation(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_segmentation_info_t* segmentation);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_copy_mask(
    const cvinfer_result_t* result,
    size_t index,
    uint8_t* destination,
    size_t destination_bytes,
    size_t* copied_bytes);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_timing(
    const cvinfer_result_t* result,
    cvinfer_timing_t* timing);

#ifdef __cplusplus
}
#endif
