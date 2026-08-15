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
    CVINFER_MODEL_YOLO11_DET = 1,
    CVINFER_MODEL_YOLO11_SEG = 2,
    CVINFER_MODEL_RF_DETR_DET = 3,
    CVINFER_MODEL_RF_DETR_SEG = 4,
    CVINFER_MODEL_YOLO11_POSE = 5,
    CVINFER_MODEL_YOLO26_DET = 6,
    CVINFER_MODEL_YOLO26_SEG = 7,
    CVINFER_MODEL_YOLO26_POSE = 8
} cvinfer_model_type_t;

typedef enum cvinfer_compute_backend {
    CVINFER_BACKEND_AUTO = 0,
    CVINFER_BACKEND_CPU = 1,
    CVINFER_BACKEND_CUDA = 2
} cvinfer_compute_backend_t;

/* Inference engine backend used to load and execute the model. */
typedef enum cvinfer_inference_backend {
    CVINFER_INFERENCE_BACKEND_TENSORRT = 0,
    CVINFER_INFERENCE_BACKEND_DENGLIN = 1
} cvinfer_inference_backend_t;
typedef struct cvinfer_model_config {
    /* <= 0: read input dimensions from the selected inference session. */
    int32_t input_width;
    int32_t input_height;
    /* <= 0: infer category count from the selected output shape. */
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
    /* <= 0: read input dimensions from the selected inference session. */
    int32_t input_width;
    int32_t input_height;
    /* <= 0: infer category count from the selected output shape. */
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
    /* Pose output layout; defaults to COCO [17,3]. */
    int32_t keypoint_count;
    int32_t keypoint_dims;
    /* Defaults to TensorRT; select Denglin when the native backend is built. */
    int32_t inference_backend;
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

typedef struct cvinfer_keypoint {
    float x;
    float y;
    float score;
} cvinfer_keypoint_t;

typedef struct cvinfer_pose_info {
    cvinfer_box_t box;
    int32_t keypoint_count;
    int32_t keypoint_dims;
    size_t keypoint_bytes;
} cvinfer_pose_info_t;

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
/* Timing is disabled by default; pass enabled != 0 to collect inference timing. */
CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_set_timing_enabled(
    cvinfer_model_t* model,
    int32_t enabled);
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

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_pose(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_pose_info_t* pose);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_copy_keypoints(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_keypoint_t* destination,
    size_t destination_bytes,
    size_t* copied_bytes);

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_timing(
    const cvinfer_result_t* result,
    cvinfer_timing_t* timing);

#ifdef __cplusplus
}
#endif
