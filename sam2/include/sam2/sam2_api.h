#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(CVINFER_SAM2_BUILD)
#define SAM2_API __declspec(dllexport)
#else
#define SAM2_API __declspec(dllimport)
#endif
#else
#define SAM2_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sam2_model sam2_model_t;
typedef struct sam2_result sam2_result_t;
typedef struct sam2_mask_info {
    float x1, y1, x2, y2, score;
    int32_t width, height;
    size_t mask_bytes;
} sam2_mask_info_t;
/* 0=success, 1=invalid argument, 2=runtime/model error. Errors are thread-local. */
SAM2_API const char* sam2_last_error(void);
SAM2_API int sam2_model_create(const char* directory, float mask_threshold,
                              int32_t max_points, sam2_model_t** model);
SAM2_API void sam2_model_destroy(sam2_model_t* model);
SAM2_API int sam2_model_set_image(sam2_model_t* model, const uint8_t* bgr, int32_t width, int32_t height, int32_t channels);
SAM2_API int sam2_model_set_box(sam2_model_t* model, float x1, float y1, float x2, float y2);
SAM2_API int sam2_model_add_point(sam2_model_t* model, float x, float y, int32_t positive);
SAM2_API int sam2_model_clear_box(sam2_model_t* model);
SAM2_API int sam2_model_clear_prompts(sam2_model_t* model);
SAM2_API int sam2_model_remove_point(sam2_model_t* model, size_t index);
SAM2_API int sam2_model_clear_points(sam2_model_t* model);
SAM2_API int sam2_model_predict(sam2_model_t* model, int32_t use_previous_mask, sam2_result_t** result);
SAM2_API void sam2_result_destroy(sam2_result_t* result);
SAM2_API size_t sam2_result_count(const sam2_result_t* result);
SAM2_API int sam2_result_get_mask(const sam2_result_t* result, size_t index, sam2_mask_info_t* info);
SAM2_API int sam2_result_copy_mask(const sam2_result_t* result, size_t index, uint8_t* data,
                                  size_t capacity, size_t* copied);
#ifdef __cplusplus
}
#endif
