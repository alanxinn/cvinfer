#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(CVINFER_SAM3_BUILD)
#define SAM3_API __declspec(dllexport)
#else
#define SAM3_API __declspec(dllimport)
#endif
#else
#define SAM3_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct sam3_model sam3_model_t;
typedef struct sam3_result sam3_result_t;
typedef struct sam3_mask_info {
    float x1, y1, x2, y2, score;
    int32_t width, height;
    size_t mask_bytes;
} sam3_mask_info_t;
/* 0=success, 1=invalid argument, 2=runtime/model error. Errors are thread-local. */
SAM3_API const char* sam3_last_error(void);
SAM3_API int sam3_model_create(const char* directory, float confidence, float mask_threshold,
                              int32_t max_points, sam3_model_t** model);
SAM3_API void sam3_model_destroy(sam3_model_t* model);
SAM3_API int sam3_model_set_image(sam3_model_t* model, const uint8_t* bgr, int32_t width, int32_t height, int32_t channels);
SAM3_API int sam3_model_set_box(sam3_model_t* model, float x1, float y1, float x2, float y2);
SAM3_API int sam3_model_add_point(sam3_model_t* model, float x, float y, int32_t positive);
SAM3_API int sam3_model_clear_points(sam3_model_t* model);
SAM3_API int sam3_model_predict(sam3_model_t* model, sam3_result_t** result);
SAM3_API int sam3_model_predict_text(sam3_model_t* model, const char* text, sam3_result_t** result);
SAM3_API int sam3_model_predict_tokens(sam3_model_t* model, const int64_t* ids, const int64_t* attention,
                                      size_t length, sam3_result_t** result);
SAM3_API void sam3_result_destroy(sam3_result_t* result);
SAM3_API size_t sam3_result_count(const sam3_result_t* result);
SAM3_API int sam3_result_get_mask(const sam3_result_t* result, size_t index, sam3_mask_info_t* info);
SAM3_API int sam3_result_copy_mask(const sam3_result_t* result, size_t index, uint8_t* data,
                                  size_t capacity, size_t* copied);
#ifdef __cplusplus
}
#endif
