#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(CVINFER_DINOV3_BUILD)
#define DINOV3_API __declspec(dllexport)
#else
#define DINOV3_API __declspec(dllimport)
#endif
#else
#define DINOV3_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct dinov3_model dinov3_model_t;
typedef struct dinov3_result dinov3_result_t;
typedef struct dinov3_tensor_info {
    int32_t rank;
    int64_t shape[8];
    size_t elements;
} dinov3_tensor_info_t;
/* 0=success, 1=invalid argument, 2=runtime/model error, 4=buffer too small.
 * All tensors use FP32. Capacity and copied are FLOAT ELEMENT counts, not bytes.
 * Tensor names: global_embedding, normalized_embedding, patch_tokens, feature_map.
 */
DINOV3_API const char* dinov3_last_error(void);
DINOV3_API int dinov3_model_create(const char* engine_path, dinov3_model_t** model);
DINOV3_API void dinov3_model_destroy(dinov3_model_t* model);
DINOV3_API int dinov3_model_input_size(const dinov3_model_t* model, int32_t* width, int32_t* height);
DINOV3_API int dinov3_model_infer_bgr(dinov3_model_t* model, const uint8_t* bgr, int32_t width,
                                    int32_t height, int32_t channels, dinov3_result_t** result);
DINOV3_API int dinov3_model_infer_chw(dinov3_model_t* model, const float* values, size_t elements,
                                    dinov3_result_t** result);
DINOV3_API void dinov3_result_destroy(dinov3_result_t* result);
DINOV3_API int dinov3_result_tensor_info(const dinov3_result_t* result, const char* name, dinov3_tensor_info_t* info);
DINOV3_API int dinov3_result_copy_tensor(const dinov3_result_t* result, const char* name, float* destination,
                                       size_t capacity, size_t* copied);
#ifdef __cplusplus
}
#endif
