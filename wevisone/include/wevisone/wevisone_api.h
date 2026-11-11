#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(CVINFER_WEVISONE_BUILD)
#define WEVISONE_API __declspec(dllexport)
#else
#define WEVISONE_API __declspec(dllimport)
#endif
#else
#define WEVISONE_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct wevisone_model wevisone_model_t;
typedef struct wevisone_result wevisone_result_t;
typedef struct wevisone_box { float x1, y1, x2, y2; } wevisone_box_t;
typedef struct wevisone_options {
    uint32_t struct_size;
    int32_t max_new_tokens;
    float temperature, top_p;
    int32_t top_k;
    float repetition_penalty, mask_threshold;
    uint64_t seed;
    int32_t decode_pixels;
} wevisone_options_t;
typedef struct wevisone_annotation {
    wevisone_box_t box;
    float x, y;
    int32_t is_point, mask_index;
} wevisone_annotation_t;
typedef struct wevisone_mask_info {
    wevisone_box_t box;
    int32_t width, height;
    size_t alpha_elements, binary_bytes;
} wevisone_mask_info_t;
typedef struct wevisone_summary {
    int32_t count, rejected, truncated;
    double latency_ms;
    size_t annotations, masks, absent_labels;
} wevisone_summary_t;
/* UTF-8 strings. 0=success, 1=invalid argument, 2=runtime/model error,
   4=insufficient output capacity. A NULL buffer with capacity=0 queries size.
   String capacities include the trailing NUL. Each model is single-caller. */
WEVISONE_API const char* wevisone_last_error(void);
WEVISONE_API void wevisone_options_init(wevisone_options_t* options);
WEVISONE_API int wevisone_model_create(const char* directory, wevisone_model_t** model);
WEVISONE_API void wevisone_model_destroy(wevisone_model_t* model);
WEVISONE_API int wevisone_model_set_image(wevisone_model_t* model, const uint8_t* bgr, int32_t width, int32_t height, int32_t channels);
/* query_utf8: newline-separated category list, or one referring expression.
   prompt_utf8: optional exact prompt overriding the official task template. */
WEVISONE_API int wevisone_model_predict(wevisone_model_t* model, const char* task, const char* query_utf8,
    const char* prompt_utf8, const wevisone_box_t* reference_boxes, size_t reference_count,
    const wevisone_options_t* options, wevisone_result_t** result);
WEVISONE_API int wevisone_model_matte_boxes(wevisone_model_t* model, const wevisone_box_t* boxes, size_t count,
    const wevisone_options_t* options, wevisone_result_t** result);
WEVISONE_API void wevisone_result_destroy(wevisone_result_t* result);
WEVISONE_API int wevisone_result_summary(const wevisone_result_t* result, wevisone_summary_t* summary);
/* field: 0=raw response, 1=prompt, 2=annotation label, 3=absent label. */
WEVISONE_API int wevisone_result_copy_text(const wevisone_result_t* result, int32_t field, size_t index,
    char* destination, size_t capacity, size_t* required);
WEVISONE_API int wevisone_result_annotation(const wevisone_result_t* result, size_t index, wevisone_annotation_t* annotation);
WEVISONE_API int wevisone_result_mask_info(const wevisone_result_t* result, size_t index, wevisone_mask_info_t* info);
WEVISONE_API int wevisone_result_copy_alpha(const wevisone_result_t* result, size_t index, float* destination, size_t capacity, size_t* required);
WEVISONE_API int wevisone_result_copy_binary(const wevisone_result_t* result, size_t index, uint8_t* destination, size_t capacity, size_t* required);
#ifdef __cplusplus
}
#endif
