#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#if defined(CVINFER_HANDS_BUILD)
#define HANDS_API __declspec(dllexport)
#else
#define HANDS_API __declspec(dllimport)
#endif
#else
#define HANDS_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct hands_model hands_model_t;
typedef struct hands_config {
    float palm_threshold;
    float hand_threshold;
    float nms_threshold;
    int32_t max_hands;
} hands_config_t;
typedef struct hands_hand {
    float box[4];          /* Original image pixels: x1,y1,x2,y2. */
    float palm_score;
    float confidence;
    float handedness;      /* Model probability: 0=Left, 1=Right; no implicit mirror. */
    float landmarks[63];   /* MediaPipe order, 21 xyz triples; xy in image pixels,
                             z relative to wrist in the same scale as x. */
    float world_landmarks[63]; /* Rotated xyz in metres, model hand-centred origin. */
} hands_hand_t;
/* 0=success, 1=invalid argument, 2=runtime/model error, 4=buffer too small.
 * Paths are UTF-8. One model instance per concurrent inference stream.
 * Native inference loads engines; scripts/prepare_mediapipe_hands.py downloads
 * and builds them automatically. Config may be NULL to select defaults.
 */
HANDS_API const char* hands_last_error(void);
HANDS_API void hands_config_init(hands_config_t* config);
HANDS_API int hands_model_create(const char* palm_engine, const char* hand_engine,
                                 const hands_config_t* config, hands_model_t** model);
HANDS_API void hands_model_destroy(hands_model_t* model);
/* Contiguous HWC uint8 BGR, exactly 3 channels. Caller owns all output memory.
 * count receives the required number of hands, including on status 4. A NULL
 * output is allowed only with capacity=0. No partial output on status 4.
 */
HANDS_API int hands_model_infer_bgr(hands_model_t* model, const uint8_t* bgr,
    int32_t width, int32_t height, int32_t channels,
    hands_hand_t* output, size_t capacity, size_t* count);
#ifdef __cplusplus
}
#endif
