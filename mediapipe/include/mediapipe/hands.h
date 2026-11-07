#pragma once
#include "mediapipe/hands_api.h"
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::mediapipe {
// Fixed batch-one FP32 NHWC IO engines from prepare_mediapipe_hands.py.
class HANDS_API HandDetector {
public:
    HandDetector(const std::string& palm_engine, const std::string& hand_engine,
                 hands_config_t config = {.5F, .5F, .3F, 2});
    ~HandDetector();
    HandDetector(HandDetector&&) noexcept;
    HandDetector& operator=(HandDetector&&) noexcept;
    HandDetector(const HandDetector&) = delete;
    HandDetector& operator=(const HandDetector&) = delete;
    std::vector<hands_hand_t> infer_bgr(const uint8_t* bgr, int width, int height, int channels = 3);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cvinfer::mediapipe
