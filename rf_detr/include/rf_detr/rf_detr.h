#pragma once

#include "core/trt_core.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::rf_detr {

enum class ComputeBackend {
    Auto = 0,
    Cpu = 1,
    Cuda = 2
};

struct Box { float x1{}, y1{}, x2{}, y2{}, score{}; int class_id{}; };
struct Detection { Box box; };
struct Segmentation {
    Box box;
    std::vector<std::uint8_t> mask;
    int mask_width{};
    int mask_height{};
};
struct Config {
    int input_width{};
    int input_height{};
    // <= 0: infer the category count from the exported output shape.
    int class_count{};
    std::string input_name;
    // Optional output name. The second output is discovered by shape when empty.
    std::string output_name;
    float confidence_threshold{0.25F};
    float nms_threshold{0.45F};
    float mask_threshold{0.5F};
    bool class_agnostic_nms{false};
    // RF-DETR uses ImageNet normalization after RGB conversion.
    bool letterbox{false};
    ComputeBackend preprocess_backend{ComputeBackend::Auto};
    ComputeBackend postprocess_backend{ComputeBackend::Auto};
    bool enable_timing{false};
};

struct Timing {
    double preprocess_ms{};
    double inference_ms{};
    double postprocess_ms{};
    double total_ms{};
};

std::vector<float> preprocess_bgr(const std::uint8_t* data, int width, int height, int channels,
                                  const Config& config);
std::vector<Detection> decode_detections(const core::Tensor& boxes, const core::Tensor& logits,
                                         const Config& config, int image_width, int image_height);
std::vector<Segmentation> decode_segmentations(const std::vector<core::Tensor>& outputs,
                                                const Config& config, int image_width, int image_height);
std::vector<Detection> nms(std::vector<Detection> candidates, float threshold, bool class_agnostic = false);

class Detector {
public:
    Detector(const std::string& engine_path, Config config = {});
    std::vector<Detection> infer(const std::vector<float>& chw, int image_width, int image_height);
    std::vector<Detection> infer_image(const std::uint8_t* data, int width, int height, int channels);
    void set_timing_enabled(bool enabled) noexcept { config_.enable_timing = enabled; timing_ = {}; }
    const Timing& last_timing() const noexcept { return timing_; }

private:
    core::TrtSession session_;
    Config config_;
    Timing timing_;
};

class Segmentor {
public:
    Segmentor(const std::string& engine_path, Config config = {});
    std::vector<Segmentation> infer(const std::vector<float>& chw, int image_width, int image_height);
    std::vector<Segmentation> infer_image(const std::uint8_t* data, int width, int height, int channels);
    void set_timing_enabled(bool enabled) noexcept { config_.enable_timing = enabled; timing_ = {}; }
    const Timing& last_timing() const noexcept { return timing_; }

private:
    core::TrtSession session_;
    Config config_;
    Timing timing_;
};

} // namespace cvinfer::rf_detr
