#pragma once

#include "core/trt_core.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::yolo {
enum class ComputeBackend {
    Auto = 0,
    Cpu = 1,
    Cuda = 2
};
namespace cuda_backend { struct PreprocessContext; struct SegmentationPostprocessContext; }
struct Box { float x1{}, y1{}, x2{}, y2{}, score{}; int class_id{}; };
struct Detection { Box box; };
struct Segmentation { Box box; std::vector<std::uint8_t> mask; int mask_width{}; int mask_height{}; };
struct Keypoint { float x{}, y{}, score{}; };
struct Pose { Box box; std::vector<Keypoint> keypoints; int keypoint_dims{3}; };
struct Config {
    // 由 TensorRT engine 的输入 shape 自动填充；非正值表示未显式指定。
    int input_width{};
    int input_height{};
    // <= 0: infer the category count from the exported output shape.
    int class_count{};
    std::string input_name;
    std::string output_name;
    std::string prototype_name;
    float confidence_threshold{0.25F};
    float nms_threshold{0.45F};
    float mask_threshold{0.5F};
    bool class_agnostic_nms{false};
    int keypoint_count{17};
    int keypoint_dims{3};
    ComputeBackend preprocess_backend{ComputeBackend::Auto};
    ComputeBackend postprocess_backend{ComputeBackend::Auto};
    bool enable_timing{false};
};
struct Letterbox {
    float scale{1.0F};
    float pad_x{};
    float pad_y{};
    int source_width{};
    int source_height{};
};
struct Timing {
    double preprocess_ms{};
    double inference_ms{};
    double postprocess_ms{};
    double total_ms{};
};

bool cuda_backend_available() noexcept;

std::vector<float> preprocess_bgr(const std::uint8_t* data, int width, int height, int channels,
                                  const Config& config, Letterbox* transform = nullptr);
std::vector<Detection> nms(std::vector<Detection> candidates, float threshold, bool class_agnostic = false);
std::vector<Detection> decode_detections(const core::Tensor& output, const Config& config, int image_width, int image_height);
std::vector<Segmentation> decode_segmentations(const std::vector<core::Tensor>& outputs, const Config& config, int image_width, int image_height);
std::vector<Segmentation> decode_segmentations(const std::vector<const core::Tensor*>& outputs, const Config& config, int image_width, int image_height);
std::vector<Pose> decode_poses(const core::Tensor& output, const Config& config, int image_width, int image_height);

class Detector {
public:
    Detector(const std::string& engine_path, Config config = {});
    std::vector<Detection> infer(const std::vector<float>& chw, int image_width, int image_height);
    std::vector<Detection> infer_image(const std::uint8_t* data, int width, int height, int channels);
    void set_timing_enabled(bool enabled) noexcept { config_.enable_timing = enabled; timing_ = {}; }
    const Timing& last_timing() const noexcept { return timing_; }
private:
    std::vector<Detection> infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height);
    core::TrtSession session_;
    Config config_;
    Timing timing_;
    std::shared_ptr<cuda_backend::PreprocessContext> cuda_preprocess_;
};
class Segmentor {
public:
    Segmentor(const std::string& engine_path, Config config = {});
    std::vector<Segmentation> infer(const std::vector<float>& chw, int image_width, int image_height);
    std::vector<Segmentation> infer_image(const std::uint8_t* data, int width, int height, int channels);
    void set_timing_enabled(bool enabled) noexcept { config_.enable_timing = enabled; timing_ = {}; }
    const Timing& last_timing() const noexcept { return timing_; }
private:
    std::vector<Segmentation> infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height);
    core::TrtSession session_;
    Config config_;
    std::vector<std::string> output_names_;
    Timing timing_;
    std::shared_ptr<cuda_backend::PreprocessContext> cuda_preprocess_;
    std::shared_ptr<cuda_backend::SegmentationPostprocessContext> cuda_postprocess_;
};
class PoseEstimator {
public:
    PoseEstimator(const std::string& engine_path, Config config = {});
    std::vector<Pose> infer(const std::vector<float>& chw, int image_width, int image_height);
    std::vector<Pose> infer_image(const std::uint8_t* data, int width, int height, int channels);
    void set_timing_enabled(bool enabled) noexcept { config_.enable_timing = enabled; timing_ = {}; }
    const Timing& last_timing() const noexcept { return timing_; }
private:
    std::vector<Pose> infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height);
    core::TrtSession session_;
    Config config_;
    Timing timing_;
    std::shared_ptr<cuda_backend::PreprocessContext> cuda_preprocess_;
};
} // namespace cvinfer::yolo
