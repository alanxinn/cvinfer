#pragma once
#include "core/tensor.h"
#include <memory>
#include <string>

namespace cvinfer::dinov3 {
struct Result {
    core::Tensor global_embedding;     // [1, D], CLS (ViT) / pooled feature (ConvNeXt)
    core::Tensor normalized_embedding; // [1, D], L2-normalized global embedding
    core::Tensor patch_tokens;         // [1, Hf*Wf, D], no CLS/register tokens
    core::Tensor feature_map;          // [1, D, Hf, Wf], row-major NCHW
};

// Batch-one fixed-resolution engines from scripts/export_dinov3.py.
// Each concurrent inference stream should use its own FeatureExtractor.
class FeatureExtractor {
public:
    explicit FeatureExtractor(const std::string& engine_path);
    ~FeatureExtractor();
    FeatureExtractor(FeatureExtractor&&) noexcept;
    FeatureExtractor& operator=(FeatureExtractor&&) noexcept;
    FeatureExtractor(const FeatureExtractor&) = delete;
    FeatureExtractor& operator=(const FeatureExtractor&) = delete;
    int input_width() const noexcept;
    int input_height() const noexcept;
    Result infer_bgr(const std::uint8_t* bgr, int width, int height, int channels = 3);
    // Already normalized FP32 RGB CHW, exactly 3*input_height*input_width values.
    Result infer_chw(const float* data, std::size_t elements);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cvinfer::dinov3
