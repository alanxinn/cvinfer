#include "dinov3/dinov3.h"
#include "core/trt_core.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace cvinfer::dinov3 {
namespace {
// DINOv3's official processor applies rescale -> antialiased bilinear resize
// -> ImageNet normalization. Keep floating values throughout; resizing uint8
// first, as in SAM3, would quantize away information here.
std::vector<float> preprocess(const std::uint8_t* data, int width, int height, int out_w, int out_h) {
    struct Filter { int first; std::vector<float> weights; };
    const auto filters = [](int source, int destination) {
        std::vector<Filter> result;
        const float scale = float(source) / destination, support = std::max(1.F, scale);
        for (int i = 0; i < destination; ++i) {
            const float center = (i + .5F) * scale;
            const int first = std::max(0, int(std::ceil(center - support - .5F)));
            const int last = std::min(source - 1, int(std::floor(center + support - .5F)));
            Filter f{first, {}};
            float sum = 0;
            for (int j = first; j <= last; ++j) {
                const float weight = std::max(0.F, 1.F - std::abs((j + .5F - center) / support));
                f.weights.push_back(weight); sum += weight;
            }
            if (sum <= 0) throw std::runtime_error("Invalid DINOv3 resize filter");
            for (auto& weight : f.weights) weight /= sum;
            result.push_back(std::move(f));
        }
        return result;
    };
    const auto xs = filters(width, out_w), ys = filters(height, out_h);
    std::vector<float> horizontal(static_cast<std::size_t>(height) * out_w * 3);
    for (int y = 0; y < height; ++y) for (int x = 0; x < out_w; ++x) for (int c = 0; c < 3; ++c) {
        float value = 0;
        for (std::size_t k = 0; k < xs[x].weights.size(); ++k)
            value += xs[x].weights[k] * (data[(static_cast<std::size_t>(y) * width + xs[x].first + k) * 3 + 2 - c] / 255.F);
        horizontal[(static_cast<std::size_t>(y) * out_w + x) * 3 + c] = value;
    }
    constexpr float mean[] = {.485F, .456F, .406F}, stddev[] = {.229F, .224F, .225F};
    const auto plane = static_cast<std::size_t>(out_h) * out_w;
    std::vector<float> chw(plane * 3);
    for (int y = 0; y < out_h; ++y) for (int x = 0; x < out_w; ++x) for (int c = 0; c < 3; ++c) {
        float value = 0;
        for (std::size_t k = 0; k < ys[y].weights.size(); ++k)
            value += ys[y].weights[k] * horizontal[(static_cast<std::size_t>(ys[y].first) + k) * out_w * 3 + x * 3 + c];
        chw[c * plane + static_cast<std::size_t>(y) * out_w + x] = (value - mean[c]) / stddev[c];
    }
    return chw;
}
void validate_tensor(const core::Tensor& tensor) {
    if (tensor.data_type != core::TensorDataType::Float32 || tensor.data.empty() || tensor.data.size() % sizeof(float))
        throw std::runtime_error("DINOv3 requires non-empty FP32 output: " + tensor.name);
    const auto* values = reinterpret_cast<const float*>(tensor.data.data());
    for (std::size_t i = 0; i < tensor.data.size() / sizeof(float); ++i)
        if (!std::isfinite(values[i])) throw std::runtime_error("Non-finite DINOv3 features; verify engine precision: " + tensor.name);
}
}

struct FeatureExtractor::Impl {
    core::TrtSession session;
    int width{}, height{};
    explicit Impl(const std::string& path) : session(core::TrtSession::load(path)) {
        if (session.input_names() != std::vector<std::string>{"pixel_values"})
            throw std::runtime_error("DINOv3 engine must have one input named pixel_values");
        const auto dims = session.tensor_shape("pixel_values");
        if (dims.size() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] <= 0 || dims[3] <= 0 ||
            dims[2] > std::numeric_limits<int>::max() || dims[3] > std::numeric_limits<int>::max())
            throw std::runtime_error("DINOv3 requires a resolved [1,3,H,W] input shape");
        height = static_cast<int>(dims[2]); width = static_cast<int>(dims[3]);
        const auto outputs = session.output_names();
        for (const auto* name : {"global_embedding", "patch_tokens", "feature_map"})
            if (std::find(outputs.begin(), outputs.end(), name) == outputs.end())
                throw std::runtime_error(std::string("DINOv3 engine output is missing: ") + name);
    }
};
FeatureExtractor::FeatureExtractor(const std::string& path) : impl_(std::make_unique<Impl>(path)) {}
FeatureExtractor::~FeatureExtractor() = default;
FeatureExtractor::FeatureExtractor(FeatureExtractor&&) noexcept = default;
FeatureExtractor& FeatureExtractor::operator=(FeatureExtractor&&) noexcept = default;
int FeatureExtractor::input_width() const noexcept { return impl_ ? impl_->width : 0; }
int FeatureExtractor::input_height() const noexcept { return impl_ ? impl_->height : 0; }
Result FeatureExtractor::infer_bgr(const std::uint8_t* bgr, int width, int height, int channels) {
    if (!impl_) throw std::runtime_error("DINOv3 session is not initialized");
    if (!bgr || width <= 0 || height <= 0 || channels != 3)
        throw std::invalid_argument("DINOv3 requires a contiguous 3-channel BGR image");
    const auto input = preprocess(bgr, width, height, impl_->width, impl_->height);
    return infer_chw(input.data(), input.size());
}
Result FeatureExtractor::infer_chw(const float* values, std::size_t elements) {
    if (!impl_) throw std::runtime_error("DINOv3 session is not initialized");
    const auto expected = static_cast<std::size_t>(impl_->width) * impl_->height * 3;
    if (!values || elements != expected) throw std::invalid_argument("DINOv3 CHW input size does not match the engine");
    for (std::size_t i = 0; i < elements; ++i)
        if (!std::isfinite(values[i])) throw std::invalid_argument("DINOv3 CHW input contains non-finite values");
    impl_->session.set_shape("pixel_values", {1, 3, impl_->height, impl_->width});
    impl_->session.set_input("pixel_values", values, elements * sizeof(float));
    impl_->session.infer();
    Result result;
    result.global_embedding = impl_->session.get_output("global_embedding");
    result.patch_tokens = impl_->session.get_output("patch_tokens");
    result.feature_map = impl_->session.get_output("feature_map");
    for (const auto* t : {&result.global_embedding, &result.patch_tokens, &result.feature_map}) validate_tensor(*t);
    const auto& g = result.global_embedding.shape;
    const auto& p = result.patch_tokens.shape;
    const auto& f = result.feature_map.shape;
    if (g.size() != 2 || g[0] != 1 || p.size() != 3 || p[0] != 1 || f.size() != 4 || f[0] != 1 ||
        g[1] != p[2] || g[1] != f[1] || p[1] != f[2] * f[3])
        throw std::runtime_error("DINOv3 output shapes are inconsistent");
    result.normalized_embedding = result.global_embedding;
    result.normalized_embedding.name = "normalized_embedding";
    const auto* raw = reinterpret_cast<const float*>(result.global_embedding.data.data());
    auto* normalized = reinterpret_cast<float*>(result.normalized_embedding.data.data());
    double norm = 0;
    for (int64_t i = 0; i < g[1]; ++i) norm += double(raw[i]) * raw[i];
    // Match torch.nn.functional.normalize: a zero vector stays zero. The
    // raw feature extraction is valid even when cosine similarity is undefined.
    norm = std::max(std::sqrt(norm), 1e-12);
    for (int64_t i = 0; i < g[1]; ++i) normalized[i] = static_cast<float>(raw[i] / norm);
    return result;
}
} // namespace cvinfer::dinov3
