#include "sam2/sam2.h"
#include "core/trt_core.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cvinfer::sam2 {
namespace {
using core::Tensor;
using core::TrtSession;
using Tensors = std::map<std::string, Tensor>;

template<class T> Tensor tensor(std::string name, std::vector<int64_t> shape,
                               const std::vector<T>& data, core::TensorDataType type) {
    Tensor t;
    t.name = std::move(name); t.shape = std::move(shape); t.data_type = type;
    t.data.resize(data.size() * sizeof(T));
    if (!data.empty()) std::memcpy(t.data.data(), data.data(), t.data.size());
    return t;
}
const float* floats(const Tensor& t) {
    std::size_t count = 1;
    for (auto d : t.shape) {
        if (d <= 0 || count > std::numeric_limits<std::size_t>::max() / sizeof(float) / static_cast<std::size_t>(d))
            throw std::runtime_error("Invalid SAM2 tensor dimensions: " + t.name);
        count *= static_cast<std::size_t>(d);
    }
    if (t.data_type != core::TensorDataType::Float32 || t.data.size() != count * sizeof(float))
        throw std::runtime_error("SAM2 engines require FP32 external I/O: " + t.name);
    const auto* p = reinterpret_cast<const float*>(t.data.data());
    if (std::any_of(p, p + count, [](float x) { return !std::isfinite(x); }))
        throw std::runtime_error("SAM2 produced non-finite values: " + t.name);
    return p;
}
Tensors run(TrtSession& session, const Tensors& inputs) {
    for (const auto& name : session.input_names()) {
        const auto& t = inputs.at(name);
        session.set_shape(name, t.shape);
        session.set_input(name, t.data.data(), t.data.size());
    }
    session.infer();
    Tensors result;
    for (auto& output : session.outputs()) result.emplace(output.name, std::move(output));
    return result;
}

// Tensor-based torchvision Resize: antialiased bilinear on floating RGB,
// followed by ImageNet normalization. Do not round resized values to uint8.
std::vector<float> preprocess(const cv::Mat& image, int width, int height) {
    struct Filter { int first; std::vector<float> weights; };
    auto filters = [](int source, int dest) {
        std::vector<Filter> table;
        const float scale = float(source) / dest, support = std::max(1.F, scale);
        for (int i = 0; i < dest; ++i) {
            const float center = (i + .5F) * scale;
            const int first = std::max(0, int(std::ceil(center - support - .5F)));
            const int last = std::min(source - 1, int(std::floor(center + support - .5F)));
            Filter f{first, {}};
            float sum = 0;
            for (int j = first; j <= last; ++j) {
                float v = std::max(0.F, 1.F - std::abs((j + .5F - center) / support));
                f.weights.push_back(v); sum += v;
            }
            for (auto& v : f.weights) v /= sum;
            table.push_back(std::move(f));
        }
        return table;
    };
    const auto xs = filters(image.cols, width), ys = filters(image.rows, height);
    std::vector<float> horizontal(static_cast<std::size_t>(image.rows) * width * 3);
    for (int y = 0; y < image.rows; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        float value = 0;
        for (std::size_t k = 0; k < xs[x].weights.size(); ++k)
            value += xs[x].weights[k] * (image.at<cv::Vec3b>(y, xs[x].first + int(k))[2 - c] / 255.F);
        horizontal[(static_cast<std::size_t>(y) * width + x) * 3 + c] = value;
    }
    const std::size_t plane = static_cast<std::size_t>(width) * height;
    std::vector<float> data(plane * 3);
    constexpr float mean[] = {.485F, .456F, .406F}, stddev[] = {.229F, .224F, .225F};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        float value = 0;
        for (std::size_t k = 0; k < ys[y].weights.size(); ++k)
            value += ys[y].weights[k] * horizontal[(static_cast<std::size_t>(ys[y].first + int(k)) * width + x) * 3 + c];
        data[c * plane + static_cast<std::size_t>(y) * width + x] = (value - mean[c]) / stddev[c];
    }
    return data;
}
} // namespace

struct Segmentor::Impl {
    std::filesystem::path directory;
    Config config;
    int width{}, height{};
    Box box;
    bool has_box{};
    std::vector<Point> points;
    Tensors features;
    std::vector<float> previous_mask;
    std::unique_ptr<TrtSession> decoder;
    void require_image() const {
        if (features.empty()) throw std::runtime_error("Call SAM2 set_image before prompting");
    }
    void check_point(float x, float y) const {
        require_image();
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || y < 0 || x >= width || y >= height)
            throw std::invalid_argument("SAM2 point must be finite and inside the original image");
    }
};
Segmentor::Segmentor(const std::string& directory, Config config) : impl_(std::make_unique<Impl>()) {
    if (!std::isfinite(config.mask_threshold) || config.mask_threshold <= 0 || config.mask_threshold >= 1 ||
        config.max_points < 1 || config.max_points > 4096)
        throw std::invalid_argument("SAM2 mask threshold must be (0,1); max_points must be 1..4096");
    impl_->directory = std::filesystem::u8path(directory); impl_->config = config;
    std::ifstream manifest(impl_->directory / "bundle.json");
    if (!manifest || std::filesystem::file_size(impl_->directory / "bundle.json") > 1024 * 1024)
        throw std::invalid_argument("Missing or oversized SAM2 bundle.json; use build_sam2_engines.py");
    std::ostringstream contents; contents << manifest.rdbuf();
    const auto metadata = contents.str();
    std::smatch limit;
    if (!std::regex_search(metadata, std::regex(R"json("format"\s*:\s*"cvinfer-sam2.1-v1")json")) ||
        !std::regex_search(metadata, limit, std::regex(R"json("max_points"\s*:\s*([0-9]{1,4})\s*[,}])json")))
        throw std::invalid_argument("Invalid SAM2.1 bundle metadata");
    const int exported_limit = std::stoi(limit[1].str());
    if (exported_limit < 1 || exported_limit > 4096 || config.max_points > exported_limit)
        throw std::invalid_argument("SAM2 max_points exceeds the exported bundle limit");
    for (const auto* name : {"image_encoder.engine", "mask_decoder.engine"})
        if (!std::filesystem::is_regular_file(impl_->directory / name))
            throw std::invalid_argument(std::string("Missing SAM2 engine: ") + name);
}
Segmentor::~Segmentor() = default;
Segmentor::Segmentor(Segmentor&&) noexcept = default;
Segmentor& Segmentor::operator=(Segmentor&&) noexcept = default;
void Segmentor::set_image(const std::uint8_t* bgr, int w, int h, int channels) {
    if (!bgr || w <= 0 || h <= 0 || channels != 3 ||
        static_cast<std::size_t>(w) > std::numeric_limits<std::size_t>::max() / 3 / static_cast<std::size_t>(h))
        throw std::invalid_argument("SAM2 requires a contiguous nonempty 3-channel BGR image");
    impl_->features.clear(); impl_->decoder.reset(); clear_prompts();
    auto encoder = TrtSession::load((impl_->directory / "image_encoder.engine").u8string());
    if (encoder.tensor_shape("pixel_values") != std::vector<int64_t>{1, 3, 1024, 1024})
        throw std::runtime_error("SAM2.1 image encoder must accept 1x3x1024x1024");
    const cv::Mat image(h, w, CV_8UC3, const_cast<std::uint8_t*>(bgr));
    auto features = run(encoder, {{"pixel_values", tensor("pixel_values", {1, 3, 1024, 1024},
                            preprocess(image, 1024, 1024), core::TensorDataType::Float32)}});
    const std::map<std::string, std::vector<int64_t>> shapes{
        {"f0", {1, 32, 256, 256}}, {"f1", {1, 64, 128, 128}}, {"f2", {1, 256, 64, 64}}};
    for (const auto& entry : shapes) {
        const auto& t = features.at(entry.first);
        if (t.shape != entry.second) throw std::runtime_error("SAM2 encoder feature shape mismatch: " + entry.first);
        floats(t);
    }
    impl_->features = std::move(features); impl_->width = w; impl_->height = h;
}
void Segmentor::set_box(Box b) {
    impl_->require_image();
    if (!std::isfinite(b.x1) || !std::isfinite(b.y1) || !std::isfinite(b.x2) || !std::isfinite(b.y2) ||
        b.x1 < 0 || b.y1 < 0 || b.x2 > impl_->width || b.y2 > impl_->height || b.x1 >= b.x2 || b.y1 >= b.y2)
        throw std::invalid_argument("SAM2 box must be finite, ordered XYXY and inside the image");
    impl_->box = b; impl_->has_box = true; clear_points();
}
void Segmentor::clear_box() { impl_->has_box = false; impl_->previous_mask.clear(); }
void Segmentor::add_point(Point p) {
    impl_->check_point(p.x, p.y);
    if (impl_->points.size() >= static_cast<std::size_t>(impl_->config.max_points))
        throw std::invalid_argument("SAM2 max_points exceeded");
    impl_->points.push_back(p); // Preserve last logits for explicit iterative refinement.
}
void Segmentor::remove_point(std::size_t index) {
    if (index >= impl_->points.size()) throw std::invalid_argument("SAM2 point index out of range");
    impl_->points.erase(impl_->points.begin() + index); impl_->previous_mask.clear();
}
void Segmentor::clear_points() { impl_->points.clear(); impl_->previous_mask.clear(); }
void Segmentor::clear_prompts() { clear_box(); clear_points(); }
Mask Segmentor::predict(bool use_previous_mask) {
    auto& s = *impl_;
    s.require_image();
    if (!s.has_box && s.points.empty()) throw std::invalid_argument("SAM2 requires a box or at least one point");
    if (use_previous_mask && s.previous_mask.empty()) throw std::invalid_argument("SAM2 previous mask is unavailable");
    if (!s.decoder) s.decoder = std::make_unique<TrtSession>(TrtSession::load((s.directory / "mask_decoder.engine").u8string()));
    std::vector<float> coords;
    std::vector<std::int32_t> labels;
    auto add = [&](float x, float y, std::int32_t label) {
        coords.insert(coords.end(), {x * 1024 / s.width, y * 1024 / s.height}); labels.push_back(label);
    };
    if (s.has_box) { add(s.box.x1, s.box.y1, 2); add(s.box.x2, s.box.y2, 3); }
    for (const auto& p : s.points) add(p.x, p.y, p.positive ? 1 : 0);
    auto inputs = s.features;
    const auto n = static_cast<int64_t>(labels.size());
    inputs.emplace("point_coords", tensor("point_coords", {1, n, 2}, coords, core::TensorDataType::Float32));
    inputs.emplace("point_labels", tensor("point_labels", {1, n}, labels, core::TensorDataType::Int32));
    inputs.emplace("mask_input", tensor("mask_input", {1, 1, 256, 256},
        use_previous_mask ? s.previous_mask : std::vector<float>(256 * 256), core::TensorDataType::Float32));
    inputs.emplace("has_mask_input", tensor("has_mask_input", {1}, std::vector<float>{use_previous_mask ? 1.F : 0.F}, core::TensorDataType::Float32));
    const auto outputs = run(*s.decoder, inputs);
    const auto& masks = outputs.at("masks"); const auto& scores = outputs.at("iou_scores");
    if (masks.shape != std::vector<int64_t>{1, 3, 256, 256} || scores.shape != std::vector<int64_t>{1, 3})
        throw std::runtime_error("Invalid SAM2 decoder output shapes");
    const auto* quality = floats(scores); const auto* values = floats(masks);
    const auto best = std::max_element(quality, quality + 3) - quality;
    const auto* selected = values + best * 256 * 256;
    // Feedback uses the official predictor's clamped low-resolution logits.
    std::vector<float> previous(256 * 256);
    std::transform(selected, selected + previous.size(), previous.begin(), [](float v) { return std::clamp(v, -32.F, 32.F); });
    cv::Mat low(256, 256, CV_32F, const_cast<float*>(selected)), full;
    cv::resize(low, full, cv::Size(s.width, s.height), 0, 0, cv::INTER_LINEAR);
    Mask result;
    result.width = s.width; result.height = s.height; result.score = quality[best];
    result.pixels.resize(static_cast<std::size_t>(s.width) * s.height);
    const float threshold = std::log(s.config.mask_threshold / (1.F - s.config.mask_threshold));
    int x1 = s.width, y1 = s.height, x2 = 0, y2 = 0;
    for (int y = 0; y < s.height; ++y) for (int x = 0; x < s.width; ++x) {
        if (full.at<float>(y, x) > threshold) {
            result.pixels[static_cast<std::size_t>(y) * s.width + x] = 255;
            x1 = std::min(x1, x); y1 = std::min(y1, y); x2 = std::max(x2, x + 1); y2 = std::max(y2, y + 1);
        }
    }
    if (x2 > x1 && y2 > y1) result.box = {float(x1), float(y1), float(x2), float(y2)};
    s.previous_mask = std::move(previous);
    return result;
}
} // namespace cvinfer::sam2
