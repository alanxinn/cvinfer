#include "sam3/sam3.h"
#include "core/trt_core.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cvinfer::sam3 {
namespace {
using core::Tensor;
using core::TrtSession;
using Tensors = std::map<std::string, Tensor>;

// Separable antialiased bilinear resize, matching the processor's half-pixel
// convention. Downsampling uses a widened triangle rather than aliasing the
// image with an ordinary two-tap OpenCV resize.
std::vector<float> preprocess(const cv::Mat& image, int width, int height) {
    struct Filter { int first; std::vector<float> weights; };
    const auto filters = [](int source, int destination) {
        std::vector<Filter> table;
        const float scale = float(source) / destination, support = std::max(1.F, scale);
        for (int i = 0; i < destination; ++i) {
            const float center = (i + .5F) * scale;
            const int first = std::max(0, int(std::ceil(center - support - .5F)));
            const int last = std::min(source - 1, int(std::floor(center + support - .5F)));
            Filter filter{first, {}};
            float total = 0;
            for (int j = first; j <= last; ++j) {
                const float value = std::max(0.F, 1.F - std::abs((j + .5F - center) / support));
                filter.weights.push_back(value); total += value;
            }
            for (auto& value : filter.weights) value /= total;
            table.push_back(std::move(filter));
        }
        return table;
    };
    const auto xs = filters(image.cols, width), ys = filters(image.rows, height);
    std::vector<float> horizontal(static_cast<std::size_t>(image.rows) * width * 3);
    for (int y = 0; y < image.rows; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        float value = 0;
        for (std::size_t k = 0; k < xs[x].weights.size(); ++k)
            value += xs[x].weights[k] * image.at<cv::Vec3b>(y, xs[x].first + int(k))[2 - c];
        horizontal[(static_cast<std::size_t>(y) * width + x) * 3 + c] = value;
    }
    const auto plane = static_cast<std::size_t>(height) * width;
    std::vector<float> output(plane * 3);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        float value = 0;
        for (std::size_t k = 0; k < ys[y].weights.size(); ++k)
            value += ys[y].weights[k] * horizontal[(static_cast<std::size_t>(ys[y].first + int(k)) * width + x) * 3 + c];
        // The official image processor resizes uint8 images before rescaling.
        output[c * plane + static_cast<std::size_t>(y) * width + x] = std::round(value) / 127.5F - 1.F;
    }
    return output;
}

float sigmoid(float x) { return x >= 0 ? 1.F / (1.F + std::exp(-x)) : std::exp(x) / (1.F + std::exp(x)); }
const float* floats(const Tensor& tensor) {
    if (tensor.data_type != core::TensorDataType::Float32 || tensor.data.size() % sizeof(float))
        throw std::runtime_error("SAM3 bundle must use FP32 external I/O: " + tensor.name);
    return reinterpret_cast<const float*>(tensor.data.data());
}
template<class T> Tensor tensor(const std::string& name, std::vector<int64_t> shape,
                               const std::vector<T>& values, core::TensorDataType type) {
    Tensor result;
    result.name = name; result.shape = std::move(shape); result.data_type = type;
    result.data.resize(values.size() * sizeof(T));
    if (!values.empty()) std::memcpy(result.data.data(), values.data(), result.data.size());
    return result;
}
Tensors run(TrtSession& session, const Tensors& inputs) {
    for (const auto& name : session.input_names()) {
        const auto it = inputs.find(name);
        if (it == inputs.end()) throw std::runtime_error("SAM3 bundle input missing: " + name);
        session.set_shape(name, it->second.shape);
        session.set_input(name, it->second.data.data(), it->second.data.size());
    }
    session.infer();
    Tensors result;
    for (auto& output : session.outputs()) result.emplace(output.name, std::move(output));
    return result;
}
std::string unhex(const std::string& text) {
    if (text.size() % 2) throw std::runtime_error("Invalid SAM3 tokenizer hex string");
    std::string result;
    for (std::size_t i = 0; i < text.size(); i += 2) {
        std::size_t read = 0;
        const auto value = std::stoul(text.substr(i, 2), &read, 16);
        if (read != 2) throw std::runtime_error("Invalid SAM3 tokenizer hex string");
        result.push_back(static_cast<char>(value));
    }
    return result;
}

// CLIP byte BPE for ordinary English ASCII phrases. The exported vocabulary
// and merge ranks come from the exact official processor, never a substitute.
class Tokenizer {
    std::unordered_map<std::string, int64_t> vocabulary_;
    std::unordered_map<std::string, int> ranks_;
    int64_t bos_{}, eos_{}, pad_{};
    int64_t vocabulary_size_{};
public:
    explicit Tokenizer(const std::filesystem::path& directory) {
        std::ifstream vocab(directory / "vocab.tsv"), merges(directory / "merges.tsv");
        if (!vocab || !merges) throw std::runtime_error("SAM3 tokenizer files are missing; run export_sam3.py");
        if (!(vocab >> bos_ >> eos_ >> pad_)) throw std::runtime_error("Invalid SAM3 tokenizer header");
        if (std::min({bos_, eos_, pad_}) < 0) throw std::runtime_error("Invalid SAM3 special token ids");
        vocabulary_size_ = std::max({bos_, eos_, pad_}) + 1;
        int64_t id; std::string value;
        while (vocab >> id >> value) {
            if (id < 0 || id > 1000000) throw std::runtime_error("Invalid SAM3 vocabulary token id");
            vocabulary_.emplace(unhex(value), id);
            vocabulary_size_ = std::max(vocabulary_size_, id + 1);
        }
        int rank; std::string left, right;
        while (merges >> rank >> left >> right) ranks_.emplace(unhex(left) + '\0' + unhex(right), rank);
        if (vocabulary_.empty()) throw std::runtime_error("Empty SAM3 tokenizer vocabulary");
    }
    bool valid_id(int64_t id) const { return id >= 0 && id < vocabulary_size_; }
    std::pair<std::vector<int64_t>, std::vector<int64_t>> encode(std::string text, std::size_t length) const {
        if (length < 3 || text.empty()) throw std::invalid_argument("SAM3 requires a non-empty short text phrase");
        for (auto& c : text) {
            const auto byte = static_cast<unsigned char>(c);
            if (byte >= 127 || (byte < 32 && !std::isspace(byte)))
                throw std::invalid_argument("Native SAM3 text accepts English ASCII; use predict_tokens with the official tokenizer for Unicode");
            c = static_cast<char>(std::tolower(byte));
        }
        // CLIP basic_clean performs HTML unescaping twice.
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& item : std::vector<std::pair<std::string, std::string>>{
                     {"&amp;", "&"}, {"&quot;", "\""}, {"&#39;", "'"}, {"&lt;", "<"}, {"&gt;", ">"}}) {
                std::size_t at = 0;
                while ((at = text.find(item.first, at)) != std::string::npos) {
                    text.replace(at, item.first.size(), item.second); at += item.second.size();
                }
            }
        const std::regex pattern("'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\\sa-z0-9]+");
        std::vector<int64_t> ids{bos_};
        for (std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) {
            const auto word = it->str();
            std::vector<std::string> parts;
            for (const auto c : word) parts.emplace_back(1, c);
            parts.back() += "</w>";
            while (parts.size() > 1) {
                int best = std::numeric_limits<int>::max();
                std::size_t position = parts.size();
                for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
                    const auto rank = ranks_.find(parts[i] + '\0' + parts[i + 1]);
                    if (rank != ranks_.end() && rank->second < best) { best = rank->second; position = i; }
                }
                if (position == parts.size()) break;
                // Merge every occurrence of the best pair, as in CLIP BPE.
                const auto left = parts[position], right = parts[position + 1];
                for (std::size_t i = 0; i + 1 < parts.size();) {
                    if (parts[i] == left && parts[i + 1] == right) {
                        parts[i] += parts[i + 1]; parts.erase(parts.begin() + i + 1);
                    } else ++i;
                }
            }
            for (const auto& part : parts) {
                const auto found = vocabulary_.find(part);
                if (found == vocabulary_.end()) throw std::runtime_error("SAM3 tokenizer vocabulary mismatch");
                ids.push_back(found->second);
            }
        }
        if (ids.size() == 1) throw std::invalid_argument("SAM3 text phrase contains only whitespace");
        if (ids.size() + 1 > length) throw std::invalid_argument("SAM3 text is too long; use a shorter phrase");
        ids.push_back(eos_);
        std::vector<int64_t> mask(length, 0);
        std::fill_n(mask.begin(), ids.size(), 1);
        ids.resize(length, pad_);
        return {std::move(ids), std::move(mask)};
    }
};
} // namespace

struct Segmentor::Impl {
    std::filesystem::path directory;
    Config config;
    cv::Mat image;
    Box box;
    bool has_box{};
    std::vector<Point> points;
    Tensors detector_features, tracker_features;
    int tracker_width{}, tracker_height{};
    std::unique_ptr<Tokenizer> tokenizer;
    std::map<std::string, std::unique_ptr<TrtSession>> decoders;
    TrtSession& decoder(const std::string& name) {
        auto& session = decoders[name];
        if (!session) session = std::make_unique<TrtSession>(TrtSession::load((directory / (name + ".engine")).u8string()));
        return *session;
    }
    void require_image() const {
        if (image.empty()) throw std::runtime_error("Call SAM3 set_image before prompting");
    }
    Tensors encode_image(const char* name, int* input_width = nullptr, int* input_height = nullptr) {
        require_image();
        // Release decoder/text contexts before the large vision encoder is
        // loaded. Host feature caches survive mode switches and image clicks.
        decoders.clear();
        // Only one image encoder/context is resident at a time on an 8GB GPU.
        auto session = TrtSession::load((directory / (std::string(name) + ".engine")).u8string());
        const auto shape = session.tensor_shape("pixel_values");
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0)
            throw std::runtime_error("Invalid SAM3 vision engine input shape");
        const auto h = static_cast<int>(shape[2]), w = static_cast<int>(shape[3]);
        if (input_width) *input_width = w;
        if (input_height) *input_height = h;
        const auto data = preprocess(image, w, h);
        return run(session, {{"pixel_values", tensor("pixel_values", shape, data, core::TensorDataType::Float32)}});
    }
    Mask make_mask(const float* values, int h, int w, float score, Box bounds, bool probability) const {
        if (!std::isfinite(score)) throw std::runtime_error("SAM3 produced a non-finite mask score");
        cv::Mat low(h, w, CV_32F), full;
        for (int i = 0; i < h * w; ++i) {
            if (!std::isfinite(values[i])) throw std::runtime_error("SAM3 produced non-finite mask logits; verify engine precision");
            low.ptr<float>()[i] = probability ? sigmoid(values[i]) : values[i];
        }
        cv::resize(low, full, image.size(), 0, 0, cv::INTER_LINEAR);
        Mask result;
        result.box = bounds; result.score = score;
        result.width = image.cols; result.height = image.rows;
        result.pixels.resize(image.total());
        const float threshold = probability ? config.mask_threshold : std::log(config.mask_threshold / (1 - config.mask_threshold));
        for (std::size_t i = 0; i < result.pixels.size(); ++i)
            result.pixels[i] = full.ptr<float>()[i] > threshold ? 255 : 0;
        return result;
    }
};

Segmentor::Segmentor(const std::string& directory, Config config) : impl_(std::make_unique<Impl>()) {
    if (!(config.confidence_threshold >= 0 && config.confidence_threshold <= 1) ||
        !(config.mask_threshold > 0 && config.mask_threshold < 1) || config.max_points < 1)
        throw std::invalid_argument("Invalid SAM3 configuration");
    impl_->directory = std::filesystem::u8path(directory); impl_->config = config;
    for (const auto* name : {"detector_vision", "tracker_vision", "text_encoder", "text_decoder", "box_decoder", "interactive_decoder"})
        if (!std::filesystem::is_regular_file(impl_->directory / (std::string(name) + ".engine")))
            throw std::runtime_error(std::string("SAM3 engine is missing: ") + name);
}
Segmentor::~Segmentor() = default;
Segmentor::Segmentor(Segmentor&&) noexcept = default;
Segmentor& Segmentor::operator=(Segmentor&&) noexcept = default;
void Segmentor::set_image(const std::uint8_t* bgr, int width, int height, int channels) {
    if (!bgr || width <= 0 || height <= 0 || channels != 3) throw std::invalid_argument("SAM3 requires a contiguous 3-channel BGR image");
    impl_->image = cv::Mat(height, width, CV_8UC3, const_cast<std::uint8_t*>(bgr)).clone();
    impl_->detector_features.clear(); impl_->tracker_features.clear();
    impl_->decoders.clear();
    impl_->points.clear(); impl_->has_box = false;
}
void Segmentor::set_box(Box box) {
    impl_->require_image();
    for (const auto x : {box.x1, box.y1, box.x2, box.y2}) if (!std::isfinite(x)) throw std::invalid_argument("SAM3 box must be finite");
    if (box.x1 < 0 || box.y1 < 0 || box.x2 > impl_->image.cols || box.y2 > impl_->image.rows || box.x2 <= box.x1 || box.y2 <= box.y1)
        throw std::invalid_argument("SAM3 box is outside the image or empty");
    impl_->box = box; impl_->has_box = true; impl_->points.clear();
}
void Segmentor::add_point(Point point) {
    impl_->require_image();
    if (!impl_->has_box) throw std::runtime_error("Set a SAM3 box before adding foreground/background points");
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || point.x < 0 || point.y < 0 || point.x >= impl_->image.cols || point.y >= impl_->image.rows)
        throw std::invalid_argument("SAM3 point is outside the image");
    if (impl_->points.size() >= static_cast<std::size_t>(impl_->config.max_points)) throw std::invalid_argument("SAM3 maximum prompt point count exceeded");
    impl_->points.push_back(point);
}
void Segmentor::clear_points() { impl_->points.clear(); }
Mask Segmentor::predict() {
    impl_->require_image();
    if (!impl_->has_box) throw std::runtime_error("Set a SAM3 box before interactive inference");
    if (impl_->tracker_features.empty()) impl_->tracker_features = impl_->encode_image("tracker_vision", &impl_->tracker_width, &impl_->tracker_height);
    auto inputs = impl_->tracker_features;
    const float sx = float(impl_->tracker_width) / impl_->image.cols, sy = float(impl_->tracker_height) / impl_->image.rows;
    const auto b = impl_->box;
    inputs.emplace("input_boxes", tensor("input_boxes", {1, 1, 4}, std::vector<float>{b.x1 * sx, b.y1 * sy, b.x2 * sx, b.y2 * sy}, core::TensorDataType::Float32));
    if (!impl_->points.empty()) {
        std::vector<float> xy; std::vector<int64_t> labels;
        for (const auto p : impl_->points) { xy.push_back(p.x * sx); xy.push_back(p.y * sy); labels.push_back(p.positive ? 1 : 0); }
        const auto n = static_cast<int64_t>(labels.size());
        inputs.emplace("input_points", tensor("input_points", {1, 1, n, 2}, xy, core::TensorDataType::Float32));
        inputs.emplace("input_labels", tensor("input_labels", {1, 1, n}, labels, core::TensorDataType::Int64));
    }
    const auto outputs = run(impl_->decoder(impl_->points.empty() ? "box_decoder" : "interactive_decoder"), inputs);
    const auto& masks = outputs.at("masks"); const auto& scores = outputs.at("iou_scores");
    if (masks.shape.size() != 5 || masks.shape[0] != 1 || masks.shape[1] != 1 || masks.shape[2] <= 0)
        throw std::runtime_error("Invalid SAM3 tracker output shape");
    const auto count = masks.shape[2];
    if (scores.data.size() != static_cast<std::size_t>(count) * sizeof(float)) throw std::runtime_error("SAM3 tracker score count mismatch");
    const auto* quality = floats(scores);
    const auto best = std::max_element(quality, quality + count) - quality;
    const auto h = static_cast<int>(masks.shape[3]), w = static_cast<int>(masks.shape[4]);
    return impl_->make_mask(floats(masks) + best * h * w, h, w, quality[best], b, false);
}
std::vector<Mask> Segmentor::predict_text(const std::string& text) {
    impl_->require_image();
    if (!impl_->tokenizer) impl_->tokenizer = std::make_unique<Tokenizer>(impl_->directory);
    const auto shape = impl_->decoder("text_encoder").tensor_shape("input_ids");
    if (shape.size() != 2 || shape[0] != 1 || shape[1] < 3) throw std::runtime_error("Invalid SAM3 text encoder shape");
    const auto tokens = impl_->tokenizer->encode(text, static_cast<std::size_t>(shape[1]));
    return predict_tokens(tokens.first, tokens.second);
}
std::vector<Mask> Segmentor::predict_tokens(const std::vector<int64_t>& ids, const std::vector<int64_t>& mask) {
    impl_->require_image();
    if (!impl_->tokenizer) impl_->tokenizer = std::make_unique<Tokenizer>(impl_->directory);
    if (std::any_of(ids.begin(), ids.end(), [&](auto id) { return !impl_->tokenizer->valid_id(id); }))
        throw std::invalid_argument("SAM3 token id is outside the exported vocabulary");
    if (impl_->detector_features.empty()) impl_->detector_features = impl_->encode_image("detector_vision");
    auto& text_encoder = impl_->decoder("text_encoder");
    const auto shape = text_encoder.tensor_shape("input_ids");
    if (shape.size() != 2 || shape[0] != 1 || static_cast<int64_t>(ids.size()) != shape[1] || ids.size() != mask.size())
        throw std::invalid_argument("SAM3 token ids and attention mask must match the exported text engine length");
    if (std::any_of(ids.begin(), ids.end(), [](auto id) { return id < 0; }) ||
        std::any_of(mask.begin(), mask.end(), [](auto value) { return value != 0 && value != 1; }) ||
        std::none_of(mask.begin(), mask.end(), [](auto value) { return value == 1; }))
        throw std::invalid_argument("Invalid SAM3 tokens or attention mask");
    const auto text = run(text_encoder, {{"input_ids", tensor("input_ids", shape, ids, core::TensorDataType::Int64)},
                                         {"attention_mask", tensor("attention_mask", shape, mask, core::TensorDataType::Int64)}});
    auto inputs = impl_->detector_features;
    inputs.emplace("text_features", text.at("text_features"));
    inputs.emplace("attention_mask", tensor("attention_mask", shape, mask, core::TensorDataType::Int64));
    const auto output = run(impl_->decoder("text_decoder"), inputs);
    const auto& masks = output.at("masks"); const auto& boxes = output.at("boxes"); const auto& scores = output.at("scores");
    if (masks.shape.size() != 4 || masks.shape[0] != 1 || boxes.shape.size() != 3 || boxes.shape[0] != 1 || boxes.shape[2] != 4 || boxes.shape[1] != masks.shape[1])
        throw std::runtime_error("Invalid SAM3 concept decoder output shape");
    const auto count = masks.shape[1];
    if (scores.data.size() != static_cast<std::size_t>(count) * sizeof(float)) throw std::runtime_error("SAM3 concept score count mismatch");
    const auto* score = floats(scores); const auto* bounds = floats(boxes); const auto* values = floats(masks);
    const auto h = static_cast<int>(masks.shape[2]), w = static_cast<int>(masks.shape[3]);
    std::vector<Mask> result;
    for (int64_t i = 0; i < count; ++i) {
        if (!std::isfinite(score[i])) throw std::runtime_error("SAM3 produced non-finite concept scores; verify engine precision");
        if (score[i] <= impl_->config.confidence_threshold) continue;
        for (int c = 0; c < 4; ++c) if (!std::isfinite(bounds[i * 4 + c])) throw std::runtime_error("SAM3 produced non-finite boxes");
        Box box{std::clamp(bounds[i * 4], 0.F, 1.F) * impl_->image.cols, std::clamp(bounds[i * 4 + 1], 0.F, 1.F) * impl_->image.rows,
                std::clamp(bounds[i * 4 + 2], 0.F, 1.F) * impl_->image.cols, std::clamp(bounds[i * 4 + 3], 0.F, 1.F) * impl_->image.rows};
        result.push_back(impl_->make_mask(values + i * h * w, h, w, score[i], box, true));
    }
    return result;
}
} // namespace cvinfer::sam3
