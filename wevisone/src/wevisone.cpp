#include "wevisone/wevisone.h"
#include "tokenizer.h"
#include "core/trt_core.h"
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace cvinfer::wevisone {
namespace {
using core::Tensor;
using core::TrtSession;
using Json = nlohmann::json;
const char* names[] = {"detection", "referring", "point", "count_box", "count_point", "ocr",
                       "ocr_grounding", "segmentation", "matting", "visual_prompt", "freeform"};
void require(bool value, const char* text) { if (!value) throw std::invalid_argument(text); }
void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("WeVisOne CUDA: ") + cudaGetErrorString(error));
}
std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string join(const std::vector<std::string>& values, const char* sep) {
    std::string result;
    for (const auto& item : values) {
        const auto value = trim(item);
        if (value.empty()) continue;
        if (!result.empty()) result += sep;
        result += value;
    }
    return result;
}
void validate_box(const Box& b, int w, int h) {
    require(std::isfinite(b.x1) && std::isfinite(b.y1) && std::isfinite(b.x2) && std::isfinite(b.y2)
            && b.x1 >= 0 && b.y1 >= 0 && b.x2 <= w && b.y2 <= h && b.x2 > b.x1 && b.y2 > b.y1,
            "WeVisOne boxes must be finite original-image XYXY rectangles inside the image");
}
std::string box_tokens(const Box& b, int w, int h) {
    std::ostringstream s;
    for (const float x : {b.x1 / w, b.y1 / h, b.x2 / w, b.y2 / h})
        s << '<' << std::clamp(static_cast<int>(x * 1000), 0, 1000) << '>';
    return s.str();
}
std::string prompt_for(const Request& r, int width, int height) {
    if (!r.prompt.empty()) return r.prompt;
    const auto text = join(r.query, " ");
    std::vector<std::string> labels;
    for (const auto& value : r.query) {
        std::istringstream lines(value); std::string line;
        while (std::getline(lines, line)) if (!trim(line).empty()) labels.push_back(trim(line));
    }
    auto phrase = text;
    if (!phrase.empty() && phrase.back() != '.' && phrase.back() != '?' && phrase.back() != '!') phrase += '.';
    switch (r.task) {
    case Task::Detection: require(!text.empty(), "Detection query is required");
        return "Locate all the instances that match the following description: " + join(labels, ". ") + ".";
    case Task::Referring: require(!text.empty(), "Referring query is required");
        return "Locate the instances that match the following description: " + phrase;
    case Task::Point: require(!text.empty(), "Point query is required"); return "point to: " + join(labels, ". ");
    case Task::CountBox: return "How many " + (text.empty() ? "objects" : text) + " in the image? Please count them with boxes.";
    case Task::CountPoint: return "How many " + (text.empty() ? "objects" : text) + " in the image?";
    case Task::Ocr: return "detect all the word in the box format";
    case Task::OcrGrounding: require(!text.empty(), "OCR grounding query is required"); return "Locate the following text in the image: " + text;
    case Task::Segmentation: case Task::Matting:
        require(!text.empty(), "Segmentation/matting query is required");
        return "Locate the instances that match the following description and provide the alpha mattes: " + phrase;
    case Task::VisualPrompt: {
        require(!r.reference_boxes.empty(), "VisualPrompt requires at least one reference box");
        std::string boxes;
        for (const auto& b : r.reference_boxes) {
            validate_box(b, width, height);
            if (!boxes.empty()) boxes += ',';
            boxes += box_tokens(b, width, height);
        }
        return "Locate all similar objects in the image based on the reference boxes: <ref>object 1</ref><box>" + boxes + "</box>";
    }
    case Task::Freeform: require(!text.empty(), "Freeform query or prompt is required"); return text;
    }
    throw std::invalid_argument("Unknown WeVisOne task");
}

// Pillow-compatible antialiased bicubic coefficients (a=-0.5), resizing uint8
// before normalization. Qwen and the pixel decoder have independent canvases.
cv::Mat resize_rgb(const cv::Mat& bgr, int width, int height) {
    struct Filter { int start; std::vector<double> weights; };
    const auto kernel = [](double x) {
        x = std::abs(x);
        return x < 1 ? ((1.5 * x - 2.5) * x) * x + 1 :
               x < 2 ? ((-.5 * x + 2.5) * x - 4) * x + 2 : 0.;
    };
    const auto filters = [&](int source, int destination) {
        std::vector<Filter> table;
        const double scale = double(source) / destination, factor = std::max(1., scale);
        for (int i = 0; i < destination; ++i) {
            const double center = (i + .5) * scale;
            const int first = std::max(0, int(center - 2 * factor + .5));
            const int last = std::min(source, int(center + 2 * factor + .5));
            Filter f{first, {}}; double total = 0;
            for (int j = first; j < last; ++j) { const double v = kernel((j + .5 - center) / factor); f.weights.push_back(v); total += v; }
            for (auto& v : f.weights) v /= total;
            table.push_back(std::move(f));
        }
        return table;
    };
    const auto xs = filters(bgr.cols, width), ys = filters(bgr.rows, height);
    cv::Mat horizontal(bgr.rows, width, CV_8UC3), output(height, width, CV_8UC3);
    for (int y = 0; y < bgr.rows; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        double value = 0;
        for (size_t k = 0; k < xs[x].weights.size(); ++k) value += xs[x].weights[k] * bgr.at<cv::Vec3b>(y, xs[x].start + int(k))[2 - c];
        horizontal.at<cv::Vec3b>(y, x)[c] = static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
        double value = 0;
        for (size_t k = 0; k < ys[y].weights.size(); ++k) value += ys[y].weights[k] * horizontal.at<cv::Vec3b>(ys[y].start + int(k), x)[c];
        output.at<cv::Vec3b>(y, x)[c] = static_cast<uint8_t>(std::clamp(std::lround(value), 0L, 255L));
    }
    return output;
}
template<class T> void input(TrtSession& session, const char* name, std::vector<int64_t> shape, const std::vector<T>& values) {
    session.set_shape(name, shape); session.set_input(name, values.data(), values.size() * sizeof(T));
}
std::vector<float> floats(const Tensor& t) {
    if (t.data_type != core::TensorDataType::Float32 || t.data.size() % sizeof(float))
        throw std::runtime_error("WeVisOne requires FP32 external features: " + t.name);
    std::vector<float> result(t.data.size() / sizeof(float));
    std::memcpy(result.data(), t.data.data(), t.data.size()); return result;
}
size_t volume(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (const auto d : shape) {
        if (d <= 0 || uint64_t(d) > std::numeric_limits<size_t>::max() / n) throw std::runtime_error("Invalid WeVisOne tensor dimensions");
        n *= static_cast<size_t>(d);
    }
    return n;
}
std::vector<float> device_floats(TrtSession& session, const char* name) {
    if (session.device_output_type(name) != core::TensorDataType::Float32) throw std::runtime_error("WeVisOne graph output must be FP32");
    std::vector<float> result(volume(session.device_output_shape(name)));
    cuda_check(cudaMemcpy(result.data(), session.device_output(name), result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}
void validate_options(const Options& o) {
    require(o.max_new_tokens > 0 && o.top_k >= 0 && std::isfinite(o.temperature) && o.temperature >= 0
            && std::isfinite(o.top_p) && o.top_p > 0 && o.top_p <= 1
            && std::isfinite(o.repetition_penalty) && o.repetition_penalty > 0
            && std::isfinite(o.mask_threshold) && o.mask_threshold >= 0 && o.mask_threshold <= 1,
            "Invalid WeVisOne generation or mask options");
}
int64_t sample(std::vector<float> logits, const std::vector<int64_t>& seen, const Options& o, std::mt19937_64& random) {
    for (const auto id : seen) if (id >= 0 && size_t(id) < logits.size())
        logits[id] = logits[id] < 0 ? logits[id] * o.repetition_penalty : logits[id] / o.repetition_penalty;
    if (logits.empty() || std::any_of(logits.begin(), logits.end(), [](float v) { return !std::isfinite(v); }))
        throw std::runtime_error("WeVisOne produced invalid logits");
    const auto range = std::minmax_element(logits.begin(), logits.end());
    if (*range.first == *range.second)
        throw std::runtime_error("WeVisOne produced uniform logits; check the engine conversion precision (FP32 normalization must be preserved)");
    if (o.temperature == 0) return std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::vector<size_t> indices(logits.size()); std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    if (o.top_k > 0 && size_t(o.top_k) < indices.size()) indices.resize(o.top_k);
    std::vector<double> weights; double sum = 0;
    for (const auto i : indices) { const double weight = std::exp((double(logits[i]) - logits[indices[0]]) / o.temperature); weights.push_back(weight); sum += weight; }
    double cumulative = 0; size_t keep = 0;
    do { cumulative += weights[keep++] / sum; } while (keep < weights.size() && cumulative < o.top_p);
    weights.resize(keep);
    return static_cast<int64_t>(indices[std::discrete_distribution<size_t>(weights.begin(), weights.end())(random)]);
}
struct PixelTarget { Box box; size_t annotation{}, obj{}; };
std::vector<PixelTarget> parse_response(Result& result, Task task, int w, int h) {
    std::vector<PixelTarget> targets;
    const std::regex refs(R"(<ref>((?:(?!</?ref>)[\s\S])*)</ref>\s*<(box|point|polygon)>([\s\S]*?)(?:</(?:box|point|polygon)>|(?=<ref>)|$))");
    const std::regex coords(R"(<(\d{1,4})>)");
    const std::regex absent(R"(^\s*None\s*$)", std::regex::icase);
    const std::regex absent_ref(R"(<ref>((?:(?!</?ref>)[\s\S])*)</ref>\s*(?:<(?:box|point|polygon)>\s*)?None(?:\s*</(?:box|point|polygon)>|\s*(?=<ref>)|\s*$))", std::regex::icase);
    for (std::sregex_iterator it(result.response.begin(), result.response.end(), absent_ref), end; it != end; ++it)
        result.absent_labels.push_back(trim((*it)[1].str()));
    for (std::sregex_iterator it(result.response.begin(), result.response.end(), refs), end; it != end; ++it) {
        const auto label = trim((*it)[1].str()), type = (*it)[2].str();
        auto body = (*it)[3].str();
        if (std::regex_match(body, absent)) continue;
        size_t obj_ordinal = 0, search = 0;
        const auto body_start = static_cast<size_t>((*it).position(3));
        while ((search = result.response.find("<OBJ>", search)) != std::string::npos && search < body_start) {
            ++obj_ordinal; search += 5;
        }
        // Official pixel parsing also accepts adjacent box+OBJ groups without
        // commas. Only normalize the local parser input, retaining raw response.
        for (size_t at = 0; (at = body.find("<OBJ>", at)) != std::string::npos;) {
            at += 5; body.insert(at, ","); ++at;
        }
        std::istringstream segments(body); std::string segment;
        while (std::getline(segments, segment, ',')) {
            std::vector<int> values;
            for (std::sregex_iterator c(segment.begin(), segment.end(), coords); c != end; ++c) values.push_back(std::stoi((*c)[1].str()));
            const bool obj = segment.find("<OBJ>") != std::string::npos;
            const auto ordinal = obj_ordinal; if (obj) ++obj_ordinal;
            const bool point = type == "point";
            if (values.size() != (point ? 2U : 4U) || std::any_of(values.begin(), values.end(), [](int v) { return v > 1000; })) continue;
            Annotation a; a.label = label; a.is_point = point;
            if (point) { a.x = values[0] * (w / 1000.F); a.y = values[1] * (h / 1000.F); }
            else {
                a.box = {values[0] * (w / 1000.F), values[1] * (h / 1000.F), values[2] * (w / 1000.F), values[3] * (h / 1000.F)};
                if (a.box.x2 <= a.box.x1 || a.box.y2 <= a.box.y1) continue;
                if (obj) targets.push_back({a.box, result.annotations.size(), ordinal});
            }
            result.annotations.push_back(std::move(a));
        }
    }
    result.rejected = std::regex_match(result.response, absent) ||
                      (result.annotations.empty() && !result.absent_labels.empty());
    if ((task == Task::CountBox || task == Task::CountPoint) && !result.truncated
        && (!result.annotations.empty() || result.rejected)) {
        const auto count = std::count_if(result.annotations.begin(), result.annotations.end(),
            [&](const Annotation& a) { return a.is_point == (task == Task::CountPoint); });
        if (static_cast<size_t>(count) == result.annotations.size()) result.count = static_cast<int>(count);
    }
    return targets;
}
}
Task parse_task(const std::string& name) {
    for (int i = 0; i < 11; ++i) if (name == names[i]) return static_cast<Task>(i);
    throw std::invalid_argument("Unknown WeVisOne task: " + name);
}
const char* task_name(Task task) {
    const auto i = static_cast<int>(task); require(i >= 0 && i < 11, "Unknown WeVisOne task"); return names[i];
}

struct Model::Impl {
    Json bundle;
    Tokenizer tokenizer;
    TrtSession vision, language;
    std::unique_ptr<TrtSession> pixel;
    std::filesystem::path directory;
    cv::Mat image;
    std::map<std::string, Tensor> vision_features;
    std::vector<float> cnn_image;
    int width{}, height{}, patch{}, temporal{}, merge{}, hidden{}, layers{}, kv_heads{}, head_dim{}, max_context{}, max_prefill{}, max_objects{}, cnn_size{};
    int64_t image_id{}, obj_id{}, eos_id{}, pad_id{};
    void* cache{};
    size_t cache_capacity{}, cache_element_bytes{};
    int64_t cache_length{1}, next_position{};
    bool vision_ready{false};
    static Json read_bundle(const std::filesystem::path& p) {
        std::ifstream f(p / "bundle.json"); if (!f) throw std::runtime_error("Missing WeVisOne bundle.json");
        auto b = Json::parse(f);
        if (b.at("format") != "cvinfer-wevisone-v1" || b.value("status", "") != "ENGINE_BUILT_REAL_VALIDATION_REQUIRED")
            throw std::runtime_error("Unsupported/incomplete WeVisOne engine bundle; run build_wevisone_engines.py");
        return b;
    }
    std::filesystem::path asset(const std::string& name) const {
        const auto relative = std::filesystem::u8path(name);
        if (relative.empty() || relative.is_absolute()) throw std::runtime_error("Invalid WeVisOne bundle asset");
        for (const auto& part : relative) if (part == "..") throw std::runtime_error("WeVisOne asset escapes bundle directory");
        return directory / relative;
    }
    explicit Impl(const std::string& path)
        : bundle(read_bundle(std::filesystem::u8path(path))),
          tokenizer(std::filesystem::u8path(path) / "tokenizer_native.json"), directory(std::filesystem::u8path(path)) {
        const auto& v = bundle.at("vision"); const auto& l = bundle.at("language");
        width = v.at("width"); height = v.at("height"); patch = v.at("patch_size");
        temporal = v.at("temporal_patch_size"); merge = v.at("spatial_merge_size");
        hidden = l.at("hidden_size"); layers = l.at("layers"); kv_heads = l.at("kv_heads"); head_dim = l.at("head_dim");
        max_context = l.at("max_context"); max_prefill = l.at("max_prefill");
        max_objects = bundle.at("pixel").at("max_objects"); cnn_size = bundle.at("pixel").at("image_size");
        require(width > 0 && height > 0 && width <= 4096 && height <= 4096 && patch > 0 && patch <= 128 && merge > 0 && merge <= 16
            && width % (patch * merge) == 0 && height % (patch * merge) == 0 && temporal > 0 && temporal <= 4
            && hidden > 0 && hidden <= 16384 && layers > 0 && layers <= 128 && kv_heads > 0 && kv_heads <= 128
            && head_dim > 0 && head_dim <= 512 && max_context > 0 && max_context <= 131072
            && max_prefill > 0 && max_prefill < max_context && max_objects > 0 && max_objects <= 1024
            && cnn_size > 0 && cnn_size <= 4096, "Invalid WeVisOne bundle dimensions");
        image_id = tokenizer.token_id("<|image_pad|>"); obj_id = tokenizer.token_id("<OBJ>");
        eos_id = tokenizer.token_id("<|im_end|>"); pad_id = tokenizer.token_id("<|endoftext|>");
        if (image_id != l.at("image_token_id").get<int64_t>() || obj_id != l.at("obj_token_id").get<int64_t>())
            throw std::runtime_error("WeVisOne tokenizer/model token IDs disagree");
        cache_element_bytes = l.at("cache_dtype") == "fp16" ? 2 : l.at("cache_dtype") == "fp32" ? 4 : 0;
        require(cache_element_bytes != 0, "Unsupported WeVisOne KV cache dtype");
        vision = TrtSession::load(asset(bundle.at("graphs").at("vision").at("engine")).u8string());
        language = TrtSession::load(asset(bundle.at("graphs").at("language").at("engine")).u8string());
        language.set_output_host_copy(false);
        cache_capacity = volume({layers, 2, kv_heads, int64_t(max_context) + 1, head_dim}) * cache_element_bytes;
        cuda_check(cudaMalloc(&cache, cache_capacity));
    }
    ~Impl() { if (cache) cudaFree(cache); }
    size_t cache_bytes(int64_t length) const { return volume({layers, 2, kv_heads, length, head_dim}) * cache_element_bytes; }
    void encode_image() {
        if (vision_ready) return;
        const auto resized = resize_rgb(image, width, height);
        const int gh = height / patch, gw = width / patch;
        const auto means = bundle.at("vision").at("mean").get<std::vector<float>>();
        const auto stds = bundle.at("vision").at("std").get<std::vector<float>>();
        if (means.size() != 3 || stds.size() != 3 || std::any_of(stds.begin(), stds.end(), [](float x) { return !std::isfinite(x) || x <= 0; }))
            throw std::runtime_error("Invalid WeVisOne image normalization");
        std::vector<float> values; values.reserve(size_t(gh) * gw * 3 * temporal * patch * patch);
        for (int by = 0; by < gh / merge; ++by) for (int bx = 0; bx < gw / merge; ++bx)
            for (int iy = 0; iy < merge; ++iy) for (int ix = 0; ix < merge; ++ix)
                for (int c = 0; c < 3; ++c) for (int t = 0; t < temporal; ++t)
                    for (int y = 0; y < patch; ++y) for (int x = 0; x < patch; ++x) {
                        const auto pixel = resized.at<cv::Vec3b>((by * merge + iy) * patch + y, (bx * merge + ix) * patch + x)[c];
                        values.push_back((pixel / 255.F - means[c]) / stds[c]);
                    }
        input(vision, "pixel_values", {int64_t(gh) * gw, 3LL * temporal * patch * patch}, values);
        vision.infer(); vision_features.clear();
        for (auto& t : vision.outputs()) vision_features.emplace(t.name, std::move(t));
        const auto rgb = resize_rgb(image, cnn_size, cnn_size);
        const size_t plane = size_t(cnn_size) * cnn_size; cnn_image.resize(plane * 3);
        for (int y = 0; y < cnn_size; ++y) for (int x = 0; x < cnn_size; ++x) for (int c = 0; c < 3; ++c)
            cnn_image[c * plane + size_t(y) * cnn_size + x] = rgb.at<cv::Vec3b>(y, x)[c] / 255.F;
        vision_ready = true;
    }
    std::pair<std::vector<float>, std::vector<float>> forward(const std::vector<int64_t>& ids, bool prefill) {
        const int64_t n = static_cast<int64_t>(ids.size());
        require(n > 0 && n <= max_prefill && cache_length + n <= int64_t(max_context) + 1, "WeVisOne context/profile capacity exceeded");
        std::vector<int64_t> positions(size_t(3 * n));
        std::vector<float> image_embeds(size_t(n) * hidden, 0), deep(size_t(3 * n) * hidden, 0), mask(size_t(n), 0);
        int64_t position = next_position;
        const int gh = height / patch / merge, gw = width / patch / merge;
        size_t image_index = 0;
        std::vector<float> merged;
        std::vector<std::vector<float>> deep_features;
        if (prefill) {
            merged = floats(vision_features.at("image_features"));
            if (merged.size() != size_t(gh) * gw * hidden) throw std::runtime_error("WeVisOne merged vision feature shape mismatch");
            for (int i = 0; i < 3; ++i) {
                deep_features.push_back(floats(vision_features.at("deepstack_" + std::to_string(i))));
                if (deep_features.back().size() != merged.size()) throw std::runtime_error("WeVisOne DeepStack shape mismatch");
            }
        }
        int64_t image_start = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (prefill && ids[i] == image_id) {
                if (image_start < 0) image_start = position;
                if (image_index >= size_t(gh) * gw) throw std::runtime_error("Too many WeVisOne image tokens");
                positions[i] = image_start;
                positions[n + i] = image_start + static_cast<int64_t>(image_index / gw);
                positions[2 * n + i] = image_start + static_cast<int64_t>(image_index % gw);
                mask[i] = 1;
                std::copy_n(merged.data() + image_index * hidden, hidden, image_embeds.data() + i * hidden);
                for (int d = 0; d < 3; ++d)
                    std::copy_n(deep_features[d].data() + image_index * hidden, hidden, deep.data() + (d * n + i) * hidden);
                ++image_index;
                if (image_index == size_t(gh) * gw) position = image_start + std::max(gh, gw);
            } else { positions[i] = positions[n + i] = positions[2 * n + i] = position++; }
        }
        if (prefill && image_index != size_t(gh) * gw) throw std::runtime_error("WeVisOne image placeholder count mismatch");
        input(language, "input_ids", {1, n}, ids);
        input(language, "position_ids", {3, 1, n}, positions);
        input(language, "image_features", {1, n, hidden}, image_embeds);
        input(language, "deepstack", {3, 1, n, hidden}, deep);
        input(language, "visual_mask", {1, n, 1}, mask);
        language.set_shape("past", {layers, 2, kv_heads, cache_length, head_dim});
        language.set_input_device("past", cache, cache_bytes(cache_length));
        language.infer();
        const auto expected_type = cache_element_bytes == 2 ? core::TensorDataType::Float16 : core::TensorDataType::Float32;
        if (language.device_output_type("present") != expected_type ||
            language.device_output_shape("present") != std::vector<int64_t>{layers, 2, kv_heads, cache_length + n, head_dim})
            throw std::runtime_error("WeVisOne KV cache output contract mismatch");
        cache_length += n; next_position = position;
        // Retain only GPU cache data. Never round-trip the KV cache through CPU RAM.
        cuda_check(cudaMemcpy(cache, language.device_output("present"), cache_bytes(cache_length), cudaMemcpyDeviceToDevice));
        auto logits = device_floats(language, "logits"), last_hidden = device_floats(language, "last_hidden");
        if (last_hidden.size() != size_t(hidden) || logits.size() != bundle.at("language").at("vocab_size").get<size_t>())
            throw std::runtime_error("WeVisOne language output shape mismatch");
        if (prefill && std::getenv("CVINFER_WEVISONE_TRACE")) {
            const auto lr = std::minmax_element(logits.begin(), logits.end());
            const auto hr = std::minmax_element(last_hidden.begin(), last_hidden.end());
            const auto vr = std::minmax_element(merged.begin(), merged.end());
            std::cerr << "WeVisOne trace: vision=[" << *vr.first << ',' << *vr.second << "] logits=["
                      << *lr.first << ',' << *lr.second << "] argmax=" << (lr.second - logits.begin())
                      << " hidden=[" << *hr.first << ',' << *hr.second << "]\n";
        }
        return {std::move(logits), std::move(last_hidden)};
    }
    std::vector<int64_t> chat(const std::string& prompt) const {
        auto prefix = tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>");
        prefix.insert(prefix.end(), size_t(height / patch / merge) * (width / patch / merge), image_id);
        auto suffix = tokenizer.encode("<|vision_end|>" + prompt + "<|im_end|>\n<|im_start|>assistant\n");
        prefix.insert(prefix.end(), suffix.begin(), suffix.end()); return prefix;
    }
    void decode_pixels(Result& result, const std::vector<PixelTarget>& targets,
                       const std::vector<std::vector<float>>& states, const Options& options, bool binary) {
        if (targets.empty() || !options.decode_pixels) return;
        require(targets.size() <= size_t(max_objects), "WeVisOne object count exceeds pixel decoder profile; export with larger --max-objects");
        if (!pixel) pixel = std::make_unique<TrtSession>(TrtSession::load(asset(bundle.at("graphs").at("pixel").at("engine")).u8string()));
        const int64_t n = static_cast<int64_t>(targets.size());
        std::vector<float> boxes, hidden_states;
        for (const auto& t : targets) {
            if (t.obj >= states.size() || states[t.obj].size() != size_t(hidden)) throw std::runtime_error("WeVisOne OBJ state and box alignment mismatch");
            hidden_states.insert(hidden_states.end(), states[t.obj].begin(), states[t.obj].end());
            boxes.insert(boxes.end(), {t.box.x1 / image.cols, t.box.y1 / image.rows, t.box.x2 / image.cols, t.box.y2 / image.rows});
        }
        input(*pixel, "image", {1, 3, cnn_size, cnn_size}, cnn_image);
        input(*pixel, "t_seg", {n, hidden}, hidden_states);
        input(*pixel, "boxes", {n, 4}, boxes);
        for (int i = 0; i < 4; ++i) {
            const auto name = "visual_" + std::to_string(i); const auto& t = vision_features.at(name);
            input(*pixel, name.c_str(), t.shape, floats(t));
        }
        pixel->infer(); const auto& t = pixel->get_output_ref("mask_logits");
        if (t.shape != std::vector<int64_t>{n, cnn_size, cnn_size}) throw std::runtime_error("WeVisOne pixel output shape mismatch");
        const auto logits = floats(t); const size_t plane = size_t(cnn_size) * cnn_size;
        for (size_t i = 0; i < targets.size(); ++i) {
            cv::Mat probabilities(cnn_size, cnn_size, CV_32F);
            for (size_t p = 0; p < plane; ++p) {
                const auto v = logits[i * plane + p];
                if (!std::isfinite(v)) throw std::runtime_error("Nonfinite WeVisOne pixel logits");
                probabilities.ptr<float>()[p] = v >= 0 ? 1.F / (1.F + std::exp(-v)) : std::exp(v) / (1.F + std::exp(v));
            }
            cv::Mat resized; cv::resize(probabilities, resized, image.size(), 0, 0, cv::INTER_LINEAR);
            Mask m; m.box = targets[i].box; m.width = image.cols; m.height = image.rows;
            m.alpha.assign(resized.ptr<float>(), resized.ptr<float>() + size_t(m.width) * m.height);
            if (binary) { m.binary.reserve(m.alpha.size()); for (auto a : m.alpha) m.binary.push_back(a >= options.mask_threshold ? 255 : 0); }
            result.annotations[targets[i].annotation].mask_index = static_cast<int>(result.masks.size());
            result.masks.push_back(std::move(m));
        }
    }
    Result predict(const Request& request, const std::vector<Box>* forced_boxes = nullptr) {
        require(!image.empty(), "Call SetImage before WeVisOne Predict"); validate_options(request.options);
        const auto started = std::chrono::steady_clock::now();
        Result result; result.prompt = prompt_for(request, image.cols, image.rows);
        encode_image(); auto ids = chat(result.prompt);
        require(ids.size() <= size_t(max_prefill), "WeVisOne prompt exceeds --max-prefill profile");
        cache_length = 1; next_position = 0; cuda_check(cudaMemset(cache, 0, cache_bytes(1)));
        auto output = forward(ids, true);
        std::vector<int64_t> emitted, forced;
        std::vector<std::vector<float>> states;
        if (forced_boxes) {
            std::string response = "<ref>object</ref><box>";
            for (size_t i = 0; i < forced_boxes->size(); ++i) { if (i) response += ','; response += box_tokens((*forced_boxes)[i], image.cols, image.rows) + "<OBJ>"; }
            response += "</box>"; forced = tokenizer.encode(response);
            require(ids.size() + forced.size() <= size_t(max_context), "WeVisOne box response exceeds context capacity");
        } else require(ids.size() + size_t(request.options.max_new_tokens) <= size_t(max_context), "WeVisOne max_new_tokens exceeds remaining context capacity");
        std::mt19937_64 random(request.options.seed);
        const size_t limit = forced_boxes ? forced.size() : size_t(request.options.max_new_tokens);
        bool stopped = false;
        for (size_t i = 0; i < limit; ++i) {
            std::vector<int64_t> seen = ids; seen.insert(seen.end(), emitted.begin(), emitted.end());
            std::sort(seen.begin(), seen.end()); seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
            const auto id = forced_boxes ? forced[i] : sample(std::move(output.first), seen, request.options, random);
            if (!forced_boxes && (id == eos_id || id == pad_id)) { stopped = true; break; }
            emitted.push_back(id);
            // Process OBJ itself to capture its hidden state (not the preceding token).
            output = forward({id}, false);
            if (id == obj_id) states.push_back(std::move(output.second));
        }
        result.truncated = !forced_boxes && !stopped;
        result.response = trim(tokenizer.decode(emitted));
        auto targets = parse_response(result, request.task, image.cols, image.rows);
        if (forced_boxes) {
            if (states.size() != forced_boxes->size()) throw std::runtime_error("WeVisOne forced OBJ count mismatch");
            // Box overrides remain valid even if a very small box rounds to a
            // zero-width coordinate bin in the teacher-forced token transcript.
            targets.clear(); result.annotations.clear();
            for (size_t i = 0; i < forced_boxes->size(); ++i) {
                Annotation a; a.label = "object"; a.box = (*forced_boxes)[i];
                result.annotations.push_back(std::move(a)); targets.push_back({(*forced_boxes)[i], i, i});
            }
        }
        if (!forced_boxes && !result.truncated && request.options.decode_pixels && !targets.empty()) {
            // Match the official decode_masks teacher-forcing pass. Retokenizing
            // the decoded response can differ from the tokens chosen by generate,
            // so generation-time OBJ states alone are not an exact substitute.
            const auto teacher = tokenizer.encode(result.response);
            require(ids.size() + teacher.size() <= size_t(max_context), "WeVisOne teacher-forced response exceeds context capacity");
            cache_length = 1; next_position = 0; cuda_check(cudaMemset(cache, 0, cache_bytes(1)));
            forward(ids, true); states.clear();
            size_t begin = 0;
            for (size_t i = 0; i < teacher.size(); ++i) {
                if (teacher[i] == obj_id || i + 1 - begin == size_t(max_prefill) || i + 1 == teacher.size()) {
                    std::vector<int64_t> chunk(teacher.begin() + begin, teacher.begin() + i + 1);
                    auto features = forward(chunk, false);
                    if (teacher[i] == obj_id) states.push_back(std::move(features.second));
                    begin = i + 1;
                }
            }
        }
        // A truncated response can contain incomplete instances; keep raw text and
        // parsed geometry, but do not expose partially conditioned pixel masks.
        if (!result.truncated) decode_pixels(result, targets, states, request.options, request.task == Task::Segmentation);
        result.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        return result;
    }
};
Model::Model(const std::string& directory) : impl_(std::make_unique<Impl>(directory)) {}
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;
void Model::set_image(const uint8_t* bgr, int w, int h, int c) {
    require(bgr && w > 0 && h > 0 && c == 3 && size_t(w) <= std::numeric_limits<size_t>::max() / size_t(h) / 3,
            "WeVisOne requires a contiguous 3-channel BGR image");
    impl_->image = cv::Mat(h, w, CV_8UC3, const_cast<uint8_t*>(bgr)).clone();
    impl_->vision_ready = false; impl_->vision_features.clear(); impl_->cnn_image.clear();
}
Result Model::predict(const Request& request) { task_name(request.task); return impl_->predict(request); }
Result Model::predict_bgr(const uint8_t* bgr, int w, int h, const Request& request) { set_image(bgr, w, h); return predict(request); }
Result Model::matte_boxes(const std::vector<Box>& boxes, const Options& options) {
    require(!impl_->image.empty() && !boxes.empty(), "MatteBoxes requires an image and nonempty boxes");
    require(boxes.size() <= size_t(impl_->max_objects), "MatteBoxes exceeds pixel object capacity");
    for (const auto& box : boxes) validate_box(box, impl_->image.cols, impl_->image.rows);
    Request r; r.task = Task::Matting; r.query = {"object"}; r.options = options;
    return impl_->predict(r, &boxes);
}
}
