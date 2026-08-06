#include "yolo/yolo.h"
#include "yolo/yolo_cuda.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#include <stdexcept>
#include <utility>

namespace cvinfer::yolo {
namespace {
void require_cuda_backend() {
    if (!cuda_backend_available()) {
        throw std::runtime_error("CUDA backend DLL alanxinn_cvinfer_cuda.dll was not found");
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}

std::vector<float> preprocess_bgr(const std::uint8_t* data, int width, int height, int channels,
                                  const Config& config, Letterbox* transform) {
    if (!data || width <= 0 || height <= 0 || channels < 3 || config.input_width <= 0 || config.input_height <= 0)
        throw std::invalid_argument("Invalid image or YOLO input configuration");
    const float scale = (std::min)(static_cast<float>(config.input_width) / width,
                                   static_cast<float>(config.input_height) / height);
    const int resized_w = (std::max)(1, static_cast<int>(std::round(width * scale)));
    const int resized_h = (std::max)(1, static_cast<int>(std::round(height * scale)));
    const float pad_x = (config.input_width - resized_w) * 0.5F;
    const float pad_y = (config.input_height - resized_h) * 0.5F;
    if (transform) *transform = {scale, pad_x, pad_y, width, height};
    std::vector<float> chw(static_cast<std::size_t>(3) * config.input_width * config.input_height, 114.0F / 255.0F);
    for (int y = 0; y < resized_h; ++y) {
        const int sy = (std::min)(height - 1, static_cast<int>(y / scale));
        for (int x = 0; x < resized_w; ++x) {
            const int sx = (std::min)(width - 1, static_cast<int>(x / scale));
            const auto* pixel = data + (static_cast<std::size_t>(sy) * width + sx) * channels;
            const int dx = (std::min)(config.input_width - 1, (std::max)(0, static_cast<int>(x + pad_x)));
            const int dy = (std::min)(config.input_height - 1, (std::max)(0, static_cast<int>(y + pad_y)));
            const std::size_t plane = static_cast<std::size_t>(config.input_width) * config.input_height;
            chw[static_cast<std::size_t>(dy) * config.input_width + dx] = pixel[2] / 255.0F;
            chw[plane + static_cast<std::size_t>(dy) * config.input_width + dx] = pixel[1] / 255.0F;
            chw[2 * plane + static_cast<std::size_t>(dy) * config.input_width + dx] = pixel[0] / 255.0F;
        }
    }
    return chw;
}
namespace {
float normalize_score(float value) { return (value < 0.0F || value > 1.0F) ? (1.0F / (1.0F + std::exp(-value))) : value; }
float iou(const Box& a, const Box& b) {
    const float x1 = (std::max)(a.x1, b.x1), y1 = (std::max)(a.y1, b.y1);
    const float x2 = (std::min)(a.x2, b.x2), y2 = (std::min)(a.y2, b.y2);
    const float inter = (std::max)(0.0f, x2 - x1) * (std::max)(0.0f, y2 - y1);
    const float area_a = (std::max)(0.0f, a.x2 - a.x1) * (std::max)(0.0f, a.y2 - a.y1);
    const float area_b = (std::max)(0.0f, b.x2 - b.x1) * (std::max)(0.0f, b.y2 - b.y1);
    return inter > 0.0f && area_a + area_b - inter > 0.0f ? inter / (area_a + area_b - inter) : 0.0f;
}
float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = (value & 0x8000U) << 16;
    const std::uint32_t exponent = (value >> 10) & 0x1FU;
    const std::uint32_t mantissa = value & 0x3FFU;
    std::uint32_t bits;
    if (exponent == 0) bits = sign;
    else if (exponent == 31) bits = sign | 0x7F800000U | (mantissa << 13);
    else bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
std::vector<float> tensor_values(const core::Tensor& tensor) {
    if (tensor.data_type == core::TensorDataType::Float16) {
        const auto* values = reinterpret_cast<const std::uint16_t*>(tensor.data.data());
        std::vector<float> result(tensor.data.size() / sizeof(std::uint16_t));
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = half_to_float(values[i]);
        return result;
    }
    if (tensor.data.size() % sizeof(float) != 0) return {};
    const auto* values = reinterpret_cast<const float*>(tensor.data.data());
    return std::vector<float>(values, values + tensor.data.size() / sizeof(float));
}
float confidence_value(float value) { return (value >= 0.0F && value <= 1.0F) ? value : 1.0F / (1.0F + std::exp(-value)); }
std::vector<Detection> decode(const core::Tensor& tensor, const Config& cfg, int image_width, int image_height) {
    if ((tensor.data_type != core::TensorDataType::Float16 && tensor.data.size() % sizeof(float) != 0) || tensor.shape.size() < 2) return {};
    const int attrs = tensor.shape[tensor.shape.size() - 2] <= tensor.shape.back() ? static_cast<int>(tensor.shape[tensor.shape.size() - 2]) : static_cast<int>(tensor.shape.back());
    const int boxes = static_cast<int>(tensor.data.size() / sizeof(float)) / attrs;
    if (attrs < 5 || boxes <= 0) return {};
    const bool chw = tensor.shape.back() == boxes;
    const auto values_storage = tensor_values(tensor);
    const float* values = values_storage.data();
    const float scale = (std::min)(static_cast<float>(cfg.input_width) / image_width,
                                   static_cast<float>(cfg.input_height) / image_height);
    const int resized_w = (std::max)(1, static_cast<int>(std::round(image_width * scale)));
    const int resized_h = (std::max)(1, static_cast<int>(std::round(image_height * scale)));
    const float pad_x = (cfg.input_width - resized_w) * 0.5F;
    const float pad_y = (cfg.input_height - resized_h) * 0.5F;
    std::vector<Detection> result;
    for (int i = 0; i < boxes; ++i) {
        auto at = [&](int a) { return chw ? values[static_cast<std::size_t>(a) * boxes + i] : values[static_cast<std::size_t>(i) * attrs + a]; };
        int best_class = 0; float best_score = confidence_value(at(4));
        for (int c = 1; c < attrs - 4; ++c) {
            const float score = confidence_value(at(4 + c));
            if (score > best_score) { best_score = score; best_class = c; }
        }
        if (best_score < cfg.confidence_threshold) continue;
        const float cx = at(0), cy = at(1), w = at(2), h = at(3);
        const float x1 = (cx - w * 0.5F - pad_x) / scale;
        const float y1 = (cy - h * 0.5F - pad_y) / scale;
        const float x2 = (cx + w * 0.5F - pad_x) / scale;
        const float y2 = (cy + h * 0.5F - pad_y) / scale;
        Detection d;
        d.box = {
            (std::max)(0.0F, x1), (std::max)(0.0F, y1),
            (std::min)(static_cast<float>(image_width), x2),
            (std::min)(static_cast<float>(image_height), y2),
            best_score, best_class
        };
        if (d.box.x2 <= d.box.x1 || d.box.y2 <= d.box.y1) continue;
        result.push_back(d);
    }
    return nms(std::move(result), cfg.nms_threshold);
}
}
std::vector<Detection> decode_detections(const core::Tensor& output, const Config& config, int image_width, int image_height) { return decode(output, config, image_width, image_height); }

std::vector<Detection> nms(std::vector<Detection> candidates, float threshold, bool class_agnostic) {
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.box.score > b.box.score; });
    std::vector<Detection> kept;
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& accepted : kept) if ((class_agnostic || candidate.box.class_id == accepted.box.class_id) && iou(candidate.box, accepted.box) > threshold) { suppressed = true; break; }
        if (!suppressed) kept.push_back(candidate);
    }
    return kept;
}
std::vector<Segmentation> decode_segmentations(const std::vector<core::Tensor>& tensors,
                                               const Config& cfg, int image_width, int image_height) {
    const core::Tensor* det_tensor = nullptr;
    const core::Tensor* proto_tensor = nullptr;
    for (const auto& tensor : tensors) {
        if (tensor.data_type != core::TensorDataType::Float16 && tensor.data.size() % sizeof(float) != 0) continue;
        if (tensor.shape.size() == 4 && !proto_tensor) proto_tensor = &tensor;
        else if (tensor.shape.size() >= 2 && !det_tensor) det_tensor = &tensor;
    }
    if (!det_tensor || !proto_tensor || proto_tensor->shape.size() != 4) { return {}; }
    const int proto_channels = static_cast<int>(proto_tensor->shape[1]);
    const int proto_h = static_cast<int>(proto_tensor->shape[2]);
    const int proto_w = static_cast<int>(proto_tensor->shape[3]);
    const int attrs = det_tensor->shape.size() >= 3
        ? static_cast<int>(det_tensor->shape[det_tensor->shape.size() - 2] <= det_tensor->shape.back()
            ? det_tensor->shape[det_tensor->shape.size() - 2] : det_tensor->shape.back()) : 0;
    const std::size_t det_scalar_count = det_tensor->data_type == core::TensorDataType::Float16 ? det_tensor->data.size() / sizeof(std::uint16_t) : det_tensor->data.size() / sizeof(float);
    const int boxes = attrs > 0 ? static_cast<int>(det_scalar_count) / attrs : 0;
    const int classes = (std::max)(0, (std::min)(cfg.class_count, attrs - 4 - proto_channels));
    if (attrs <= 4 || boxes <= 0 || classes <= 0 || proto_channels <= 0 || proto_w <= 0 || proto_h <= 0) { return {}; }
    const bool chw = det_tensor->shape.back() == boxes;
    std::vector<float> det_storage;
    std::vector<float> proto_storage;
    const float* det = nullptr;
    const float* proto = nullptr;
    if (det_tensor->data_type == core::TensorDataType::Float16) {
        det_storage = tensor_values(*det_tensor);
        det = det_storage.data();
    } else {
        det = reinterpret_cast<const float*>(det_tensor->data.data());
    }
    if (proto_tensor->data_type == core::TensorDataType::Float16) {
        proto_storage = tensor_values(*proto_tensor);
        proto = proto_storage.data();
    } else {
        proto = reinterpret_cast<const float*>(proto_tensor->data.data());
    }
    const float scale = (std::min)(static_cast<float>(cfg.input_width) / image_width,
                                   static_cast<float>(cfg.input_height) / image_height);
    const float pad_x = (cfg.input_width - image_width * scale) * 0.5F;
    const float pad_y = (cfg.input_height - image_height * scale) * 0.5F;
    struct Candidate { Detection detection; std::vector<float> coeffs; };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>((std::min)(boxes, 256)));
    for (int i = 0; i < boxes; ++i) {
        auto at = [&](int a) { return chw ? det[static_cast<std::size_t>(a) * boxes + i] : det[static_cast<std::size_t>(i) * attrs + a]; };
        int class_id = 0; float score = confidence_value(at(4));
        for (int c = 1; c < classes; ++c) {
            const float class_score = confidence_value(at(4 + c));
            if (class_score > score) { score = class_score; class_id = c; }
        }
        if (score < cfg.confidence_threshold) continue;
        const float x1 = (at(0) - at(2) * 0.5F - pad_x) / scale;
        const float y1 = (at(1) - at(3) * 0.5F - pad_y) / scale;
        const float x2 = (at(0) + at(2) * 0.5F - pad_x) / scale;
        const float y2 = (at(1) + at(3) * 0.5F - pad_y) / scale;
        Candidate item;
        item.detection.box = { (std::max)(0.0F, x1), (std::max)(0.0F, y1),
                               (std::min)(static_cast<float>(image_width), x2),
                               (std::min)(static_cast<float>(image_height), y2), score, class_id };
        item.coeffs.resize(static_cast<std::size_t>(proto_channels));
        for (int c = 0; c < proto_channels; ++c) item.coeffs[static_cast<std::size_t>(c)] = at(4 + classes + c);
        candidates.push_back(std::move(item));
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.detection.box.score > b.detection.box.score; });
    std::vector<Candidate> kept;
    kept.reserve(candidates.size());
    for (auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& accepted : kept) if ((cfg.class_agnostic_nms || candidate.detection.box.class_id == accepted.detection.box.class_id) && iou(candidate.detection.box, accepted.detection.box) > cfg.nms_threshold) { suppressed = true; break; }
        if (!suppressed) kept.push_back(std::move(candidate));
    }
    const float proto_stride_x = static_cast<float>(cfg.input_width) / proto_w;
    const float proto_stride_y = static_cast<float>(cfg.input_height) / proto_h;
    const float mask_threshold = (std::min)(0.999999F, (std::max)(0.000001F, cfg.mask_threshold));
    const float mask_logit_threshold = std::log(mask_threshold / (1.0F - mask_threshold));
    int mask_pad_x = static_cast<int>(std::round(pad_x / proto_stride_x));
    int mask_pad_y = static_cast<int>(std::round(pad_y / proto_stride_y));
    mask_pad_x = (std::max)(0, (std::min)(mask_pad_x, proto_w / 2));
    mask_pad_y = (std::max)(0, (std::min)(mask_pad_y, proto_h / 2));
    const int raw_mask_w = (std::max)(1, proto_w - 2 * mask_pad_x);
    const int raw_mask_h = (std::max)(1, proto_h - 2 * mask_pad_y);
    std::vector<float> raw_mask(static_cast<std::size_t>(raw_mask_w) * raw_mask_h, 0.0F);
    struct SampleCoord { int lo; int hi; float weight; };
    std::vector<SampleCoord> x_lookup(static_cast<std::size_t>(image_width));
    std::vector<SampleCoord> y_lookup(static_cast<std::size_t>(image_height));
    for (int x = 0; x < image_width; ++x) {
        float source = (x + 0.5F) * scale / proto_stride_x - 0.5F;
        source = (std::max)(0.0F, (std::min)(static_cast<float>(raw_mask_w - 1), source));
        const int lo = static_cast<int>(std::floor(source));
        x_lookup[static_cast<std::size_t>(x)] = {lo, (std::min)(raw_mask_w - 1, lo + 1), source - lo};
    }
    for (int y = 0; y < image_height; ++y) {
        float source = (y + 0.5F) * scale / proto_stride_y - 0.5F;
        source = (std::max)(0.0F, (std::min)(static_cast<float>(raw_mask_h - 1), source));
        const int lo = static_cast<int>(std::floor(source));
        y_lookup[static_cast<std::size_t>(y)] = {lo, (std::min)(raw_mask_h - 1, lo + 1), source - lo};
    }
    std::vector<Segmentation> result;
    result.reserve(kept.size());
    for (const auto& candidate : kept) {
        const Box& box = candidate.detection.box;
        const int left = (std::max)(0, static_cast<int>(std::floor(box.x1)));
        const int top = (std::max)(0, static_cast<int>(std::floor(box.y1)));
        const int right = (std::min)(image_width, static_cast<int>(std::ceil(box.x2)));
        const int bottom = (std::min)(image_height, static_cast<int>(std::ceil(box.y2)));
        if (right <= left || bottom <= top) continue;
        const int raw_x0 = x_lookup[static_cast<std::size_t>(left)].lo;
        const int raw_x1 = x_lookup[static_cast<std::size_t>(right - 1)].hi;
        const int raw_y0 = y_lookup[static_cast<std::size_t>(top)].lo;
        const int raw_y1 = y_lookup[static_cast<std::size_t>(bottom - 1)].hi;
        const int roi_w = (std::max)(1, raw_x1 - raw_x0 + 1);
        const int roi_h = (std::max)(1, raw_y1 - raw_y0 + 1);
        for (int y = raw_y0; y <= raw_y1; ++y) {
            float* mask_row = raw_mask.data() + static_cast<std::size_t>(y) * raw_mask_w + raw_x0;
            std::fill_n(mask_row, roi_w, 0.0F);
        }
        for (int c = 0; c < proto_channels; ++c) {
            const float coefficient = candidate.coeffs[static_cast<std::size_t>(c)];
            const float* proto_channel = proto +
                static_cast<std::size_t>(c) * proto_h * proto_w +
                static_cast<std::size_t>(mask_pad_y + raw_y0) * proto_w + mask_pad_x + raw_x0;
            for (int y = 0; y < roi_h; ++y) {
                const float* proto_row = proto_channel + static_cast<std::size_t>(y) * proto_w;
                float* mask_row = raw_mask.data() +
                    static_cast<std::size_t>(raw_y0 + y) * raw_mask_w + raw_x0;
                for (int x = 0; x < roi_w; ++x) mask_row[x] += coefficient * proto_row[x];
            }
        }
        Segmentation item;
        item.box = box;
        item.mask_width = right - left;
        item.mask_height = bottom - top;
        item.mask.assign(static_cast<std::size_t>(item.mask_width) * item.mask_height, 0);
        for (int y = 0; y < item.mask_height; ++y) {
            const SampleCoord sy = y_lookup[static_cast<std::size_t>(top + y)];
            const float* row0 = raw_mask.data() + static_cast<std::size_t>(sy.lo) * raw_mask_w;
            const float* row1 = raw_mask.data() + static_cast<std::size_t>(sy.hi) * raw_mask_w;
            std::uint8_t* output_row = item.mask.data() + static_cast<std::size_t>(y) * item.mask_width;
            for (int x = 0; x < item.mask_width; ++x) {
                const SampleCoord sx = x_lookup[static_cast<std::size_t>(left + x)];
                const float top_value = row0[sx.lo] * (1.0F - sx.weight) + row0[sx.hi] * sx.weight;
                const float bottom_value = row1[sx.lo] * (1.0F - sx.weight) + row1[sx.hi] * sx.weight;
                const float logit = top_value * (1.0F - sy.weight) + bottom_value * sy.weight;
                if (logit >= mask_logit_threshold) output_row[x] = 255;
            }
        }
        result.push_back(std::move(item));
    }
    return result;
}
Detector::Detector(const std::string& path, Config config) : session_(core::TrtSession::load(path)), config_(config) {
    const auto names = session_.input_names();
    if (!names.empty()) {
        if (config_.input_name.empty()) config_.input_name = names.front();
        const auto shape = session_.tensor_shape(config_.input_name);
        if ((config_.input_width <= 0 || config_.input_height <= 0) &&
            shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
            config_.input_height = static_cast<int>(shape[2]);
            config_.input_width = static_cast<int>(shape[3]);
        }
    }
}
std::vector<Detection> Detector::infer_image(const std::uint8_t* data, int width, int height, int channels) {
    const auto total_start = std::chrono::steady_clock::now();
    if (config_.preprocess_backend == ComputeBackend::Cuda) {
        require_cuda_backend();
        if (!cuda_preprocess_) cuda_preprocess_ = cuda_backend::create_preprocess_context();
        Letterbox transform;
        const auto preprocess_start = std::chrono::steady_clock::now();
        const float* device_data = cuda_backend::preprocess_bgr(*cuda_preprocess_, data, width, height, channels,
                                                                config_, session_.cuda_stream(), &transform);
        timing_.preprocess_ms = elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
        auto result = infer_device(device_data, cuda_backend::preprocess_bytes(config_), width, height);
        timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        return result;
    }
    auto chw = preprocess_bgr(data, width, height, channels, config_);
    timing_.preprocess_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    auto result = infer(chw, width, height);
    timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    return result;
}
std::vector<Detection> Detector::infer(const std::vector<float>& chw, int image_width, int image_height) {
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const auto inputs = session_.input_names();
    const std::string input_name = config_.input_name.empty() ? (inputs.empty() ? std::string{} : inputs.front()) : config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input(input_name, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = std::chrono::steady_clock::now();
    session_.infer();
    timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<Detection> result;
    const std::string output_name = !config_.output_name.empty()
        ? config_.output_name
        : (session_.output_names().empty() ? std::string{} : session_.output_names().front());
    if (output_name.empty()) throw std::runtime_error("TensorRT engine has no output tensor");
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        result = cuda_backend::decode_detections_device(session_.device_output(output_name),
            session_.device_output_shape(output_name), session_.device_output_type(output_name),
            config_, image_width, image_height, session_.cuda_stream());
    } else {
        result = decode_detections(session_.get_output(output_name), config_, image_width, image_height);
    }
    timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
    timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    return result;
}
std::vector<Detection> Detector::infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height) {
    if (!device_data || bytes == 0) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const auto inputs = session_.input_names();
    const std::string input_name = config_.input_name.empty() ? (inputs.empty() ? std::string{} : inputs.front()) : config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input_device(input_name, device_data, bytes);
    const auto inference_start = std::chrono::steady_clock::now();
    session_.infer();
    timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = std::chrono::steady_clock::now();
    const std::string output_name = !config_.output_name.empty()
        ? config_.output_name
        : (session_.output_names().empty() ? std::string{} : session_.output_names().front());
    if (output_name.empty()) throw std::runtime_error("TensorRT engine has no output tensor");
    std::vector<Detection> result;
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        result = cuda_backend::decode_detections_device(session_.device_output(output_name),
            session_.device_output_shape(output_name), session_.device_output_type(output_name),
            config_, image_width, image_height, session_.cuda_stream());
    } else {
        result = decode_detections(session_.get_output(output_name), config_, image_width, image_height);
    }
    timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
    timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    return result;
}
Segmentor::Segmentor(const std::string& path, Config config) : session_(core::TrtSession::load(path)), config_(config) {
    const auto names = session_.input_names();
    if (!names.empty()) {
        if (config_.input_name.empty()) config_.input_name = names.front();
        const auto shape = session_.tensor_shape(config_.input_name);
        if ((config_.input_width <= 0 || config_.input_height <= 0) &&
            shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
            config_.input_height = static_cast<int>(shape[2]);
            config_.input_width = static_cast<int>(shape[3]);
        }
    }
}
std::vector<Segmentation> Segmentor::infer_image(const std::uint8_t* data, int width, int height, int channels) {
    const auto total_start = std::chrono::steady_clock::now();
    if (config_.preprocess_backend == ComputeBackend::Cuda) {
        require_cuda_backend();
        if (!cuda_preprocess_) cuda_preprocess_ = cuda_backend::create_preprocess_context();
        Letterbox transform;
        const auto preprocess_start = std::chrono::steady_clock::now();
        const float* device_data = cuda_backend::preprocess_bgr(*cuda_preprocess_, data, width, height, channels,
                                                                config_, session_.cuda_stream(), &transform);
        timing_.preprocess_ms = elapsed_ms(preprocess_start, std::chrono::steady_clock::now());
        auto result = infer_device(device_data, cuda_backend::preprocess_bytes(config_), width, height);
        timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        return result;
    }
    auto chw = preprocess_bgr(data, width, height, channels, config_);
    timing_.preprocess_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    auto result = infer(chw, width, height);
    timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    return result;
}
std::vector<Segmentation> Segmentor::infer(const std::vector<float>& chw, int image_width, int image_height) {
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const auto inputs = session_.input_names();
    const std::string input_name = config_.input_name.empty() ? (inputs.empty() ? std::string{} : inputs.front()) : config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input(input_name, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = std::chrono::steady_clock::now();
    session_.infer();
    timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<Segmentation> result;
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        if (!cuda_postprocess_) cuda_postprocess_ = cuda_backend::create_segmentation_postprocess_context();
        std::string det_name = config_.output_name;
        std::string proto_name = config_.prototype_name;
        for (const auto& name : session_.output_names()) {
            const auto shape = session_.device_output_shape(name);
            if (shape.size() == 4 && proto_name.empty()) proto_name = name;
            else if (shape.size() >= 2 && det_name.empty()) det_name = name;
        }
        if (det_name.empty() || proto_name.empty()) throw std::runtime_error("TensorRT segmentation outputs are incomplete");
        result = cuda_backend::decode_segmentations_device(session_.device_output(det_name),
            session_.device_output_shape(det_name), session_.device_output_type(det_name),
            session_.device_output(proto_name), session_.device_output_shape(proto_name),
            session_.device_output_type(proto_name), config_, image_width, image_height,
            session_.cuda_stream(), *cuda_postprocess_);
    } else {
        auto tensors = session_.outputs();
        if (!config_.output_name.empty() || !config_.prototype_name.empty()) {
            tensors.clear();
            if (!config_.output_name.empty()) tensors.push_back(session_.get_output(config_.output_name));
            if (!config_.prototype_name.empty()) tensors.push_back(session_.get_output(config_.prototype_name));
        }
        result = decode_segmentations(tensors, config_, image_width, image_height);
    }
    timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
    timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    return result;
}
std::vector<Segmentation> Segmentor::infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height) {
    if (!device_data || bytes == 0) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const auto inputs = session_.input_names();
    const std::string input_name = config_.input_name.empty() ? (inputs.empty() ? std::string{} : inputs.front()) : config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input_device(input_name, device_data, bytes);
    const auto inference_start = std::chrono::steady_clock::now();
    session_.infer();
    timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<Segmentation> result;
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        std::string det_name = config_.output_name;
        std::string proto_name = config_.prototype_name;
        for (const auto& name : session_.output_names()) {
            const auto shape = session_.device_output_shape(name);
            if (shape.size() == 4 && proto_name.empty()) proto_name = name;
            else if (shape.size() >= 2 && det_name.empty()) det_name = name;
        }
        if (det_name.empty() || proto_name.empty()) throw std::runtime_error("TensorRT segmentation outputs are incomplete");
        if (!cuda_postprocess_) cuda_postprocess_ = cuda_backend::create_segmentation_postprocess_context();
        result = cuda_backend::decode_segmentations_device(session_.device_output(det_name),
            session_.device_output_shape(det_name), session_.device_output_type(det_name),
            session_.device_output(proto_name), session_.device_output_shape(proto_name),
            session_.device_output_type(proto_name), config_, image_width, image_height,
            session_.cuda_stream(), *cuda_postprocess_);
    } else {
        auto tensors = session_.outputs();
        if (!config_.output_name.empty() || !config_.prototype_name.empty()) {
            tensors.clear();
            if (!config_.output_name.empty()) tensors.push_back(session_.get_output(config_.output_name));
            if (!config_.prototype_name.empty()) tensors.push_back(session_.get_output(config_.prototype_name));
        }
        result = decode_segmentations(tensors, config_, image_width, image_height);
    }
    timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
    timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    return result;
}} // namespace cvinfer::yolo






























