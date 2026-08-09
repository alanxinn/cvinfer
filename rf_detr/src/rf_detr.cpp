#include "rf_detr/rf_detr.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cvinfer::rf_detr {
namespace {

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

float half_to_float(std::uint16_t value) {
    const std::uint32_t sign = (value & 0x8000U) << 16;
    const std::uint32_t exponent = (value >> 10) & 0x1FU;
    const std::uint32_t mantissa = value & 0x3FFU;
    std::uint32_t bits = sign;
    if (exponent == 0) {
        if (mantissa != 0) {
            float result = static_cast<float>(mantissa) / 1024.0F;
            result = std::ldexp(result, -14);
            return sign ? -result : result;
        }
    } else if (exponent == 31) {
        bits |= 0x7F800000U | (mantissa << 13);
    } else {
        bits |= ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float result{};
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> tensor_values(const core::Tensor& tensor) {
    if (tensor.data_type == core::TensorDataType::Float16) {
        if (tensor.data.size() % sizeof(std::uint16_t) != 0) return {};
        const auto* source = reinterpret_cast<const std::uint16_t*>(tensor.data.data());
        std::vector<float> result(tensor.data.size() / sizeof(std::uint16_t));
        for (std::size_t i = 0; i < result.size(); ++i) result[i] = half_to_float(source[i]);
        return result;
    }
    if (tensor.data_type != core::TensorDataType::Float32 || tensor.data.size() % sizeof(float) != 0) return {};
    const auto* source = reinterpret_cast<const float*>(tensor.data.data());
    return std::vector<float>(source, source + tensor.data.size() / sizeof(float));
}

float sigmoid(float value) {
    if (value >= 0.0F) { const float e = std::exp(-value); return 1.0F / (1.0F + e); }
    const float e = std::exp(value); return e / (1.0F + e);
}

float iou(const Box& a, const Box& b) {
    const float x1 = (std::max)(a.x1, b.x1), y1 = (std::max)(a.y1, b.y1);
    const float x2 = (std::min)(a.x2, b.x2), y2 = (std::min)(a.y2, b.y2);
    const float inter = (std::max)(0.0F, x2 - x1) * (std::max)(0.0F, y2 - y1);
    const float area_a = (std::max)(0.0F, a.x2 - a.x1) * (std::max)(0.0F, a.y2 - a.y1);
    const float area_b = (std::max)(0.0F, b.x2 - b.x1) * (std::max)(0.0F, b.y2 - b.y1);
    return area_a + area_b - inter > 0.0F ? inter / (area_a + area_b - inter) : 0.0F;
}

struct Layout {
    int count{};
    int channels{};
    bool channel_first{};
};

Layout infer_layout(const core::Tensor& tensor, int preferred_channels) {
    if (tensor.shape.empty()) return {};
    std::vector<int64_t> shape = tensor.shape;
    if (shape.size() >= 3 && shape.front() == 1) shape.erase(shape.begin());
    if (shape.size() == 1) return {1, static_cast<int>(shape[0]), false};
    if (shape.size() != 2) return {};
    const int a = static_cast<int>(shape[0]), b = static_cast<int>(shape[1]);
    if (preferred_channels > 0 && a == preferred_channels) return {b, a, true};
    if (preferred_channels > 0 && b == preferred_channels) return {a, b, false};
    return a <= b ? Layout{b, a, true} : Layout{a, b, false};
}

float at(const std::vector<float>& values, const Layout& layout, int index, int channel) {
    if (layout.channel_first) return values[static_cast<std::size_t>(channel) * layout.count + index];
    return values[static_cast<std::size_t>(index) * layout.channels + channel];
}

struct MaskLayout {
    int queries{};
    int height{};
    int width{};
    int query_axis{};
};

MaskLayout infer_mask_layout(const core::Tensor& tensor, int queries) {
    if (tensor.shape.empty() || queries <= 0) return {};
    std::vector<int64_t> shape = tensor.shape;
    if (shape.size() >= 4 && shape.front() == 1) shape.erase(shape.begin());
    if (shape.size() != 3) return {};
    int query_axis = -1;
    for (int axis = 0; axis < 3; ++axis) {
        if (shape[static_cast<std::size_t>(axis)] == queries) {
            query_axis = axis;
            break;
        }
    }
    if (query_axis < 0) return {};
    int spatial[2]{};
    int spatial_index = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (axis != query_axis && spatial_index < 2)
            spatial[spatial_index++] = static_cast<int>(shape[static_cast<std::size_t>(axis)]);
    }
    if (spatial[0] <= 0 || spatial[1] <= 0) return {};
    return {queries, spatial[0], spatial[1], query_axis};
}

float mask_at(const std::vector<float>& values, const MaskLayout& layout, int query, int y, int x) {
    if (layout.query_axis == 0) {
        return values[(static_cast<std::size_t>(query) * layout.height + y) * layout.width + x];
    }
    if (layout.query_axis == 1) {
        return values[(static_cast<std::size_t>(y) * layout.queries + query) * layout.width + x];
    }
    return values[(static_cast<std::size_t>(y) * layout.width + x) * layout.queries + query];
}

} // namespace

std::vector<float> preprocess_bgr(const std::uint8_t* data, int width, int height, int channels,
                                  const Config& config) {
    if (!data || width <= 0 || height <= 0 || channels < 3 || config.input_width <= 0 || config.input_height <= 0)
        throw std::invalid_argument("Invalid image or RF-DETR input configuration");
    const int target_w = config.input_width, target_h = config.input_height;
    const float scale = config.letterbox
        ? (std::min)(static_cast<float>(target_w) / width, static_cast<float>(target_h) / height) : 1.0F;
    const int resized_w = config.letterbox ? (std::max)(1, static_cast<int>(std::round(width * scale))) : target_w;
    const int resized_h = config.letterbox ? (std::max)(1, static_cast<int>(std::round(height * scale))) : target_h;
    const float pad_x = (target_w - resized_w) * 0.5F, pad_y = (target_h - resized_h) * 0.5F;
    const std::size_t plane = static_cast<std::size_t>(target_w) * target_h;
    constexpr float mean[] = {0.485F, 0.456F, 0.406F};
    constexpr float stddev[] = {0.229F, 0.224F, 0.225F};
    std::vector<float> result(plane * 3, 0.0F);
    // RF-DETR's image transform uses antialiased/bilinear resizing.  Nearest
    // neighbour sampling noticeably changes the transformer features and can
    // drive all classification logits below the confidence threshold.
    for (int y = 0; y < resized_h; ++y) {
        const float source_y = (std::max)(0.0F, (std::min)(static_cast<float>(height - 1),
            ((static_cast<float>(y) + 0.5F) * height / resized_h) - 0.5F));
        const int y0 = static_cast<int>(std::floor(source_y));
        const int y1 = (std::min)(height - 1, y0 + 1);
        const float wy = source_y - std::floor(source_y);
        for (int x = 0; x < resized_w; ++x) {
            const float source_x = (std::max)(0.0F, (std::min)(static_cast<float>(width - 1),
                ((static_cast<float>(x) + 0.5F) * width / resized_w) - 0.5F));
            const int x0 = static_cast<int>(std::floor(source_x));
            const int x1 = (std::min)(width - 1, x0 + 1);
            const float wx = source_x - std::floor(source_x);
            const int dx = config.letterbox ? static_cast<int>(x + pad_x) : x;
            const int dy = config.letterbox ? static_cast<int>(y + pad_y) : y;
            if (dx < 0 || dx >= target_w || dy < 0 || dy >= target_h) continue;
            const auto* p00 = data + (static_cast<std::size_t>(y0) * width + x0) * channels;
            const auto* p01 = data + (static_cast<std::size_t>(y0) * width + x1) * channels;
            const auto* p10 = data + (static_cast<std::size_t>(y1) * width + x0) * channels;
            const auto* p11 = data + (static_cast<std::size_t>(y1) * width + x1) * channels;
            const float bgr[] = {
                (1.0F - wy) * ((1.0F - wx) * p00[0] + wx * p01[0]) + wy * ((1.0F - wx) * p10[0] + wx * p11[0]),
                (1.0F - wy) * ((1.0F - wx) * p00[1] + wx * p01[1]) + wy * ((1.0F - wx) * p10[1] + wx * p11[1]),
                (1.0F - wy) * ((1.0F - wx) * p00[2] + wx * p01[2]) + wy * ((1.0F - wx) * p10[2] + wx * p11[2])
            };
            const float rgb[] = {bgr[2] / 255.0F, bgr[1] / 255.0F, bgr[0] / 255.0F};
            for (int c = 0; c < 3; ++c) result[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(dy) * target_w + dx] = (rgb[c] - mean[c]) / stddev[c];
        }
    }
    return result;
}

std::vector<Detection> decode_detections(const core::Tensor& boxes_tensor, const core::Tensor& logits_tensor,
                                         const Config& config, int image_width, int image_height) {
    const auto boxes = tensor_values(boxes_tensor);
    const auto logits = tensor_values(logits_tensor);
    if (boxes.empty() || logits.empty() || image_width <= 0 || image_height <= 0) return {};
    const auto box_layout = infer_layout(boxes_tensor, 4);
    const auto score_layout = infer_layout(logits_tensor, config.class_count);
    if (box_layout.count <= 0 || box_layout.channels < 4 || score_layout.count <= 0 || score_layout.channels <= 0) return {};
    const int count = (std::min)(box_layout.count, score_layout.count);
    // RF-DETR checkpoints trained on COCO expose 91 logits: 90 categories plus
    // the final no-object channel.  The generic C API historically defaults to
    // 80 classes for YOLO, so infer the RF-DETR category count from the output
    // shape instead of silently dropping valid COCO categories.
    int classes = score_layout.channels;
    if (config.class_count > 0 && config.class_count < classes && score_layout.channels != 91)
        classes = config.class_count;
    if (score_layout.channels == 91) classes = 90;
    const float sx = config.letterbox ? (std::min)(static_cast<float>(config.input_width) / image_width, static_cast<float>(config.input_height) / image_height) : static_cast<float>(config.input_width) / image_width;
    const float sy = config.letterbox ? sx : static_cast<float>(config.input_height) / image_height;
    const float pad_x = config.letterbox ? (config.input_width - image_width * sx) * 0.5F : 0.0F;
    const float pad_y = config.letterbox ? (config.input_height - image_height * sy) * 0.5F : 0.0F;
    std::vector<Detection> result;
    result.reserve(static_cast<std::size_t>(count) * (std::min)(classes, 4));
    for (int i = 0; i < count; ++i) {
        float cx = at(boxes, box_layout, i, 0), cy = at(boxes, box_layout, i, 1);
        float w = at(boxes, box_layout, i, 2), h = at(boxes, box_layout, i, 3);
        if (std::fabs(cx) <= 1.5F && std::fabs(cy) <= 1.5F && std::fabs(w) <= 1.5F && std::fabs(h) <= 1.5F) {
            cx *= config.input_width; cy *= config.input_height; w *= config.input_width; h *= config.input_height;
        }
        const float x1 = (cx - w * 0.5F - pad_x) / sx, y1 = (cy - h * 0.5F - pad_y) / sy;
        const float x2 = (cx + w * 0.5F - pad_x) / sx, y2 = (cy + h * 0.5F - pad_y) / sy;
        if (x2 <= 0.0F || y2 <= 0.0F || x1 >= image_width || y1 >= image_height) continue;
        for (int class_id = 0; class_id < classes; ++class_id) {
            const float score = sigmoid(at(logits, score_layout, i, class_id));
            if (score < config.confidence_threshold) continue;
            result.push_back({{(std::max)(0.0F, x1), (std::max)(0.0F, y1),
                               (std::min)(static_cast<float>(image_width), x2),
                               (std::min)(static_cast<float>(image_height), y2), score, class_id}});
        }
    }
    // Match the official RF-DETR postprocess: rank query/class pairs globally
    // and keep at most num_select=300 results, without NMS.
    std::sort(result.begin(), result.end(), [](const Detection& a, const Detection& b) {
        return a.box.score > b.box.score;
    });
    if (result.size() > 300) result.resize(300);
    return result;
}

std::vector<Detection> nms(std::vector<Detection> candidates, float threshold, bool class_agnostic) {
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.box.score > b.box.score; });
    std::vector<Detection> result;
    for (const auto& item : candidates) {
        bool suppressed = false;
        for (const auto& kept : result) if ((class_agnostic || item.box.class_id == kept.box.class_id) && iou(item.box, kept.box) > threshold) { suppressed = true; break; }
        if (!suppressed) result.push_back(item);
    }
    return result;
}

std::vector<Segmentation> decode_segmentations(const std::vector<core::Tensor>& outputs,
                                                const Config& config, int image_width, int image_height) {
    const core::Tensor* boxes_tensor = nullptr;
    const core::Tensor* labels_tensor = nullptr;
    const core::Tensor* masks_tensor = nullptr;
    for (const auto& tensor : outputs) {
        if (tensor.name == "dets") boxes_tensor = &tensor;
        else if (tensor.name == "labels") labels_tensor = &tensor;
        else if (tensor.name == "masks") masks_tensor = &tensor;
    }
    for (const auto& tensor : outputs) {
        if (tensor.shape.size() == 4 && !masks_tensor) masks_tensor = &tensor;
        else if (tensor.shape.size() >= 3 && !boxes_tensor) {
            const int last = static_cast<int>(tensor.shape.back());
            const int previous = static_cast<int>(tensor.shape[tensor.shape.size() - 2]);
            if (last == 4 || previous == 4) boxes_tensor = &tensor;
        } else if (tensor.shape.size() >= 3 && !labels_tensor) {
            labels_tensor = &tensor;
        }
    }
    if (!boxes_tensor || !labels_tensor || !masks_tensor ||
        image_width <= 0 || image_height <= 0 || config.input_width <= 0 || config.input_height <= 0)
        return {};

    const auto boxes = tensor_values(*boxes_tensor);
    const auto labels = tensor_values(*labels_tensor);
    const auto masks = tensor_values(*masks_tensor);
    const auto box_layout = infer_layout(*boxes_tensor, 4);
    const auto label_layout = infer_layout(*labels_tensor, config.class_count);
    if (boxes.empty() || labels.empty() || masks.empty() ||
        box_layout.count <= 0 || box_layout.channels < 4 ||
        label_layout.count <= 0 || label_layout.channels <= 0)
        return {};
    const int query_count = (std::min)(box_layout.count, label_layout.count);
    const auto mask_layout = infer_mask_layout(*masks_tensor, query_count);
    if (mask_layout.queries <= 0) return {};

    int classes = label_layout.channels;
    if (classes == 91) classes = 90;
    else if (config.class_count > 0) classes = (std::min)(classes, config.class_count);
    if (classes <= 0) return {};

    const float sx = config.letterbox
        ? (std::min)(static_cast<float>(config.input_width) / image_width,
                     static_cast<float>(config.input_height) / image_height)
        : static_cast<float>(config.input_width) / image_width;
    const float sy = config.letterbox ? sx : static_cast<float>(config.input_height) / image_height;
    const float pad_x = config.letterbox ? (config.input_width - image_width * sx) * 0.5F : 0.0F;
    const float pad_y = config.letterbox ? (config.input_height - image_height * sy) * 0.5F : 0.0F;
    const float threshold = (std::min)(0.999999F, (std::max)(0.000001F, config.mask_threshold));
    const float threshold_logit = std::log(threshold / (1.0F - threshold));

    struct Candidate {
        Box box;
        int query{};
    };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(query_count));
    for (int query = 0; query < query_count; ++query) {
        int class_id = 0;
        float score = sigmoid(at(labels, label_layout, query, 0));
        for (int c = 1; c < classes; ++c) {
            const float candidate_score = sigmoid(at(labels, label_layout, query, c));
            if (candidate_score > score) {
                score = candidate_score;
                class_id = c;
            }
        }
        if (score < config.confidence_threshold) continue;
        float cx = at(boxes, box_layout, query, 0);
        float cy = at(boxes, box_layout, query, 1);
        float w = at(boxes, box_layout, query, 2);
        float h = at(boxes, box_layout, query, 3);
        if (std::fabs(cx) <= 1.5F && std::fabs(cy) <= 1.5F &&
            std::fabs(w) <= 1.5F && std::fabs(h) <= 1.5F) {
            cx *= config.input_width;
            cy *= config.input_height;
            w *= config.input_width;
            h *= config.input_height;
        }
        const float x1 = (cx - w * 0.5F - pad_x) / sx;
        const float y1 = (cy - h * 0.5F - pad_y) / sy;
        const float x2 = (cx + w * 0.5F - pad_x) / sx;
        const float y2 = (cy + h * 0.5F - pad_y) / sy;
        Box box{
            (std::max)(0.0F, x1), (std::max)(0.0F, y1),
            (std::min)(static_cast<float>(image_width), x2),
            (std::min)(static_cast<float>(image_height), y2),
            score, class_id
        };
        if (box.x2 <= box.x1 || box.y2 <= box.y1) continue;
        candidates.push_back({box, query});
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.box.score > b.box.score;
    });
    if (candidates.size() > 100) candidates.resize(100);

    auto sample_mask = [&](int query, float source_x, float source_y) {
        const float model_x = source_x * sx + pad_x;
        const float model_y = source_y * sy + pad_y;
        const float gx = (model_x + 0.5F) * mask_layout.width / config.input_width - 0.5F;
        const float gy = (model_y + 0.5F) * mask_layout.height / config.input_height - 0.5F;
        const float clamped_x = (std::max)(0.0F, (std::min)(static_cast<float>(mask_layout.width - 1), gx));
        const float clamped_y = (std::max)(0.0F, (std::min)(static_cast<float>(mask_layout.height - 1), gy));
        const int x0 = static_cast<int>(std::floor(clamped_x));
        const int y0 = static_cast<int>(std::floor(clamped_y));
        const int x1 = (std::min)(mask_layout.width - 1, x0 + 1);
        const int y1 = (std::min)(mask_layout.height - 1, y0 + 1);
        const float wx = clamped_x - x0;
        const float wy = clamped_y - y0;
        const float top = mask_at(masks, mask_layout, query, y0, x0) * (1.0F - wx) +
                          mask_at(masks, mask_layout, query, y0, x1) * wx;
        const float bottom = mask_at(masks, mask_layout, query, y1, x0) * (1.0F - wx) +
                             mask_at(masks, mask_layout, query, y1, x1) * wx;
        return top * (1.0F - wy) + bottom * wy;
    };

    std::vector<Segmentation> result;
    result.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const int left = (std::max)(0, static_cast<int>(std::floor(candidate.box.x1)));
        const int top = (std::max)(0, static_cast<int>(std::floor(candidate.box.y1)));
        const int right = (std::min)(image_width, static_cast<int>(std::ceil(candidate.box.x2)));
        const int bottom = (std::min)(image_height, static_cast<int>(std::ceil(candidate.box.y2)));
        if (right <= left || bottom <= top) continue;
        Segmentation item;
        item.box = candidate.box;
        item.mask_width = right - left;
        item.mask_height = bottom - top;
        item.mask.assign(static_cast<std::size_t>(item.mask_width) * item.mask_height, 0);
        for (int y = 0; y < item.mask_height; ++y) {
            for (int x = 0; x < item.mask_width; ++x) {
                const float logit = sample_mask(candidate.query, left + x + 0.5F, top + y + 0.5F);
                if (logit >= threshold_logit)
                    item.mask[static_cast<std::size_t>(y) * item.mask_width + x] = 255;
            }
        }
        result.push_back(std::move(item));
    }
    return result;
}

Detector::Detector(const std::string& path, Config config) : session_(core::TrtSession::load(path)), config_(std::move(config)) {
    const auto inputs = session_.input_names();
    if (!inputs.empty()) {
        if (config_.input_name.empty()) config_.input_name = inputs.front();
        const auto shape = session_.tensor_shape(config_.input_name);
        if (shape.size() == 4) {
            if (config_.input_height <= 0) config_.input_height = static_cast<int>(shape[2]);
            if (config_.input_width <= 0) config_.input_width = static_cast<int>(shape[3]);
        }
    }
}

std::vector<Detection> Detector::infer_image(const std::uint8_t* data, int width, int height, int channels) {
    if (config_.preprocess_backend == ComputeBackend::Cuda)
        throw std::runtime_error("RF-DETR CUDA preprocessing is not available; use --preprocess cpu or auto");
    const auto start = std::chrono::steady_clock::now();
    auto input = preprocess_bgr(data, width, height, channels, config_);
    timing_.preprocess_ms = elapsed_ms(start, std::chrono::steady_clock::now());
    auto result = infer(input, width, height);
    timing_.total_ms = elapsed_ms(start, std::chrono::steady_clock::now());
    return result;
}

std::vector<Detection> Detector::infer(const std::vector<float>& chw, int image_width, int image_height) {
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda)
        throw std::runtime_error("RF-DETR CUDA postprocessing is not available; use --postprocess cpu or auto");
    const auto inputs = session_.input_names();
    const std::string input = config_.input_name.empty() ? (inputs.empty() ? std::string{} : inputs.front()) : config_.input_name;
    if (input.empty()) throw std::runtime_error("RF-DETR TensorRT engine has no input tensor");
    session_.set_shape(input, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(true);
    session_.set_input(input, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = std::chrono::steady_clock::now();
    session_.infer();
    timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto post_start = std::chrono::steady_clock::now();
    const auto names = session_.output_names();
    const core::Tensor* boxes = nullptr; const core::Tensor* logits = nullptr;
    std::vector<core::Tensor> tensors;
    tensors.reserve(names.size());
    for (const auto& name : names) tensors.push_back(session_.get_output(name));
    for (const auto& tensor : tensors) {
        const auto shape = tensor.shape;
        int last = shape.empty() ? 0 : static_cast<int>(shape.back());
        int previous = shape.size() > 1 ? static_cast<int>(shape[shape.size() - 2]) : 0;
        const bool is_box_shape = last == 4 || previous == 4;
        if (!config_.output_name.empty() && tensor.name == config_.output_name) {
            if (is_box_shape) boxes = &tensor; else logits = &tensor;
        } else if (is_box_shape && !boxes) boxes = &tensor;
        else if (!logits) logits = &tensor;
    }
    if (!boxes || !logits) throw std::runtime_error("RF-DETR outputs must contain boxes and logits tensors");
    // RF-DETR is set-based; unlike YOLO, its exported queries are already de-duplicated.
    auto result = decode_detections(*boxes, *logits, config_, image_width, image_height);
    timing_.postprocess_ms = elapsed_ms(post_start, std::chrono::steady_clock::now());
    timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    return result;
}

Segmentor::Segmentor(const std::string& path, Config config)
    : session_(core::TrtSession::load(path)), config_(std::move(config)) {
    const auto inputs = session_.input_names();
    if (!inputs.empty()) {
        if (config_.input_name.empty()) config_.input_name = inputs.front();
        const auto shape = session_.tensor_shape(config_.input_name);
        if (shape.size() == 4) {
            if (config_.input_height <= 0) config_.input_height = static_cast<int>(shape[2]);
            if (config_.input_width <= 0) config_.input_width = static_cast<int>(shape[3]);
        }
    }
}

std::vector<Segmentation> Segmentor::infer_image(const std::uint8_t* data, int width, int height, int channels) {
    if (config_.preprocess_backend == ComputeBackend::Cuda)
        throw std::runtime_error("RF-DETR CUDA preprocessing is not available; use --preprocess cpu or auto");
    const auto total_start = std::chrono::steady_clock::now();
    auto input = preprocess_bgr(data, width, height, channels, config_);
    timing_.preprocess_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    auto result = infer(input, width, height);
    timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    return result;
}

std::vector<Segmentation> Segmentor::infer(const std::vector<float>& chw, int image_width, int image_height) {
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda)
        throw std::runtime_error("RF-DETR CUDA postprocessing is not available; use --postprocess cpu or auto");
    const auto inputs = session_.input_names();
    const std::string input = config_.input_name.empty() ? (inputs.empty() ? std::string{} : inputs.front()) : config_.input_name;
    if (input.empty()) throw std::runtime_error("RF-DETR segmentation engine has no input tensor");
    session_.set_shape(input, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(true);
    session_.set_input(input, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = std::chrono::steady_clock::now();
    session_.infer();
    timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = std::chrono::steady_clock::now();
    const auto outputs = session_.outputs();
    auto result = decode_segmentations(outputs, config_, image_width, image_height);
    timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
    timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    return result;
}

} // namespace cvinfer::rf_detr
