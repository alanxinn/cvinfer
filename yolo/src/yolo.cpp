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
        throw std::invalid_argument("Invalid image or YOLO11/YOLO26 input configuration");
    const float scale = (std::min)(static_cast<float>(config.input_width) / width,
                                   static_cast<float>(config.input_height) / height);
    const int resized_w = (std::max)(1, static_cast<int>(std::round(width * scale)));
    const int resized_h = (std::max)(1, static_cast<int>(std::round(height * scale)));
    const float pad_x = (config.input_width - resized_w) * 0.5F;
    const float pad_y = (config.input_height - resized_h) * 0.5F;
    if (transform) *transform = {scale, pad_x, pad_y, width, height};
    const std::size_t plane = static_cast<std::size_t>(config.input_width) * config.input_height;
    std::vector<float> chw(plane * 3, 114.0F / 255.0F);
    std::vector<int> source_x(static_cast<std::size_t>(resized_w));
    std::vector<int> destination_x(static_cast<std::size_t>(resized_w));
    for (int x = 0; x < resized_w; ++x) {
        source_x[static_cast<std::size_t>(x)] = (std::min)(width - 1, static_cast<int>(x / scale));
        destination_x[static_cast<std::size_t>(x)] =
            (std::min)(config.input_width - 1, (std::max)(0, static_cast<int>(x + pad_x)));
    }
    for (int y = 0; y < resized_h; ++y) {
        const int sy = (std::min)(height - 1, static_cast<int>(y / scale));
        const int dy = (std::min)(config.input_height - 1, (std::max)(0, static_cast<int>(y + pad_y)));
        const auto* source_row = data + static_cast<std::size_t>(sy) * width * channels;
        float* red_row = chw.data() + static_cast<std::size_t>(dy) * config.input_width;
        float* green_row = red_row + plane;
        float* blue_row = green_row + plane;
        for (int x = 0; x < resized_w; ++x) {
            const auto* pixel = source_row + static_cast<std::size_t>(source_x[static_cast<std::size_t>(x)]) * channels;
            const int dx = destination_x[static_cast<std::size_t>(x)];
            red_row[dx] = pixel[2] / 255.0F;
            green_row[dx] = pixel[1] / 255.0F;
            blue_row[dx] = pixel[0] / 255.0F;
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
float box_area(const Box& box) {
    return (std::max)(0.0F, box.x2 - box.x1) * (std::max)(0.0F, box.y2 - box.y1);
}
float iou(const Box& a, const Box& b, float area_a, float area_b) {
    const float x1 = (std::max)(a.x1, b.x1), y1 = (std::max)(a.y1, b.y1);
    const float x2 = (std::min)(a.x2, b.x2), y2 = (std::min)(a.y2, b.y2);
    const float inter = (std::max)(0.0F, x2 - x1) * (std::max)(0.0F, y2 - y1);
    const float union_area = area_a + area_b - inter;
    return inter > 0.0F && union_area > 0.0F ? inter / union_area : 0.0F;
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
    if ((tensor.data_type != core::TensorDataType::Float16 && tensor.data_type != core::TensorDataType::Float32) || tensor.shape.size() < 2) return {};
    const int attrs = tensor.shape[tensor.shape.size() - 2] <= tensor.shape.back() ? static_cast<int>(tensor.shape[tensor.shape.size() - 2]) : static_cast<int>(tensor.shape.back());
    const std::size_t scalar_count = tensor.data_type == core::TensorDataType::Float16
        ? tensor.data.size() / sizeof(std::uint16_t) : tensor.data.size() / sizeof(float);
    if (tensor.data_type == core::TensorDataType::Float32 && tensor.data.size() % sizeof(float) != 0) return {};
    const int boxes = attrs > 0 ? static_cast<int>(scalar_count / static_cast<std::size_t>(attrs)) : 0;
    if (attrs < 5 || boxes <= 0) return {};
    const bool chw = tensor.shape.back() == boxes;
    std::vector<float> values_storage;
    const float* values = nullptr;
    if (tensor.data_type == core::TensorDataType::Float16) {
        values_storage = tensor_values(tensor);
        values = values_storage.data();
    } else {
        values = reinterpret_cast<const float*>(tensor.data.data());
    }
    const float scale = (std::min)(static_cast<float>(cfg.input_width) / image_width,
                                   static_cast<float>(cfg.input_height) / image_height);
    const int resized_w = (std::max)(1, static_cast<int>(std::round(image_width * scale)));
    const int resized_h = (std::max)(1, static_cast<int>(std::round(image_height * scale)));
    const float pad_x = (cfg.input_width - resized_w) * 0.5F;
    const float pad_y = (cfg.input_height - resized_h) * 0.5F;
    std::vector<Detection> result;
    result.reserve(static_cast<std::size_t>(boxes));
    // YOLO26 end2end exports top-k rows as [x1,y1,x2,y2,score,class].
    // YOLO11/raw exports use [cx,cy,w,h,class_scores...].
    const bool end2end = attrs == 6 && boxes <= 1000;
    for (int i = 0; i < boxes; ++i) {
        auto at = [&](int a) { return chw ? values[static_cast<std::size_t>(a) * boxes + i] : values[static_cast<std::size_t>(i) * attrs + a]; };
        int best_class = 0;
        float best_score = 0.0F;
        float x1{}, y1{}, x2{}, y2{};
        if (end2end) {
            best_score = confidence_value(at(4));
            best_class = static_cast<int>(std::lround(at(5)));
            x1 = at(0); y1 = at(1); x2 = at(2); y2 = at(3);
        } else {
            best_score = confidence_value(at(4));
            for (int c = 1; c < attrs - 4; ++c) {
                const float score = confidence_value(at(4 + c));
                if (score > best_score) { best_score = score; best_class = c; }
            }
            const float cx = at(0), cy = at(1), w = at(2), h = at(3);
            x1 = cx - w * 0.5F; y1 = cy - h * 0.5F;
            x2 = cx + w * 0.5F; y2 = cy + h * 0.5F;
        }
        if (best_score < cfg.confidence_threshold) continue;
        x1 = (x1 - pad_x) / scale;
        y1 = (y1 - pad_y) / scale;
        x2 = (x2 - pad_x) / scale;
        y2 = (y2 - pad_y) / scale;
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
    if (end2end) {
        // YOLO26's one-to-one head already performs score filtering and NMS.
        return result;
    }
    return nms(std::move(result), cfg.nms_threshold, cfg.class_agnostic_nms);
}
}
std::vector<Detection> decode_detections(const core::Tensor& output, const Config& config, int image_width, int image_height) { return decode(output, config, image_width, image_height); }

std::vector<Detection> nms(std::vector<Detection> candidates, float threshold, bool class_agnostic) {
    if (candidates.size() < 2) return candidates;
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) { return a.box.score > b.box.score; });
    std::vector<Detection> kept;
    kept.reserve(candidates.size());
    std::vector<float> kept_areas;
    kept_areas.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        const float candidate_area = box_area(candidate.box);
        bool suppressed = false;
        for (std::size_t i = 0; i < kept.size(); ++i) {
            const auto& accepted = kept[i];
            if ((class_agnostic || candidate.box.class_id == accepted.box.class_id) &&
                iou(candidate.box, accepted.box, candidate_area, kept_areas[i]) > threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            kept_areas.push_back(candidate_area);
        }
    }
    return kept;
}
namespace {
std::vector<Segmentation> decode_segmentations_impl(const std::vector<const core::Tensor*>& tensors,
                                                    const Config& cfg, int image_width, int image_height) {
    const core::Tensor* det_tensor = nullptr;
    const core::Tensor* proto_tensor = nullptr;
    for (const auto* tensor_ptr : tensors) {
        if (!tensor_ptr) continue;
        const auto& tensor = *tensor_ptr;
        if (tensor.data_type != core::TensorDataType::Float16 && tensor.data_type != core::TensorDataType::Float32) continue;
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
    if (det_tensor->data_type != core::TensorDataType::Float16 && det_tensor->data_type != core::TensorDataType::Float32) return {};
    if (proto_tensor->data_type != core::TensorDataType::Float16 && proto_tensor->data_type != core::TensorDataType::Float32) return {};
    const std::size_t det_scalar_count = det_tensor->data_type == core::TensorDataType::Float16 ? det_tensor->data.size() / sizeof(std::uint16_t) : det_tensor->data.size() / sizeof(float);
    const int boxes = attrs > 0 ? static_cast<int>(det_scalar_count) / attrs : 0;
    // YOLO26 segmentation exports exactly [xyxy, score, class, mask_coeffs].
    // Requiring the exact attribute width avoids mistaking a low-resolution
    // YOLO11 segmentation tensor for an end-to-end tensor.
    const bool end2end = boxes <= 1000 && attrs == 6 + proto_channels;
    // The exported tensor carries one score per class; derive the class count
    // from its attribute width instead of assuming COCO-80.
    const int classes = end2end ? 0 : (std::max)(0, attrs - 4 - proto_channels);
    if (attrs <= 4 || boxes <= 0 || (!end2end && classes <= 0) || proto_channels <= 0 || proto_w <= 0 || proto_h <= 0) { return {}; }
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
    const int resized_w = (std::max)(1, static_cast<int>(std::round(image_width * scale)));
    const int resized_h = (std::max)(1, static_cast<int>(std::round(image_height * scale)));
    const float pad_x = (cfg.input_width - resized_w) * 0.5F;
    const float pad_y = (cfg.input_height - resized_h) * 0.5F;
    struct Candidate { Detection detection; std::size_t coeff_offset{}; };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>((std::min)(boxes, 256)));
    std::vector<float> coefficient_storage;
    coefficient_storage.reserve(static_cast<std::size_t>((std::min)(boxes, 256)) * proto_channels);
    for (int i = 0; i < boxes; ++i) {
        auto at = [&](int a) { return chw ? det[static_cast<std::size_t>(a) * boxes + i] : det[static_cast<std::size_t>(i) * attrs + a]; };
        int class_id = 0;
        float score = confidence_value(at(4));
        float x1{}, y1{}, x2{}, y2{};
        int coefficient_start = 0;
        if (end2end) {
            class_id = static_cast<int>(std::lround(at(5)));
            x1 = at(0); y1 = at(1); x2 = at(2); y2 = at(3);
            coefficient_start = 6;
        } else {
            for (int c = 1; c < classes; ++c) {
                const float class_score = confidence_value(at(4 + c));
                if (class_score > score) { score = class_score; class_id = c; }
            }
            const float cx = at(0), cy = at(1), w = at(2), h = at(3);
            x1 = cx - w * 0.5F; y1 = cy - h * 0.5F;
            x2 = cx + w * 0.5F; y2 = cy + h * 0.5F;
            coefficient_start = 4 + classes;
        }
        if (score < cfg.confidence_threshold) continue;
        x1 = (x1 - pad_x) / scale;
        y1 = (y1 - pad_y) / scale;
        x2 = (x2 - pad_x) / scale;
        y2 = (y2 - pad_y) / scale;
        Candidate item;
        item.detection.box = { (std::max)(0.0F, x1), (std::max)(0.0F, y1),
                               (std::min)(static_cast<float>(image_width), x2),
                               (std::min)(static_cast<float>(image_height), y2), score, class_id };
        item.coeff_offset = coefficient_storage.size();
        coefficient_storage.resize(item.coeff_offset + static_cast<std::size_t>(proto_channels));
        for (int c = 0; c < proto_channels; ++c)
            coefficient_storage[item.coeff_offset + static_cast<std::size_t>(c)] = at(coefficient_start + c);
        candidates.push_back(std::move(item));
    }
    std::vector<Candidate> kept;
    kept.reserve(candidates.size());
    std::vector<float> kept_areas;
    kept_areas.reserve(candidates.size());
    if (end2end) {
        // YOLO26's one-to-one head already performs top-k selection and does not require NMS.
        kept = std::move(candidates);
    } else {
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) { return a.detection.box.score > b.detection.box.score; });
        for (auto& candidate : candidates) {
            const float candidate_area = box_area(candidate.detection.box);
            bool suppressed = false;
            for (std::size_t i = 0; i < kept.size(); ++i) {
                const auto& accepted = kept[i];
                if ((cfg.class_agnostic_nms || candidate.detection.box.class_id == accepted.detection.box.class_id) &&
                    iou(candidate.detection.box, accepted.detection.box, candidate_area, kept_areas[i]) > cfg.nms_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.push_back(std::move(candidate));
                kept_areas.push_back(candidate_area);
            }
        }
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
        const float* coeffs = coefficient_storage.data() + candidate.coeff_offset;
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
            const float coefficient = coeffs[static_cast<std::size_t>(c)];
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

}

std::vector<Pose> decode_poses(const core::Tensor& tensor, const Config& cfg, int image_width, int image_height) {
    if ((tensor.data_type != core::TensorDataType::Float16 && tensor.data_type != core::TensorDataType::Float32) ||
        tensor.shape.size() < 2 || image_width <= 0 || image_height <= 0) return {};
    const int dim_a = static_cast<int>(tensor.shape[tensor.shape.size() - 2]);
    const int dim_b = static_cast<int>(tensor.shape.back());
    const int attrs = (std::min)(dim_a, dim_b);
    const std::size_t scalar_count = tensor.data_type == core::TensorDataType::Float16
        ? tensor.data.size() / sizeof(std::uint16_t) : tensor.data.size() / sizeof(float);
    if (tensor.data_type == core::TensorDataType::Float32 && tensor.data.size() % sizeof(float) != 0) return {};
    const int boxes = attrs > 0 ? static_cast<int>(scalar_count / static_cast<std::size_t>(attrs)) : 0;
    if (attrs < 6 || boxes <= 0) return {};
    const bool chw = dim_b == boxes;
    std::vector<float> storage;
    const float* values = nullptr;
    if (tensor.data_type == core::TensorDataType::Float16) {
        storage = tensor_values(tensor);
        values = storage.data();
    } else {
        values = reinterpret_cast<const float*>(tensor.data.data());
    }
    const float scale = (std::min)(static_cast<float>(cfg.input_width) / image_width,
                                   static_cast<float>(cfg.input_height) / image_height);
    const int resized_w = (std::max)(1, static_cast<int>(std::round(image_width * scale)));
    const int resized_h = (std::max)(1, static_cast<int>(std::round(image_height * scale)));
    const float pad_x = (cfg.input_width - resized_w) * 0.5F;
    const float pad_y = (cfg.input_height - resized_h) * 0.5F;
    int keypoint_dims = cfg.keypoint_dims > 0 ? cfg.keypoint_dims : 3;
    int keypoint_count = cfg.keypoint_count > 0 ? cfg.keypoint_count : 0;
    int keypoint_values = keypoint_count * keypoint_dims;
    // YOLO26 pose exports exactly [xyxy, score, class, keypoints...].
    // The configured layout is preferred; otherwise infer the common
    // x/y/score (3-value) layout from the exported attribute width. Raw
    // YOLO11 pose tensors normally have thousands of anchors and therefore
    // do not satisfy the end-to-end query-count heuristic.
    const int end2end_extra = attrs - 6;
    bool end2end = boxes <= 1000 && end2end_extra > 0 &&
        (end2end_extra == keypoint_values ||
         (end2end_extra % 3 == 0 && end2end_extra / 3 <= 128));
    if (end2end && end2end_extra != keypoint_values) {
        keypoint_dims = 3;
        keypoint_count = end2end_extra / 3;
        keypoint_values = keypoint_count * keypoint_dims;
    }
    const int classes = end2end ? 0 : attrs - 4 - keypoint_values;
    const int required_attrs = end2end ? 6 + keypoint_values : 4 + classes + keypoint_values;
    if ((!end2end && classes <= 0) || keypoint_count <= 0 || keypoint_dims < 2 || attrs < required_attrs) return {};
    auto at = [&](int i, int a) {
        return chw ? values[static_cast<std::size_t>(a) * boxes + i]
                   : values[static_cast<std::size_t>(i) * attrs + a];
    };
    struct Candidate { Pose pose; };
    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>((std::min)(boxes, 300)));
    for (int i = 0; i < boxes; ++i) {
        int class_id = 0;
        float score = confidence_value(at(i, 4));
        int keypoint_start = 0;
        float x1{}, y1{}, x2{}, y2{};
        if (end2end) {
            class_id = static_cast<int>(std::lround(at(i, 5)));
            x1 = at(i, 0); y1 = at(i, 1); x2 = at(i, 2); y2 = at(i, 3);
            keypoint_start = 6;
        } else {
            for (int c = 1; c < classes; ++c) {
                const float candidate_score = confidence_value(at(i, 4 + c));
                if (candidate_score > score) { score = candidate_score; class_id = c; }
            }
            const float cx = at(i, 0), cy = at(i, 1), w = at(i, 2), h = at(i, 3);
            x1 = cx - w * 0.5F; y1 = cy - h * 0.5F;
            x2 = cx + w * 0.5F; y2 = cy + h * 0.5F;
            keypoint_start = 4 + classes;
        }
        if (score < cfg.confidence_threshold) continue;
        x1 = (x1 - pad_x) / scale; y1 = (y1 - pad_y) / scale;
        x2 = (x2 - pad_x) / scale; y2 = (y2 - pad_y) / scale;
        Pose item;
        item.keypoint_dims = keypoint_dims;
        item.box = { (std::max)(0.0F, x1), (std::max)(0.0F, y1),
                     (std::min)(static_cast<float>(image_width), x2),
                     (std::min)(static_cast<float>(image_height), y2), score, class_id };
        if (item.box.x2 <= item.box.x1 || item.box.y2 <= item.box.y1) continue;
        item.keypoints.resize(static_cast<std::size_t>(keypoint_count));
        for (int k = 0; k < keypoint_count; ++k) {
            const int offset = keypoint_start + k * keypoint_dims;
            const float kx = (at(i, offset) - pad_x) / scale;
            const float ky = (at(i, offset + 1) - pad_y) / scale;
            const float ks = keypoint_dims >= 3 ? confidence_value(at(i, offset + 2)) : 1.0F;
            item.keypoints[static_cast<std::size_t>(k)] = {
                (std::max)(0.0F, (std::min)(static_cast<float>(image_width), kx)),
                (std::max)(0.0F, (std::min)(static_cast<float>(image_height), ky)), ks};
        }
        candidates.push_back({std::move(item)});
    }
    if (!end2end) {
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.pose.box.score > b.pose.box.score;
        });
        std::vector<Candidate> kept;
        kept.reserve(candidates.size());
        std::vector<float> kept_areas;
        kept_areas.reserve(candidates.size());
        for (auto& candidate : candidates) {
            const float candidate_area = box_area(candidate.pose.box);
            bool suppressed = false;
            for (std::size_t i = 0; i < kept.size(); ++i) {
                const auto& accepted = kept[i];
                if ((cfg.class_agnostic_nms || candidate.pose.box.class_id == accepted.pose.box.class_id) &&
                    iou(candidate.pose.box, accepted.pose.box, candidate_area, kept_areas[i]) > cfg.nms_threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.push_back(std::move(candidate));
                kept_areas.push_back(candidate_area);
            }
        }
        candidates = std::move(kept);
    }
    std::vector<Pose> result;
    result.reserve(candidates.size());
    for (auto& candidate : candidates) result.push_back(std::move(candidate.pose));
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
    if (config_.output_name.empty()) {
        const auto outputs = session_.output_names();
        if (!outputs.empty()) config_.output_name = outputs.front();
    }
}
std::vector<Detection> Detector::infer_image(const std::uint8_t* data, int width, int height, int channels) {
    const bool measure_timing = config_.enable_timing;
    const auto total_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (config_.preprocess_backend == ComputeBackend::Cuda) {
        require_cuda_backend();
        if (!cuda_preprocess_) cuda_preprocess_ = cuda_backend::create_preprocess_context();
        Letterbox transform;
        const auto preprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const float* device_data = cuda_backend::preprocess_bgr(*cuda_preprocess_, data, width, height, channels,
                                                                config_, session_.cuda_stream(), &transform);
        const auto preprocess_ms = measure_timing ? elapsed_ms(preprocess_start, std::chrono::steady_clock::now()) : 0.0;
        auto result = infer_device(device_data, cuda_backend::preprocess_bytes(config_), width, height);
        if (measure_timing) {
            timing_.preprocess_ms = preprocess_ms;
            timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        }
        return result;
    }
    auto chw = preprocess_bgr(data, width, height, channels, config_);
    const auto preprocess_ms = measure_timing ? elapsed_ms(total_start, std::chrono::steady_clock::now()) : 0.0;
    auto result = infer(chw, width, height);
    if (measure_timing) {
        timing_.preprocess_ms = preprocess_ms;
        timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    }
    return result;
}

std::vector<Segmentation> decode_segmentations(const std::vector<core::Tensor>& tensors,
                                               const Config& cfg, int image_width, int image_height) {
    std::vector<const core::Tensor*> views;
    views.reserve(tensors.size());
    for (const auto& tensor : tensors) views.push_back(&tensor);
    return decode_segmentations_impl(views, cfg, image_width, image_height);
}

std::vector<Segmentation> decode_segmentations(const std::vector<const core::Tensor*>& tensors,
                                               const Config& cfg, int image_width, int image_height) {
    return decode_segmentations_impl(tensors, cfg, image_width, image_height);
}
std::vector<Detection> Detector::infer(const std::vector<float>& chw, int image_width, int image_height) {
    const bool measure_timing = config_.enable_timing;
    if (measure_timing) timing_.preprocess_ms = 0.0;
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const std::string& input_name = config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input(input_name, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    session_.infer();
    if (measure_timing) timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<Detection> result;
    const std::string& output_name = config_.output_name;
    if (output_name.empty()) throw std::runtime_error("TensorRT engine has no output tensor");
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        const auto output_shape = session_.device_output_shape(output_name);
        if (output_shape.size() >= 2) {
            const auto dim_a = output_shape[output_shape.size() - 2];
            const auto dim_b = output_shape.back();
            if ((std::min)(dim_a, dim_b) == 6 && (std::max)(dim_a, dim_b) <= 1000)
                throw std::runtime_error("YOLO26 end2end CUDA postprocessing is not available; use --postprocess cpu or auto");
        }
        result = cuda_backend::decode_detections_device(session_.device_output(output_name),
            session_.device_output_shape(output_name), session_.device_output_type(output_name),
            config_, image_width, image_height, session_.cuda_stream());
    } else {
        result = decode_detections(session_.get_output_ref(output_name), config_, image_width, image_height);
    }
    if (measure_timing) {
        timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
        timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    }
    return result;
}
std::vector<Detection> Detector::infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height) {
    const bool measure_timing = config_.enable_timing;
    if (measure_timing) timing_.preprocess_ms = 0.0;
    if (!device_data || bytes == 0) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const std::string& input_name = config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input_device(input_name, device_data, bytes);
    const auto inference_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    session_.infer();
    if (measure_timing) timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const std::string& output_name = config_.output_name;
    if (output_name.empty()) throw std::runtime_error("TensorRT engine has no output tensor");
    std::vector<Detection> result;
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        const auto output_shape = session_.device_output_shape(output_name);
        if (output_shape.size() >= 2) {
            const auto dim_a = output_shape[output_shape.size() - 2];
            const auto dim_b = output_shape.back();
            if ((std::min)(dim_a, dim_b) == 6 && (std::max)(dim_a, dim_b) <= 1000)
                throw std::runtime_error("YOLO26 end2end CUDA postprocessing is not available; use --postprocess cpu or auto");
        }
        result = cuda_backend::decode_detections_device(session_.device_output(output_name),
            session_.device_output_shape(output_name), session_.device_output_type(output_name),
            config_, image_width, image_height, session_.cuda_stream());
    } else {
        result = decode_detections(session_.get_output_ref(output_name), config_, image_width, image_height);
    }
    if (measure_timing) {
        timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
        timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    }
    return result;
}
Segmentor::Segmentor(const std::string& path, Config config) : session_(core::TrtSession::load(path)), config_(config) {
    output_names_ = session_.output_names();
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
    const bool measure_timing = config_.enable_timing;
    const auto total_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (config_.preprocess_backend == ComputeBackend::Cuda) {
        require_cuda_backend();
        if (!cuda_preprocess_) cuda_preprocess_ = cuda_backend::create_preprocess_context();
        Letterbox transform;
        const auto preprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const float* device_data = cuda_backend::preprocess_bgr(*cuda_preprocess_, data, width, height, channels,
                                                                config_, session_.cuda_stream(), &transform);
        const auto preprocess_ms = measure_timing ? elapsed_ms(preprocess_start, std::chrono::steady_clock::now()) : 0.0;
        auto result = infer_device(device_data, cuda_backend::preprocess_bytes(config_), width, height);
        if (measure_timing) {
            timing_.preprocess_ms = preprocess_ms;
            timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        }
        return result;
    }
    auto chw = preprocess_bgr(data, width, height, channels, config_);
    const auto preprocess_ms = measure_timing ? elapsed_ms(total_start, std::chrono::steady_clock::now()) : 0.0;
    auto result = infer(chw, width, height);
    if (measure_timing) {
        timing_.preprocess_ms = preprocess_ms;
        timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    }
    return result;
}
std::vector<Segmentation> Segmentor::infer(const std::vector<float>& chw, int image_width, int image_height) {
    const bool measure_timing = config_.enable_timing;
    if (measure_timing) timing_.preprocess_ms = 0.0;
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const std::string& input_name = config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input(input_name, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    session_.infer();
    if (measure_timing) timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<Segmentation> result;
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        if (!cuda_postprocess_) cuda_postprocess_ = cuda_backend::create_segmentation_postprocess_context();
        std::string det_name = config_.output_name;
        std::string proto_name = config_.prototype_name;
        for (const auto& name : output_names_) {
            const auto shape = session_.device_output_shape(name);
            if (shape.size() == 4 && proto_name.empty()) proto_name = name;
            else if (shape.size() >= 2 && det_name.empty()) det_name = name;
        }
        if (det_name.empty() || proto_name.empty()) throw std::runtime_error("TensorRT segmentation outputs are incomplete");
        const auto det_shape = session_.device_output_shape(det_name);
        const auto proto_shape = session_.device_output_shape(proto_name);
        if (det_shape.size() >= 2 && proto_shape.size() == 4) {
            const auto attrs = (std::min)(det_shape[det_shape.size() - 2], det_shape.back());
            const auto boxes = (std::max)(det_shape[det_shape.size() - 2], det_shape.back());
            if (boxes <= 1000 && attrs == 6 + proto_shape[1])
                throw std::runtime_error("YOLO26 end2end CUDA postprocessing is not available; use --postprocess cpu or auto");
        }
        result = cuda_backend::decode_segmentations_device(session_.device_output(det_name),
            session_.device_output_shape(det_name), session_.device_output_type(det_name),
            session_.device_output(proto_name), session_.device_output_shape(proto_name),
            session_.device_output_type(proto_name), config_, image_width, image_height,
            session_.cuda_stream(), *cuda_postprocess_);
    } else {
        std::vector<const core::Tensor*> tensors;
        if (!config_.output_name.empty()) tensors.push_back(&session_.get_output_ref(config_.output_name));
        if (!config_.prototype_name.empty()) tensors.push_back(&session_.get_output_ref(config_.prototype_name));
        if (tensors.empty()) tensors = session_.output_refs();
        result = decode_segmentations(tensors, config_, image_width, image_height);
    }
    if (measure_timing) {
        timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
        timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    }
    return result;
}
std::vector<Segmentation> Segmentor::infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height) {
    const bool measure_timing = config_.enable_timing;
    if (measure_timing) timing_.preprocess_ms = 0.0;
    if (!device_data || bytes == 0) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda) require_cuda_backend();
    const std::string& input_name = config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(config_.postprocess_backend != ComputeBackend::Cuda);
    session_.set_input_device(input_name, device_data, bytes);
    const auto inference_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    session_.infer();
    if (measure_timing) timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::vector<Segmentation> result;
    if (config_.postprocess_backend == ComputeBackend::Cuda) {
        std::string det_name = config_.output_name;
        std::string proto_name = config_.prototype_name;
        for (const auto& name : output_names_) {
            const auto shape = session_.device_output_shape(name);
            if (shape.size() == 4 && proto_name.empty()) proto_name = name;
            else if (shape.size() >= 2 && det_name.empty()) det_name = name;
        }
        if (det_name.empty() || proto_name.empty()) throw std::runtime_error("TensorRT segmentation outputs are incomplete");
        const auto det_shape = session_.device_output_shape(det_name);
        const auto proto_shape = session_.device_output_shape(proto_name);
        if (det_shape.size() >= 2 && proto_shape.size() == 4) {
            const auto attrs = (std::min)(det_shape[det_shape.size() - 2], det_shape.back());
            const auto boxes = (std::max)(det_shape[det_shape.size() - 2], det_shape.back());
            if (boxes <= 1000 && attrs == 6 + proto_shape[1])
                throw std::runtime_error("YOLO26 end2end CUDA postprocessing is not available; use --postprocess cpu or auto");
        }
        if (!cuda_postprocess_) cuda_postprocess_ = cuda_backend::create_segmentation_postprocess_context();
        result = cuda_backend::decode_segmentations_device(session_.device_output(det_name),
            session_.device_output_shape(det_name), session_.device_output_type(det_name),
            session_.device_output(proto_name), session_.device_output_shape(proto_name),
            session_.device_output_type(proto_name), config_, image_width, image_height,
            session_.cuda_stream(), *cuda_postprocess_);
    } else {
        std::vector<const core::Tensor*> tensors;
        if (!config_.output_name.empty()) tensors.push_back(&session_.get_output_ref(config_.output_name));
        if (!config_.prototype_name.empty()) tensors.push_back(&session_.get_output_ref(config_.prototype_name));
        if (tensors.empty()) tensors = session_.output_refs();
        result = decode_segmentations(tensors, config_, image_width, image_height);
    }
    if (measure_timing) {
        timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
        timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    }
    return result;
}
PoseEstimator::PoseEstimator(const std::string& path, Config config) : session_(core::TrtSession::load(path)), config_(config) {
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
    if (config_.output_name.empty()) {
        const auto outputs = session_.output_names();
        if (!outputs.empty()) config_.output_name = outputs.front();
    }
}
std::vector<Pose> PoseEstimator::infer_image(const std::uint8_t* data, int width, int height, int channels) {
    const bool measure_timing = config_.enable_timing;
    const auto total_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (config_.preprocess_backend == ComputeBackend::Cuda) {
        require_cuda_backend();
        if (!cuda_preprocess_) cuda_preprocess_ = cuda_backend::create_preprocess_context();
        Letterbox transform;
        const auto preprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const float* device_data = cuda_backend::preprocess_bgr(*cuda_preprocess_, data, width, height, channels,
                                                                config_, session_.cuda_stream(), &transform);
        const auto preprocess_ms = measure_timing ? elapsed_ms(preprocess_start, std::chrono::steady_clock::now()) : 0.0;
        auto result = infer_device(device_data, cuda_backend::preprocess_bytes(config_), width, height);
        if (measure_timing) {
            timing_.preprocess_ms = preprocess_ms;
            timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
        }
        return result;
    }
    auto chw = preprocess_bgr(data, width, height, channels, config_);
    const auto preprocess_ms = measure_timing ? elapsed_ms(total_start, std::chrono::steady_clock::now()) : 0.0;
    auto result = infer(chw, width, height);
    if (measure_timing) {
        timing_.preprocess_ms = preprocess_ms;
        timing_.total_ms = elapsed_ms(total_start, std::chrono::steady_clock::now());
    }
    return result;
}
std::vector<Pose> PoseEstimator::infer(const std::vector<float>& chw, int image_width, int image_height) {
    const bool measure_timing = config_.enable_timing;
    if (measure_timing) timing_.preprocess_ms = 0.0;
    if (chw.empty()) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda)
        throw std::runtime_error("YOLO11/YOLO26 pose CUDA postprocessing is not available; use --postprocess cpu or auto");
    const std::string& input_name = config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(true);
    session_.set_input(input_name, chw.data(), chw.size() * sizeof(float));
    const auto inference_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    session_.infer();
    if (measure_timing) timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const std::string& output_name = config_.output_name;
    if (output_name.empty()) throw std::runtime_error("TensorRT engine has no output tensor");
    auto result = decode_poses(session_.get_output_ref(output_name), config_, image_width, image_height);
    if (measure_timing) {
        timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
        timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    }
    return result;
}
std::vector<Pose> PoseEstimator::infer_device(const void* device_data, std::size_t bytes, int image_width, int image_height) {
    const bool measure_timing = config_.enable_timing;
    if (measure_timing) timing_.preprocess_ms = 0.0;
    if (!device_data || bytes == 0) return {};
    if (config_.postprocess_backend == ComputeBackend::Cuda)
        throw std::runtime_error("YOLO11/YOLO26 pose CUDA postprocessing is not available; use --postprocess cpu or auto");
    const std::string& input_name = config_.input_name;
    if (input_name.empty()) throw std::runtime_error("TensorRT engine has no input tensor");
    session_.set_shape(input_name, {1, 3, config_.input_height, config_.input_width});
    session_.set_output_host_copy(true);
    session_.set_input_device(input_name, device_data, bytes);
    const auto inference_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    session_.infer();
    if (measure_timing) timing_.inference_ms = elapsed_ms(inference_start, std::chrono::steady_clock::now());
    const auto postprocess_start = measure_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const std::string& output_name = config_.output_name;
    if (output_name.empty()) throw std::runtime_error("TensorRT engine has no output tensor");
    auto result = decode_poses(session_.get_output_ref(output_name), config_, image_width, image_height);
    if (measure_timing) {
        timing_.postprocess_ms = elapsed_ms(postprocess_start, std::chrono::steady_clock::now());
        timing_.total_ms = timing_.inference_ms + timing_.postprocess_ms;
    }
    return result;
}
} // namespace cvinfer::yolo
