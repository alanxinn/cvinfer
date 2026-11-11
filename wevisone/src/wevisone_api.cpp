#include "wevisone/wevisone_api.h"
#include "wevisone/wevisone.h"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace w = cvinfer::wevisone;
struct wevisone_model { w::Model model; explicit wevisone_model(const char* p) : model(p) {} };
struct wevisone_result { w::Result value; };
namespace {
thread_local std::string error;
struct SmallBuffer {};
void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }
template<class F> int guard(F f) noexcept {
    try { error.clear(); f(); return 0; }
    catch (const SmallBuffer&) { error = "WeVisOne output buffer is too small"; return 4; }
    catch (const std::invalid_argument& e) { error = e.what(); return 1; }
    catch (const std::exception& e) { error = e.what(); return 2; }
    catch (...) { error = "Unknown WeVisOne error"; return 2; }
}
w::Options options(const wevisone_options_t* p) {
    if (!p) return {};
    require(p->struct_size == sizeof(*p), "WeVisOne options ABI size mismatch");
    require(p->decode_pixels == 0 || p->decode_pixels == 1, "Invalid WeVisOne decode_pixels flag");
    w::Options o; o.max_new_tokens = p->max_new_tokens; o.temperature = p->temperature; o.top_p = p->top_p;
    o.top_k = p->top_k; o.repetition_penalty = p->repetition_penalty; o.mask_threshold = p->mask_threshold;
    o.seed = p->seed; o.decode_pixels = p->decode_pixels != 0; return o;
}
std::vector<w::Box> boxes(const wevisone_box_t* values, size_t n) {
    require((values || n == 0) && n <= 1024, "Invalid WeVisOne box array/count");
    std::vector<w::Box> result; result.reserve(n);
    for (size_t i = 0; i < n; ++i) result.push_back({values[i].x1, values[i].y1, values[i].x2, values[i].y2});
    return result;
}
wevisone_box_t box(const w::Box& b) { return {b.x1, b.y1, b.x2, b.y2}; }
const w::Mask& mask(const wevisone_result_t* r, size_t i) {
    require(r && i < r->value.masks.size(), "Invalid WeVisOne result or mask index"); return r->value.masks[i];
}
template<class T> void copy(const T* source, size_t count, T* dest, size_t capacity, size_t* required) {
    require(required && (dest || capacity == 0), "Invalid WeVisOne output buffer");
    *required = count;
    if (!dest && capacity == 0) return;
    if (capacity < count) throw SmallBuffer{};
    if (count) std::memcpy(dest, source, count * sizeof(T));
}
}
extern "C" {
WEVISONE_API const char* wevisone_last_error(void) { return error.c_str(); }
WEVISONE_API void wevisone_options_init(wevisone_options_t* p) {
    if (!p) return;
    *p = {}; p->struct_size = sizeof(*p); p->max_new_tokens = 1024; p->top_p = .9F;
    p->top_k = 50; p->repetition_penalty = 1; p->mask_threshold = .5F; p->decode_pixels = 1;
}
WEVISONE_API int wevisone_model_create(const char* directory, wevisone_model_t** model) {
    if (model) *model = nullptr;
    return guard([&] { require(directory && *directory && model, "WeVisOne bundle directory and model output are required"); *model = new wevisone_model(directory); });
}
WEVISONE_API void wevisone_model_destroy(wevisone_model_t* model) { delete model; }
WEVISONE_API int wevisone_model_set_image(wevisone_model_t* model, const uint8_t* data, int32_t width, int32_t height, int32_t channels) {
    return guard([&] { require(model, "WeVisOne model is null"); model->model.set_image(data, width, height, channels); });
}
WEVISONE_API int wevisone_model_predict(wevisone_model_t* model, const char* task, const char* query,
    const char* prompt, const wevisone_box_t* refs, size_t n, const wevisone_options_t* config, wevisone_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] {
        require(model && task && result, "WeVisOne model, task and result output are required");
        w::Request r; r.task = w::parse_task(task); r.options = options(config); r.reference_boxes = boxes(refs, n);
        if (query) r.query = {query}; if (prompt) r.prompt = prompt;
        auto value = std::make_unique<wevisone_result>(); value->value = model->model.predict(r); *result = value.release();
    });
}
WEVISONE_API int wevisone_model_matte_boxes(wevisone_model_t* model, const wevisone_box_t* values, size_t n,
    const wevisone_options_t* config, wevisone_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && result, "WeVisOne model and result output are required");
        auto value = std::make_unique<wevisone_result>(); value->value = model->model.matte_boxes(boxes(values, n), options(config)); *result = value.release(); });
}
WEVISONE_API void wevisone_result_destroy(wevisone_result_t* result) { delete result; }
WEVISONE_API int wevisone_result_summary(const wevisone_result_t* result, wevisone_summary_t* s) {
    return guard([&] { require(result && s, "WeVisOne result and summary output are required"); const auto& r = result->value;
        *s = {r.count, int32_t(r.rejected), int32_t(r.truncated), r.latency_ms, r.annotations.size(), r.masks.size(), r.absent_labels.size()}; });
}
WEVISONE_API int wevisone_result_copy_text(const wevisone_result_t* result, int32_t field, size_t index,
    char* destination, size_t capacity, size_t* required) {
    if (required) *required = 0;
    return guard([&] {
        require(result, "WeVisOne result is null"); const auto& r = result->value; const std::string* text{};
        switch (field) {
        case 0: text = &r.response; break;
        case 1: text = &r.prompt; break;
        case 2: require(index < r.annotations.size(), "Invalid WeVisOne annotation index"); text = &r.annotations[index].label; break;
        case 3: require(index < r.absent_labels.size(), "Invalid WeVisOne absent-label index"); text = &r.absent_labels[index]; break;
        default: throw std::invalid_argument("Invalid WeVisOne text field");
        }
        copy(text->c_str(), text->size() + 1, destination, capacity, required);
    });
}
WEVISONE_API int wevisone_result_annotation(const wevisone_result_t* r, size_t i, wevisone_annotation_t* a) {
    return guard([&] { require(r && a && i < r->value.annotations.size(), "Invalid WeVisOne annotation output/index");
        const auto& v = r->value.annotations[i]; *a = {box(v.box), v.x, v.y, int32_t(v.is_point), v.mask_index}; });
}
WEVISONE_API int wevisone_result_mask_info(const wevisone_result_t* r, size_t i, wevisone_mask_info_t* out) {
    return guard([&] { require(out, "WeVisOne mask info output is null"); const auto& m = mask(r, i);
        *out = {box(m.box), m.width, m.height, m.alpha.size(), m.binary.size()}; });
}
WEVISONE_API int wevisone_result_copy_alpha(const wevisone_result_t* r, size_t i, float* dest, size_t capacity, size_t* required) {
    if (required) *required = 0;
    return guard([&] { const auto& m = mask(r, i); copy(m.alpha.data(), m.alpha.size(), dest, capacity, required); });
}
WEVISONE_API int wevisone_result_copy_binary(const wevisone_result_t* r, size_t i, uint8_t* dest, size_t capacity, size_t* required) {
    if (required) *required = 0;
    return guard([&] { const auto& m = mask(r, i); copy(m.binary.data(), m.binary.size(), dest, capacity, required); });
}
}
