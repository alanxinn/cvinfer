#include "sam3/sam3_api.h"
#include "sam3/sam3.h"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
struct sam3_model { cvinfer::sam3::Segmentor segmentor; explicit sam3_model(const char* p, cvinfer::sam3::Config c) : segmentor(p, c) {} };
struct sam3_result { std::vector<cvinfer::sam3::Mask> masks; };
namespace {
thread_local std::string error;
template<class F> int guard(F f) noexcept {
    try { error.clear(); f(); return 0; }
    catch (const std::invalid_argument& e) { error = e.what(); return 1; }
    catch (const std::exception& e) { error = e.what(); return 2; }
    catch (...) { error = "Unknown SAM3 error"; return 2; }
}
void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }
}
extern "C" {
SAM3_API const char* sam3_last_error(void) { return error.c_str(); }
SAM3_API int sam3_model_create(const char* directory, float confidence, float threshold, int32_t max_points, sam3_model_t** model) {
    if (model) *model = nullptr;
    return guard([&] { require(directory && *directory && model, "SAM3 directory and model output are required");
        *model = new sam3_model(directory, {confidence, threshold, max_points}); });
}
SAM3_API void sam3_model_destroy(sam3_model_t* model) { delete model; }
SAM3_API int sam3_model_set_image(sam3_model_t* model, const uint8_t* bgr, int32_t w, int32_t h, int32_t c) {
    return guard([&] { require(model != nullptr, "SAM3 model is null"); model->segmentor.set_image(bgr, w, h, c); });
}
SAM3_API int sam3_model_set_box(sam3_model_t* model, float x1, float y1, float x2, float y2) {
    return guard([&] { require(model != nullptr, "SAM3 model is null"); model->segmentor.set_box({x1, y1, x2, y2}); });
}
SAM3_API int sam3_model_add_point(sam3_model_t* model, float x, float y, int32_t positive) {
    return guard([&] { require(model && (positive == 0 || positive == 1), "SAM3 point label must be 0 or 1"); model->segmentor.add_point({x, y, positive != 0}); });
}
SAM3_API int sam3_model_clear_points(sam3_model_t* model) {
    return guard([&] { require(model != nullptr, "SAM3 model is null"); model->segmentor.clear_points(); });
}
SAM3_API int sam3_model_predict(sam3_model_t* model, sam3_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && result, "SAM3 model and result output are required");
        auto value = std::make_unique<sam3_result>(); value->masks.push_back(model->segmentor.predict()); *result = value.release(); });
}
SAM3_API int sam3_model_predict_text(sam3_model_t* model, const char* text, sam3_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && text && result, "SAM3 model, text and result output are required");
        auto value = std::make_unique<sam3_result>(); value->masks = model->segmentor.predict_text(text); *result = value.release(); });
}
SAM3_API int sam3_model_predict_tokens(sam3_model_t* model, const int64_t* ids, const int64_t* attention, size_t n, sam3_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && ids && attention && n > 0 && n <= 4096 && result, "Invalid SAM3 encoded token arguments");
        auto value = std::make_unique<sam3_result>(); value->masks = model->segmentor.predict_tokens({ids, ids + n}, {attention, attention + n}); *result = value.release(); });
}
SAM3_API void sam3_result_destroy(sam3_result_t* result) { delete result; }
SAM3_API size_t sam3_result_count(const sam3_result_t* result) { return result ? result->masks.size() : 0; }
SAM3_API int sam3_result_get_mask(const sam3_result_t* result, size_t i, sam3_mask_info_t* info) {
    return guard([&] { require(result && info && i < result->masks.size(), "Invalid SAM3 mask index or output");
        const auto& m = result->masks[i]; *info = {m.box.x1, m.box.y1, m.box.x2, m.box.y2, m.score, m.width, m.height, m.pixels.size()}; });
}
SAM3_API int sam3_result_copy_mask(const sam3_result_t* result, size_t i, uint8_t* data, size_t capacity, size_t* copied) {
    if (copied) *copied = 0;
    return guard([&] { require(result && copied && i < result->masks.size(), "Invalid SAM3 mask index or copied output");
        const auto& pixels = result->masks[i].pixels; *copied = pixels.size();
        require(data && capacity >= pixels.size(), "SAM3 mask output buffer is too small"); std::memcpy(data, pixels.data(), pixels.size()); });
}
}
