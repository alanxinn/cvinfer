#include "sam2/sam2_api.h"
#include "sam2/sam2.h"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
struct sam2_model { cvinfer::sam2::Segmentor segmentor; explicit sam2_model(const char* p, cvinfer::sam2::Config c) : segmentor(p, c) {} };
struct sam2_result { std::vector<cvinfer::sam2::Mask> masks; };
namespace {
thread_local std::string error;
template<class F> int guard(F f) noexcept {
    try { error.clear(); f(); return 0; }
    catch (const std::invalid_argument& e) { error = e.what(); return 1; }
    catch (const std::exception& e) { error = e.what(); return 2; }
    catch (...) { error = "Unknown SAM2 error"; return 2; }
}
void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }
}
extern "C" {
SAM2_API const char* sam2_last_error(void) { return error.c_str(); }
SAM2_API int sam2_model_create(const char* directory, float threshold, int32_t max_points, sam2_model_t** model) {
    if (model) *model = nullptr;
    return guard([&] { require(directory && *directory && model, "SAM2 directory and model output are required");
        *model = new sam2_model(directory, {threshold, max_points}); });
}
SAM2_API void sam2_model_destroy(sam2_model_t* model) { delete model; }
SAM2_API int sam2_model_set_image(sam2_model_t* model, const uint8_t* bgr, int32_t w, int32_t h, int32_t c) {
    return guard([&] { require(model != nullptr, "SAM2 model is null"); model->segmentor.set_image(bgr, w, h, c); });
}
SAM2_API int sam2_model_set_box(sam2_model_t* model, float x1, float y1, float x2, float y2) {
    return guard([&] { require(model != nullptr, "SAM2 model is null"); model->segmentor.set_box({x1, y1, x2, y2}); });
}
SAM2_API int sam2_model_add_point(sam2_model_t* model, float x, float y, int32_t positive) {
    return guard([&] { require(model && (positive == 0 || positive == 1), "SAM2 point label must be 0 or 1"); model->segmentor.add_point({x, y, positive != 0}); });
}
SAM2_API int sam2_model_clear_box(sam2_model_t* model) {
    return guard([&] { require(model != nullptr, "SAM2 model is null"); model->segmentor.clear_box(); });
}
SAM2_API int sam2_model_clear_prompts(sam2_model_t* model) {
    return guard([&] { require(model != nullptr, "SAM2 model is null"); model->segmentor.clear_prompts(); });
}
SAM2_API int sam2_model_remove_point(sam2_model_t* model, size_t index) {
    return guard([&] { require(model != nullptr, "SAM2 model is null"); model->segmentor.remove_point(index); });
}
SAM2_API int sam2_model_clear_points(sam2_model_t* model) {
    return guard([&] { require(model != nullptr, "SAM2 model is null"); model->segmentor.clear_points(); });
}
SAM2_API int sam2_model_predict(sam2_model_t* model, int32_t use_previous_mask, sam2_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && result && (use_previous_mask == 0 || use_previous_mask == 1), "Invalid SAM2 model, result or refinement flag");
        auto value = std::make_unique<sam2_result>(); value->masks.push_back(model->segmentor.predict(use_previous_mask != 0)); *result = value.release(); });
}
SAM2_API void sam2_result_destroy(sam2_result_t* result) { delete result; }
SAM2_API size_t sam2_result_count(const sam2_result_t* result) { return result ? result->masks.size() : 0; }
SAM2_API int sam2_result_get_mask(const sam2_result_t* result, size_t i, sam2_mask_info_t* info) {
    return guard([&] { require(result && info && i < result->masks.size(), "Invalid SAM2 mask index or output");
        const auto& m = result->masks[i]; *info = {m.box.x1, m.box.y1, m.box.x2, m.box.y2, m.score, m.width, m.height, m.pixels.size()}; });
}
SAM2_API int sam2_result_copy_mask(const sam2_result_t* result, size_t i, uint8_t* data, size_t capacity, size_t* copied) {
    if (copied) *copied = 0;
    return guard([&] { require(result && copied && i < result->masks.size(), "Invalid SAM2 mask index or copied output");
        const auto& pixels = result->masks[i].pixels; *copied = pixels.size();
        require(data && capacity >= pixels.size(), "SAM2 mask output buffer is too small"); std::memcpy(data, pixels.data(), pixels.size()); });
}
}
