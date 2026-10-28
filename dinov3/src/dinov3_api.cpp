#include "dinov3/dinov3_api.h"
#include "dinov3/dinov3.h"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

struct dinov3_model { cvinfer::dinov3::FeatureExtractor extractor; explicit dinov3_model(const char* p) : extractor(p) {} };
struct dinov3_result { cvinfer::dinov3::Result features; };
namespace {
thread_local std::string error;
template<class F> int guard(F f) noexcept {
    try { error.clear(); f(); return 0; }
    catch (const std::invalid_argument& e) { error = e.what(); return 1; }
    catch (const std::exception& e) { error = e.what(); return 2; }
    catch (...) { error = "Unknown DINOv3 error"; return 2; }
}
void require(bool condition, const char* message) { if (!condition) throw std::invalid_argument(message); }
const cvinfer::core::Tensor& get_tensor(const dinov3_result* result, const char* name) {
    require(result && name, "DINOv3 result and tensor name are required");
    for (const auto* t : {&result->features.global_embedding, &result->features.normalized_embedding,
                           &result->features.patch_tokens, &result->features.feature_map})
        if (t->name == name) return *t;
    throw std::invalid_argument("Unknown DINOv3 tensor name");
}
}
extern "C" {
DINOV3_API const char* dinov3_last_error(void) { return error.c_str(); }
DINOV3_API int dinov3_model_create(const char* path, dinov3_model_t** model) {
    if (model) *model = nullptr;
    return guard([&] { require(path && *path && model, "DINOv3 engine path and model output are required"); *model = new dinov3_model(path); });
}
DINOV3_API void dinov3_model_destroy(dinov3_model_t* model) { delete model; }
DINOV3_API int dinov3_model_input_size(const dinov3_model_t* model, int32_t* w, int32_t* h) {
    return guard([&] { require(model && w && h, "Invalid DINOv3 model or input-size output"); *w = model->extractor.input_width(); *h = model->extractor.input_height(); });
}
DINOV3_API int dinov3_model_infer_bgr(dinov3_model_t* model, const uint8_t* data, int32_t w, int32_t h, int32_t c, dinov3_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && result, "DINOv3 model and result output are required");
        auto value = std::make_unique<dinov3_result>(); value->features = model->extractor.infer_bgr(data, w, h, c); *result = value.release(); });
}
DINOV3_API int dinov3_model_infer_chw(dinov3_model_t* model, const float* data, size_t n, dinov3_result_t** result) {
    if (result) *result = nullptr;
    return guard([&] { require(model && result, "DINOv3 model and result output are required");
        auto value = std::make_unique<dinov3_result>(); value->features = model->extractor.infer_chw(data, n); *result = value.release(); });
}
DINOV3_API void dinov3_result_destroy(dinov3_result_t* result) { delete result; }
DINOV3_API int dinov3_result_tensor_info(const dinov3_result_t* result, const char* name, dinov3_tensor_info_t* info) {
    return guard([&] { require(info != nullptr, "DINOv3 tensor info output is null"); const auto& t = get_tensor(result, name);
        *info = {}; info->rank = static_cast<int32_t>(t.shape.size());
        for (std::size_t i = 0; i < t.shape.size(); ++i) info->shape[i] = t.shape[i];
        info->elements = t.data.size() / sizeof(float); });
}
DINOV3_API int dinov3_result_copy_tensor(const dinov3_result_t* result, const char* name, float* destination, size_t capacity, size_t* copied) {
    if (copied) *copied = 0;
    const auto status = guard([&] { require(copied != nullptr, "DINOv3 copied element count is null");
        *copied = get_tensor(result, name).data.size() / sizeof(float); });
    if (status) return status;
    if (capacity < *copied) { error = "DINOv3 tensor buffer is too small"; return 4; }
    return guard([&] { require(destination != nullptr, "DINOv3 destination is null");
        const auto& t = get_tensor(result, name); std::memcpy(destination, t.data.data(), t.data.size()); });
}
}
