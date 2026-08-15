#include "core/inference_session.h"

#if !defined(CVINFER_HAS_TENSORRT_BACKEND)
#define CVINFER_HAS_TENSORRT_BACKEND 1
#endif
#if !defined(CVINFER_HAS_DENGLIN_BACKEND)
#define CVINFER_HAS_DENGLIN_BACKEND 0
#endif

#if CVINFER_HAS_TENSORRT_BACKEND
#include "core/trt_core.h"
#endif

#if CVINFER_HAS_DENGLIN_BACKEND
#include "core/denglin_core.h"
#endif

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

namespace cvinfer::core {
namespace {

template <typename Session>
class SessionAdapter final : public InferenceSession {
public:
    explicit SessionAdapter(Session session) : session_(std::move(session)) {}

    std::vector<std::string> input_names() const override {
        return session_.input_names();
    }

    std::vector<std::string> output_names() const override {
        return session_.output_names();
    }

    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return session_.tensor_shape(name);
    }

    void set_shape(const std::string& name,
                   const std::vector<int64_t>& shape) override {
        session_.set_shape(name, shape);
    }

    void set_input(const std::string& name,
                   const void* data,
                   std::size_t bytes) override {
        session_.set_input(name, data, bytes);
    }

    void set_input_device(const std::string& name,
                          const void* device_data,
                          std::size_t bytes) override {
        session_.set_input_device(name, device_data, bytes);
    }

    void set_output_host_copy(bool enabled) override {
        session_.set_output_host_copy(enabled);
    }

    void infer() override { session_.infer(); }

    void infer_async() override { session_.infer_async(); }
    void synchronize() override { session_.synchronize(); }
    bool is_pending() const noexcept override { return session_.is_pending(); }

    Tensor get_output(const std::string& name) const override {
        return session_.get_output(name);
    }

    const Tensor& get_output_ref(const std::string& name) const override {
        return session_.get_output_ref(name);
    }

    std::vector<Tensor> outputs() const override { return session_.outputs(); }

    std::vector<const Tensor*> output_refs() const override {
        return session_.output_refs();
    }

    const void* device_output(const std::string& name) const override {
        return session_.device_output(name);
    }

    std::vector<int64_t> device_output_shape(
        const std::string& name) const override {
        return session_.device_output_shape(name);
    }

    TensorDataType device_output_type(const std::string& name) const override {
        return session_.device_output_type(name);
    }

    void* cuda_stream() const noexcept override { return session_.cuda_stream(); }

private:
    Session session_;
};

bool has_suffix(const std::string& path, const char* suffix) {
    if (!suffix) return false;
    if (path.size() < std::char_traits<char>::length(suffix)) return false;
    const auto offset = path.size() - std::char_traits<char>::length(suffix);
    for (std::size_t i = 0; suffix[i] != '\0'; ++i) {
        const auto left = static_cast<unsigned char>(path[offset + i]);
        const auto right = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(left) != std::tolower(right)) return false;
    }
    return true;
}

} // namespace

std::unique_ptr<InferenceSession> create_inference_session(
    const std::string& model_path,
    const InferenceSessionOptions& options) {
    if (model_path.empty()) {
        throw std::invalid_argument("Inference model path is empty");
    }

    if (options.backend == InferenceBackend::TensorRT) {
#if CVINFER_HAS_TENSORRT_BACKEND
        return std::make_unique<SessionAdapter<TrtSession>>(
            TrtSession::load(model_path));
#else
        throw std::runtime_error(
            "TensorRT inference backend is not available in this native library");
#endif
    }

    if (options.backend != InferenceBackend::Denglin) {
        throw std::invalid_argument("Unsupported inference backend");
    }

#if CVINFER_HAS_DENGLIN_BACKEND
    DenglinSessionOptions denglin_options;
    // Serialized Denglin engines/caches carry their own metadata. ONNX
    // loading needs a parser input declaration, so provide the common YOLO /
    // RF-DETR defaults when the model adapter did not receive explicit names.
    if (has_suffix(model_path, ".onnx")) {
        denglin_options.input_name =
            options.input_name.empty() ? "images" : options.input_name;
        denglin_options.input_shape = options.input_shape.empty()
            ? std::vector<int64_t>{1, 3, 640, 640}
            : options.input_shape;
        denglin_options.output_names = options.output_names;
    }
    return std::make_unique<SessionAdapter<DenglinSession>>(
        DenglinSession::load(model_path, std::move(denglin_options)));
#else
    (void)options;
    throw std::runtime_error(
        "Denglin inference backend is not enabled in this build");
#endif
}

} // namespace cvinfer::core
