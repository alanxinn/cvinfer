#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::core {

// Backend selection is intentionally independent from the preprocessing and
// postprocessing backend enums used by YOLO/RF-DETR.
enum class InferenceBackend {
    TensorRT = 0,
    Denglin = 1
};

struct InferenceSessionOptions {
    InferenceBackend backend{InferenceBackend::TensorRT};
    std::string input_name;
    std::vector<int64_t> input_shape;
    std::vector<std::string> output_names;
};

// Backend-neutral inference contract used by model adapters. Backend-specific
// pointers and engine objects stay hidden behind this interface.
class InferenceSession {
public:
    virtual ~InferenceSession() = default;

    virtual std::vector<std::string> input_names() const = 0;
    virtual std::vector<std::string> output_names() const = 0;
    virtual std::vector<int64_t> tensor_shape(const std::string& name) const = 0;
    virtual void set_shape(const std::string& name,
                           const std::vector<int64_t>& shape) = 0;

    virtual void set_input(const std::string& name,
                           const void* data,
                           std::size_t bytes) = 0;
    virtual void set_input_device(const std::string& name,
                                  const void* device_data,
                                  std::size_t bytes) = 0;
    virtual void set_output_host_copy(bool enabled) = 0;

    virtual void infer() = 0;
    virtual void infer_async() = 0;
    virtual void synchronize() = 0;
    virtual bool is_pending() const noexcept = 0;

    virtual Tensor get_output(const std::string& name) const = 0;
    virtual const Tensor& get_output_ref(const std::string& name) const = 0;
    virtual std::vector<Tensor> outputs() const = 0;
    virtual std::vector<const Tensor*> output_refs() const = 0;

    virtual const void* device_output(const std::string& name) const = 0;
    virtual std::vector<int64_t> device_output_shape(
        const std::string& name) const = 0;
    virtual TensorDataType device_output_type(const std::string& name) const = 0;
    virtual void* cuda_stream() const noexcept = 0;
};

std::unique_ptr<InferenceSession> create_inference_session(
    const std::string& model_path,
    const InferenceSessionOptions& options = {});

} // namespace cvinfer::core
