#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::core {

class TrtSession {
public:
    TrtSession();
    ~TrtSession();
    TrtSession(TrtSession&&) noexcept;
    TrtSession& operator=(TrtSession&&) noexcept;
    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;

    static TrtSession load(const std::string& engine_path);
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;
    std::vector<int64_t> tensor_shape(const std::string& name) const;
    void set_shape(const std::string& name, const std::vector<int64_t>& shape);
    void set_input(const std::string& name, const void* data, std::size_t bytes);
    void set_input_device(const std::string& name, const void* device_data, std::size_t bytes);
    void set_output_host_copy(bool enabled);
    void infer();
    // TensorRT keeps the historical synchronous implementation; these
    // methods provide the same public Session contract as Denglin.
    void infer_async();
    void synchronize();
    bool is_pending() const noexcept;
    Tensor get_output(const std::string& name) const;
    // Returns a view valid until the next infer() call on this session.
    const Tensor& get_output_ref(const std::string& name) const;
    std::vector<Tensor> outputs() const;
    // Returns views valid until the next infer() call on this session.
    std::vector<const Tensor*> output_refs() const;
    const void* device_output(const std::string& name) const;
    std::vector<int64_t> device_output_shape(const std::string& name) const;
    TensorDataType device_output_type(const std::string& name) const;
    void* cuda_stream() const noexcept;

private:
    struct Impl;
    explicit TrtSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace cvinfer::core
