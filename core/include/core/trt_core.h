#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::core {

enum class TensorDataType { Float32, Float16, Int8, Int32, Bool, Unknown };
const char* tensor_data_type_name(TensorDataType type);

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    TensorDataType data_type{TensorDataType::Unknown};
    std::vector<std::uint8_t> data;
};

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
    Tensor get_output(const std::string& name) const;
    std::vector<Tensor> outputs() const;
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
