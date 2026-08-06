#include "core/trt_core.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cvinfer::core {
cvinfer::core::TensorDataType to_data_type(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return TensorDataType::Float32;
    case nvinfer1::DataType::kHALF: return TensorDataType::Float16;
    case nvinfer1::DataType::kINT8: return TensorDataType::Int8;
    case nvinfer1::DataType::kINT32: return TensorDataType::Int32;
    case nvinfer1::DataType::kBOOL: return TensorDataType::Bool;
    default: return TensorDataType::Unknown;
    }
}

namespace {
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) last_message_ = message ? message : "";
    }
    std::string last_message_;
};
Logger g_logger;

template <typename T> void destroy(T* object) { if (object) object->destroy(); }
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream text; text << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(text.str());
    }
}
std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(std::uint16_t);
    case nvinfer1::DataType::kINT8: return sizeof(std::int8_t);
    case nvinfer1::DataType::kINT32: return sizeof(std::int32_t);
    case nvinfer1::DataType::kBOOL: return sizeof(bool);
    default: throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}
std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("Tensor shape is unresolved or invalid");
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}
std::vector<int64_t> to_shape(const nvinfer1::Dims& dims) {
    std::vector<int64_t> result;
    for (int i = 0; i < dims.nbDims; ++i) result.push_back(static_cast<int64_t>(dims.d[i]));
    return result;
}
nvinfer1::Dims to_dims(const std::vector<int64_t>& shape) {
    if (shape.empty() || shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        throw std::invalid_argument("Invalid tensor shape");
    nvinfer1::Dims dims{}; dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        if (shape[static_cast<std::size_t>(i)] <= 0 || shape[static_cast<std::size_t>(i)] > std::numeric_limits<int>::max())
            throw std::invalid_argument("Tensor dimensions must be positive");
        dims.d[i] = static_cast<int>(shape[static_cast<std::size_t>(i)]);
    }
    return dims;
}
} // namespace

struct TrtSession::Impl {
    nvinfer1::IRuntime* runtime{};
    nvinfer1::ICudaEngine* engine{};
    nvinfer1::IExecutionContext* context{};
    cudaStream_t stream{};
    std::vector<void*> buffers;
    std::vector<void*> owned_buffers;
    std::vector<std::size_t> buffer_bytes;
    std::unordered_map<std::string, int> bindings;
    std::unordered_map<std::string, std::vector<std::uint8_t>> inputs;
    std::unordered_map<std::string, const void*> device_inputs;
    std::unordered_map<std::string, std::size_t> device_input_bytes;
    std::unordered_map<std::string, Tensor> outputs;
    std::unordered_map<std::string, std::vector<int64_t>> output_shapes;
    bool output_host_copy{true};
    ~Impl() {
        for (void* buffer : owned_buffers) if (buffer) cudaFree(buffer);
        if (stream) cudaStreamDestroy(stream);
        destroy(context); destroy(engine); destroy(runtime);
    }
    int binding(const std::string& name) const {
        const auto it = bindings.find(name);
        if (it == bindings.end()) throw std::invalid_argument("Unknown TensorRT tensor: " + name);
        return it->second;
    }
};

const char* tensor_data_type_name(TensorDataType type) {
    switch (type) {
    case TensorDataType::Float32: return "FP32";
    case TensorDataType::Float16: return "FP16";
    case TensorDataType::Int8: return "INT8";
    case TensorDataType::Int32: return "INT32";
    case TensorDataType::Bool: return "BOOL";
    default: return "UNKNOWN";
    }
}

TrtSession::TrtSession() = default;
TrtSession::~TrtSession() = default;
TrtSession::TrtSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TrtSession::TrtSession(TrtSession&&) noexcept = default;
TrtSession& TrtSession::operator=(TrtSession&&) noexcept = default;

TrtSession TrtSession::load(const std::string& engine_path) {
    if (engine_path.empty()) throw std::invalid_argument("TensorRT engine path is empty");
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Unable to open TensorRT engine: " + engine_path);
    const std::streamsize size = file.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty: " + engine_path);
    file.seekg(0, std::ios::beg);
    std::vector<char> serialized(static_cast<std::size_t>(size));
    if (!file.read(serialized.data(), size)) throw std::runtime_error("Unable to read TensorRT engine: " + engine_path);
    auto impl = std::make_unique<Impl>();
    impl->runtime = nvinfer1::createInferRuntime(g_logger);
    if (!impl->runtime) throw std::runtime_error("TensorRT runtime creation failed");
    impl->engine = impl->runtime->deserializeCudaEngine(serialized.data(), serialized.size());
    if (!impl->engine) throw std::runtime_error("TensorRT engine deserialization failed");
    impl->context = impl->engine->createExecutionContext();
    if (!impl->context) throw std::runtime_error("TensorRT execution context creation failed");
    check_cuda(cudaStreamCreate(&impl->stream), "cudaStreamCreate");
    const int count = impl->engine->getNbBindings();
    impl->buffers.resize(static_cast<std::size_t>(count), nullptr);
    impl->owned_buffers.resize(static_cast<std::size_t>(count), nullptr);
    impl->buffer_bytes.resize(static_cast<std::size_t>(count), 0);
    for (int i = 0; i < count; ++i) impl->bindings.emplace(impl->engine->getBindingName(i), i);
    return TrtSession(std::move(impl));
}

std::vector<std::string> TrtSession::input_names() const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    std::vector<std::string> result;
    for (const auto& entry : impl_->bindings) if (impl_->engine->bindingIsInput(entry.second)) result.push_back(entry.first);
    return result;
}

std::vector<std::string> TrtSession::output_names() const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    std::vector<std::string> result;
    for (const auto& entry : impl_->bindings) if (!impl_->engine->bindingIsInput(entry.second)) result.push_back(entry.first);
    return result;
}

std::vector<int64_t> TrtSession::tensor_shape(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const int index = impl_->binding(name);
    auto dims = impl_->engine->getBindingDimensions(index);
    bool dynamic = false;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            dynamic = true;
            break;
        }
    }
    // 动态输入没有静态 binding shape 时，使用 profile 0 的优化尺寸。
    // 该尺寸与 TensorRT builder 选择的常用推理尺寸一致，适合作为
    // 前处理和 set_shape 的默认输入尺寸。
    if (dynamic && impl_->engine->bindingIsInput(index) &&
        impl_->engine->getNbOptimizationProfiles() > 0) {
        dims = impl_->engine->getProfileDimensions(
            index, 0, nvinfer1::OptProfileSelector::kOPT);
    }
    return to_shape(dims);
}

void TrtSession::set_shape(const std::string& name, const std::vector<int64_t>& shape) {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const int index = impl_->binding(name);
    if (!impl_->engine->bindingIsInput(index)) throw std::invalid_argument("Only input tensors can have shapes set");
    if (!impl_->context->setBindingDimensions(index, to_dims(shape))) throw std::runtime_error("TensorRT rejected shape: " + name);
}

void TrtSession::set_input(const std::string& name, const void* data, std::size_t bytes) {
    if (!impl_ || !data) throw std::invalid_argument("Input data is null");
    const int index = impl_->binding(name);
    if (!impl_->engine->bindingIsInput(index)) throw std::invalid_argument("Tensor is not an input: " + name);
    impl_->inputs[name] = std::vector<std::uint8_t>(static_cast<const std::uint8_t*>(data), static_cast<const std::uint8_t*>(data) + bytes);
    impl_->device_inputs.erase(name);
    impl_->device_input_bytes.erase(name);
    (void)index;
}

void TrtSession::set_input_device(const std::string& name, const void* device_data, std::size_t bytes) {
    if (!impl_ || !device_data) throw std::invalid_argument("Device input data is null");
    const int index = impl_->binding(name);
    if (!impl_->engine->bindingIsInput(index)) throw std::invalid_argument("Tensor is not an input: " + name);
    impl_->device_inputs[name] = device_data;
    impl_->device_input_bytes[name] = bytes;
    impl_->inputs.erase(name);
}

void TrtSession::set_output_host_copy(bool enabled) {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->output_host_copy = enabled;
}

void TrtSession::infer() {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    if (!impl_->context->allInputDimensionsSpecified()) throw std::runtime_error("Dynamic input dimensions were not specified");
    for (const auto& entry : impl_->bindings) {
        const int index = entry.second;
        const auto dims = impl_->context->getBindingDimensions(index);
        const std::size_t bytes = volume(dims) * element_size(impl_->engine->getBindingDataType(index));
        if (impl_->engine->bindingIsInput(index)) {
            const auto device_input = impl_->device_inputs.find(entry.first);
            if (device_input != impl_->device_inputs.end()) {
                const auto size = impl_->device_input_bytes.find(entry.first);
                if (size == impl_->device_input_bytes.end() || size->second != bytes)
                    throw std::runtime_error("Device input byte size mismatch: " + entry.first);
                impl_->buffers[static_cast<std::size_t>(index)] = const_cast<void*>(device_input->second);
                continue;
            }
        }
        if (impl_->buffer_bytes[static_cast<std::size_t>(index)] < bytes) {
            if (impl_->owned_buffers[static_cast<std::size_t>(index)])
                check_cuda(cudaFree(impl_->owned_buffers[static_cast<std::size_t>(index)]), "cudaFree");
            check_cuda(cudaMalloc(&impl_->owned_buffers[static_cast<std::size_t>(index)], bytes), "cudaMalloc");
            impl_->buffer_bytes[static_cast<std::size_t>(index)] = bytes;
        }
        impl_->buffers[static_cast<std::size_t>(index)] = impl_->owned_buffers[static_cast<std::size_t>(index)];
        if (impl_->engine->bindingIsInput(index)) {
            const auto input = impl_->inputs.find(entry.first);
            if (input == impl_->inputs.end() || input->second.size() != bytes)
                throw std::runtime_error("Input byte size mismatch: " + entry.first);
            check_cuda(cudaMemcpyAsync(impl_->buffers[static_cast<std::size_t>(index)], input->second.data(), bytes,
                                       cudaMemcpyHostToDevice, impl_->stream), "cudaMemcpyAsync(H2D)");
        }
    }
    if (!impl_->context->enqueueV2(impl_->buffers.data(), impl_->stream, nullptr))
        throw std::runtime_error("TensorRT enqueueV2 failed");
    impl_->outputs.clear();
    impl_->output_shapes.clear();
    for (const auto& entry : impl_->bindings) {
        const int index = entry.second;
        if (impl_->engine->bindingIsInput(index)) continue;
        const auto dims = impl_->context->getBindingDimensions(index);
        const std::size_t bytes = volume(dims) * element_size(impl_->engine->getBindingDataType(index));
        const auto shape = to_shape(dims);
        impl_->output_shapes.emplace(entry.first, shape);
        if (!impl_->output_host_copy) continue;
        Tensor output;
        output.name = entry.first;
        output.shape = shape;
        output.data_type = to_data_type(impl_->engine->getBindingDataType(index));
        output.data.resize(bytes);
        check_cuda(cudaMemcpyAsync(output.data.data(), impl_->buffers[static_cast<std::size_t>(index)], bytes,
                                   cudaMemcpyDeviceToHost, impl_->stream), "cudaMemcpyAsync(D2H)");
        impl_->outputs.emplace(output.name, std::move(output));
    }
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
}

Tensor TrtSession::get_output(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const auto it = impl_->outputs.find(name);
    if (it == impl_->outputs.end()) throw std::invalid_argument("Output tensor is unavailable: " + name);
    return it->second;
}

const void* TrtSession::device_output(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const int index = impl_->binding(name);
    if (impl_->engine->bindingIsInput(index)) throw std::invalid_argument("Tensor is not an output: " + name);
    if (impl_->output_shapes.find(name) == impl_->output_shapes.end()) throw std::invalid_argument("Output tensor is unavailable: " + name);
    return impl_->buffers[static_cast<std::size_t>(index)];
}

std::vector<int64_t> TrtSession::device_output_shape(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const int index = impl_->binding(name);
    if (impl_->engine->bindingIsInput(index)) throw std::invalid_argument("Tensor is not an output: " + name);
    return to_shape(impl_->context->getBindingDimensions(index));
}

TensorDataType TrtSession::device_output_type(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const int index = impl_->binding(name);
    if (impl_->engine->bindingIsInput(index)) throw std::invalid_argument("Tensor is not an output: " + name);
    if (impl_->output_shapes.find(name) == impl_->output_shapes.end())
        throw std::invalid_argument("Output tensor is unavailable: " + name);
    return to_data_type(impl_->engine->getBindingDataType(index));
}
void* TrtSession::cuda_stream() const noexcept {
    return impl_ ? reinterpret_cast<void*>(impl_->stream) : nullptr;
}
std::vector<Tensor> TrtSession::outputs() const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    std::vector<Tensor> result; result.reserve(impl_->outputs.size());
    for (const auto& item : impl_->outputs) result.push_back(item.second);
    return result;
}
} // namespace cvinfer::core




