#include "core/trt_core.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cvinfer::core {
#if NV_TENSORRT_MAJOR < 10
#error "The trt10 branch requires TensorRT 10 or newer"
#endif
cvinfer::core::TensorDataType to_data_type(nvinfer1::DataType type) {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return TensorDataType::Float32;
    case nvinfer1::DataType::kHALF: return TensorDataType::Float16;
    case nvinfer1::DataType::kINT8: return TensorDataType::Int8;
    case nvinfer1::DataType::kINT32: return TensorDataType::Int32;
    case nvinfer1::DataType::kBOOL: return TensorDataType::Bool;
    case nvinfer1::DataType::kINT64: return TensorDataType::Int64;
    case nvinfer1::DataType::kUINT8: return TensorDataType::UInt8;
    case nvinfer1::DataType::kBF16: return TensorDataType::BFloat16;
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
    case nvinfer1::DataType::kINT64: return sizeof(std::int64_t);
    case nvinfer1::DataType::kUINT8: return sizeof(std::uint8_t);
    case nvinfer1::DataType::kBF16: return sizeof(std::uint16_t);
    default: throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}
std::size_t volume(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) throw std::runtime_error("Tensor shape is unresolved or invalid");
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("Tensor shape is unresolved or invalid");
        const auto dimension = static_cast<std::size_t>(dims.d[i]);
        if (result > std::numeric_limits<std::size_t>::max() / dimension)
            throw std::overflow_error("Tensor volume exceeds addressable memory");
        result *= dimension;
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
        if (shape[static_cast<std::size_t>(i)] <= 0)
            throw std::invalid_argument("Tensor dimensions must be positive");
        dims.d[i] = shape[static_cast<std::size_t>(i)];
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
    std::unordered_map<std::string, int> tensor_indices;
    std::unordered_map<std::string, std::vector<std::uint8_t>> inputs;
    std::unordered_map<std::string, const void*> device_inputs;
    std::unordered_map<std::string, std::size_t> device_input_bytes;
    std::unordered_map<std::string, Tensor> outputs;
    std::unordered_map<std::string, std::vector<int64_t>> output_shapes;
    bool output_host_copy{true};
    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        delete context;
        for (void* buffer : owned_buffers) if (buffer) cudaFree(buffer);
        if (stream) cudaStreamDestroy(stream);
        delete engine;
        delete runtime;
    }
    int tensor_index(const std::string& name) const {
        const auto it = tensor_indices.find(name);
        if (it == tensor_indices.end()) throw std::invalid_argument("Unknown TensorRT tensor: " + name);
        return it->second;
    }
};

TrtSession::TrtSession() = default;
TrtSession::~TrtSession() = default;
TrtSession::TrtSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TrtSession::TrtSession(TrtSession&&) noexcept = default;
TrtSession& TrtSession::operator=(TrtSession&&) noexcept = default;

TrtSession TrtSession::load(const std::string& engine_path) {
    if (engine_path.empty()) throw std::invalid_argument("TensorRT engine path is empty");
    std::ifstream file(std::filesystem::u8path(engine_path), std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Unable to open TensorRT engine: " + engine_path);
    const std::streamsize size = file.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty: " + engine_path);
    file.seekg(0, std::ios::beg);
    std::vector<char> serialized(static_cast<std::size_t>(size));
    if (!file.read(serialized.data(), size)) throw std::runtime_error("Unable to read TensorRT engine: " + engine_path);
    auto impl = std::make_unique<Impl>();
    if (!initLibNvInferPlugins(&g_logger, ""))
        throw std::runtime_error("TensorRT plugin initialization failed");
    impl->runtime = nvinfer1::createInferRuntime(g_logger);
    if (!impl->runtime) throw std::runtime_error("TensorRT runtime creation failed");
    impl->engine = impl->runtime->deserializeCudaEngine(serialized.data(), serialized.size());
    if (!impl->engine) throw std::runtime_error("TensorRT engine deserialization failed; rebuild old engines with this TensorRT version. " + g_logger.last_message_);
    impl->context = impl->engine->createExecutionContext();
    if (!impl->context) throw std::runtime_error("TensorRT execution context creation failed");
    check_cuda(cudaStreamCreate(&impl->stream), "cudaStreamCreate");
    const int count = impl->engine->getNbIOTensors();
    impl->buffers.resize(static_cast<std::size_t>(count), nullptr);
    impl->owned_buffers.resize(static_cast<std::size_t>(count), nullptr);
    impl->buffer_bytes.resize(static_cast<std::size_t>(count), 0);
    for (int i = 0; i < count; ++i) {
        const char* name = impl->engine->getIOTensorName(i);
        if (impl->engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
            impl->engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR)
            throw std::runtime_error(std::string("Only linear device I/O tensors are supported: ") + name);
        impl->tensor_indices.emplace(name, i);
    }
    return TrtSession(std::move(impl));
}

std::vector<std::string> TrtSession::input_names() const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    std::vector<std::string> result;
    for (const auto& entry : impl_->tensor_indices) if (impl_->engine->getTensorIOMode(entry.first.c_str()) == nvinfer1::TensorIOMode::kINPUT) result.push_back(entry.first);
    return result;
}

std::vector<std::string> TrtSession::output_names() const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    std::vector<std::string> result;
    for (const auto& entry : impl_->tensor_indices) if (impl_->engine->getTensorIOMode(entry.first.c_str()) == nvinfer1::TensorIOMode::kOUTPUT) result.push_back(entry.first);
    return result;
}

std::vector<int64_t> TrtSession::tensor_shape(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->tensor_index(name); // Validate the name before querying TensorRT.
    auto dims = impl_->engine->getTensorShape(name.c_str());
    bool dynamic = false;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            dynamic = true;
            break;
        }
    }
    // 动态输入没有静态 shape 时，使用 profile 0 的优化尺寸。
    // 该尺寸与 TensorRT builder 选择的常用推理尺寸一致，适合作为
    // 前处理和 set_shape 的默认输入尺寸。
    if (dynamic && (impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) &&
        impl_->engine->getNbOptimizationProfiles() > 0) {
        dims = impl_->engine->getProfileShape(
            name.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
    }
    return to_shape(dims);
}

void TrtSession::set_shape(const std::string& name, const std::vector<int64_t>& shape) {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->tensor_index(name);
    if (!(impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)) throw std::invalid_argument("Only input tensors can have shapes set");
    if (!impl_->context->setInputShape(name.c_str(), to_dims(shape))) throw std::runtime_error("TensorRT rejected shape: " + name);
}

void TrtSession::set_input(const std::string& name, const void* data, std::size_t bytes) {
    if (!impl_ || !data) throw std::invalid_argument("Input data is null");
    impl_->tensor_index(name);
    if (!(impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)) throw std::invalid_argument("Tensor is not an input: " + name);
    auto& input = impl_->inputs[name];
    input.resize(bytes);
    if (bytes > 0) std::memcpy(input.data(), data, bytes);
    impl_->device_inputs.erase(name);
    impl_->device_input_bytes.erase(name);
}

void TrtSession::set_input_device(const std::string& name, const void* device_data, std::size_t bytes) {
    if (!impl_ || !device_data) throw std::invalid_argument("Device input data is null");
    impl_->tensor_index(name);
    if (!(impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)) throw std::invalid_argument("Tensor is not an input: " + name);
    impl_->device_inputs[name] = device_data;
    impl_->device_input_bytes[name] = bytes;
    impl_->inputs.erase(name);
}

void TrtSession::set_output_host_copy(bool enabled) {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->output_host_copy = enabled;
}

void TrtSession::infer_async() {
    // The TensorRT backend currently synchronizes inside infer(); retain that
    // behavior while satisfying the backend-neutral Session interface.
    infer();
}

void TrtSession::synchronize() {}

bool TrtSession::is_pending() const noexcept { return false; }

void TrtSession::infer() {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->outputs.clear();
    impl_->output_shapes.clear();
    const auto allocate = [&](const std::string& name, int index) {
        const auto dims = impl_->context->getTensorShape(name.c_str());
        const auto size = element_size(impl_->engine->getTensorDataType(name.c_str()));
        const auto elements = volume(dims);
        if (elements > std::numeric_limits<std::size_t>::max() / size)
            throw std::overflow_error("Tensor byte size exceeds addressable memory: " + name);
        const std::size_t bytes = elements * size;
        if (impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            const auto device_input = impl_->device_inputs.find(name);
            if (device_input != impl_->device_inputs.end()) {
                if (impl_->device_input_bytes.at(name) != bytes)
                    throw std::runtime_error("Device input byte size mismatch: " + name);
                impl_->buffers[index] = const_cast<void*>(device_input->second);
                return bytes;
            }
        }
        if (impl_->buffer_bytes[index] < bytes) {
            if (impl_->owned_buffers[index]) {
                check_cuda(cudaFree(impl_->owned_buffers[index]), "cudaFree");
                impl_->owned_buffers[index] = nullptr;
                impl_->buffer_bytes[index] = 0;
            }
            check_cuda(cudaMalloc(&impl_->owned_buffers[index], bytes), "cudaMalloc");
            impl_->buffer_bytes[index] = bytes;
        }
        impl_->buffers[index] = impl_->owned_buffers[index];
        return bytes;
    };
    for (const auto& entry : impl_->tensor_indices) {
        const auto& name = entry.first;
        if (impl_->engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT) continue;
        const auto bytes = allocate(name, entry.second);
        if (!impl_->context->setTensorAddress(name.c_str(), impl_->buffers[entry.second]))
            throw std::runtime_error("TensorRT rejected input address: " + name);
        if (impl_->device_inputs.count(name)) continue;
        const auto input = impl_->inputs.find(name);
        if (input == impl_->inputs.end() || input->second.size() != bytes)
            throw std::runtime_error("Input byte size mismatch: " + name);
        check_cuda(cudaMemcpyAsync(impl_->buffers[entry.second], input->second.data(), bytes,
                                   cudaMemcpyHostToDevice, impl_->stream), "cudaMemcpyAsync(H2D)");
    }
    // allInputDimensionsSpecified() does not validate the name-based API.
    // inferShapes() is the TensorRT 10 replacement after input shapes/addresses.
    if (impl_->context->inferShapes(0, nullptr) != 0)
        throw std::runtime_error("TensorRT input shapes or shape tensor values are unresolved");
    for (const auto& entry : impl_->tensor_indices) {
        if (impl_->engine->getTensorIOMode(entry.first.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) continue;
        allocate(entry.first, entry.second);
        if (!impl_->context->setTensorAddress(entry.first.c_str(), impl_->buffers[entry.second]))
            throw std::runtime_error("TensorRT rejected output address: " + entry.first);
    }
    if (!impl_->context->enqueueV3(impl_->stream))
        throw std::runtime_error("TensorRT enqueueV3 failed: " + g_logger.last_message_);
    for (const auto& entry : impl_->tensor_indices) {
        const auto& name = entry.first;
        if (impl_->engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) continue;
        const auto dims = impl_->context->getTensorShape(name.c_str());
        const auto bytes = volume(dims) * element_size(impl_->engine->getTensorDataType(name.c_str()));
        const auto shape = to_shape(dims);
        impl_->output_shapes.emplace(name, shape);
        if (!impl_->output_host_copy) continue;
        Tensor output;
        output.name = name;
        output.shape = shape;
        output.data_type = to_data_type(impl_->engine->getTensorDataType(name.c_str()));
        output.data.resize(bytes);
        check_cuda(cudaMemcpyAsync(output.data.data(), impl_->buffers[entry.second], bytes,
                                   cudaMemcpyDeviceToHost, impl_->stream), "cudaMemcpyAsync(D2H)");
        impl_->outputs.emplace(name, std::move(output));
    }
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
}

Tensor TrtSession::get_output(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const auto it = impl_->outputs.find(name);
    if (it == impl_->outputs.end()) throw std::invalid_argument("Output tensor is unavailable: " + name);
    return it->second;
}

const Tensor& TrtSession::get_output_ref(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const auto it = impl_->outputs.find(name);
    if (it == impl_->outputs.end()) throw std::invalid_argument("Output tensor is unavailable: " + name);
    return it->second;
}

const void* TrtSession::device_output(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    const int index = impl_->tensor_index(name);
    if ((impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT)) throw std::invalid_argument("Tensor is not an output: " + name);
    if (impl_->output_shapes.find(name) == impl_->output_shapes.end()) throw std::invalid_argument("Output tensor is unavailable: " + name);
    return impl_->buffers[static_cast<std::size_t>(index)];
}

std::vector<int64_t> TrtSession::device_output_shape(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->tensor_index(name);
    if (impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) throw std::invalid_argument("Tensor is not an output: " + name);
    return to_shape(impl_->context->getTensorShape(name.c_str()));
}

TensorDataType TrtSession::device_output_type(const std::string& name) const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    impl_->tensor_index(name);
    if (impl_->engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) throw std::invalid_argument("Tensor is not an output: " + name);
    if (impl_->output_shapes.find(name) == impl_->output_shapes.end())
        throw std::invalid_argument("Output tensor is unavailable: " + name);
    return to_data_type(impl_->engine->getTensorDataType(name.c_str()));
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

std::vector<const Tensor*> TrtSession::output_refs() const {
    if (!impl_) throw std::runtime_error("TensorRT session is not initialized");
    std::vector<const Tensor*> result;
    result.reserve(impl_->outputs.size());
    for (const auto& item : impl_->outputs) result.push_back(&item.second);
    return result;
}
} // namespace cvinfer::core
