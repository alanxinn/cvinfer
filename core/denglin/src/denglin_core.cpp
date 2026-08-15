#include "core/denglin_core.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlnne/dlnne.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cvinfer::core {
namespace {

template <typename T>
struct NneDeleter {
    void operator()(T* object) const noexcept {
        if (object) object->Destroy();
    }
};

using BuilderPtr = std::unique_ptr<dl::nne::Builder, NneDeleter<dl::nne::Builder>>;
using ParserPtr = std::unique_ptr<dl::nne::Parser, NneDeleter<dl::nne::Parser>>;
using NetworkPtr = std::unique_ptr<dl::nne::Network, NneDeleter<dl::nne::Network>>;
using EnginePtr = std::unique_ptr<dl::nne::Engine, NneDeleter<dl::nne::Engine>>;
using ContextPtr = std::unique_ptr<dl::nne::ExecutionContext, NneDeleter<dl::nne::ExecutionContext>>;

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void check_driver_init() {
    if (cuInit(0) == CUDA_SUCCESS) return;
    throw std::runtime_error("Denglin CUDA driver initialization failed");
}

bool has_unresolved_dimension(const dl::nne::Dims& dims) {
    if (dims.nbDims < 0) return true;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return true;
    }
    return false;
}

std::vector<int64_t> to_shape(const dl::nne::Dims& dims) {
    if (dims.nbDims < 0 || dims.nbDims > dl::nne::Dims::MAX_DIMS) {
        return {};
    }
    std::vector<int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) {
        shape.push_back(static_cast<int64_t>(dims.d[i]));
    }
    return shape;
}

dl::nne::Dims to_dims(const std::vector<int64_t>& shape,
                      bool allow_dynamic = false) {
    if (shape.empty() || shape.size() > static_cast<std::size_t>(dl::nne::Dims::MAX_DIMS)) {
        throw std::invalid_argument("Denglin input shape is invalid");
    }
    dl::nne::Dims dims{};
    dims.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        const auto value = shape[static_cast<std::size_t>(i)];
        if (value == 0 || value < -1 || value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Denglin input dimensions are invalid");
        }
        if (value < 0 && !allow_dynamic) {
            throw std::invalid_argument("Denglin runtime dimensions must be positive");
        }
        dims.d[i] = static_cast<int>(value);
    }
    return dims;
}

std::size_t element_size(dl::nne::DataType type) {
    switch (type) {
    case dl::nne::kFLOAT16:
    case dl::nne::kUINT16:
    case dl::nne::kINT16:
    case dl::nne::kBFLOAT16:
        return 2;
    case dl::nne::kFLOAT32:
    case dl::nne::kINT32:
    case dl::nne::kUINT32:
        return 4;
    case dl::nne::kFLOAT64:
    case dl::nne::kINT64:
    case dl::nne::kUINT64:
        return 8;
    case dl::nne::kINT8:
    case dl::nne::kUINT8:
    case dl::nne::kBOOL:
        return 1;
    case dl::nne::kINT4:
    case dl::nne::kUINT4:
        return 0; // handled as packed nibbles by tensor_bytes
    default:
        throw std::runtime_error("Denglin tensor data type is unknown");
    }
}

std::size_t tensor_volume(const dl::nne::Dims& dims) {
    if (dims.nbDims < 0 || dims.nbDims > dl::nne::Dims::MAX_DIMS) {
        throw std::runtime_error("Denglin tensor dimensions are invalid");
    }
    std::size_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("Denglin tensor shape is unresolved");
        const auto dimension = static_cast<std::size_t>(dims.d[i]);
        if (volume > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("Denglin tensor size overflow");
        }
        volume *= dimension;
    }
    return volume;
}

std::size_t tensor_bytes(const dl::nne::Dims& dims, dl::nne::DataType type) {
    const auto volume = tensor_volume(dims);
    if (type == dl::nne::kINT4 || type == dl::nne::kUINT4) {
        return (volume + 1U) / 2U;
    }
    const auto bytes_per_element = element_size(type);
    if (volume > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
        throw std::overflow_error("Denglin tensor byte size overflow");
    }
    return volume * bytes_per_element;
}

TensorDataType to_data_type(dl::nne::DataType type) {
    switch (type) {
    case dl::nne::kFLOAT32: return TensorDataType::Float32;
    case dl::nne::kFLOAT16: return TensorDataType::Float16;
    case dl::nne::kFLOAT64: return TensorDataType::Float64;
    case dl::nne::kINT8: return TensorDataType::Int8;
    case dl::nne::kINT16: return TensorDataType::Int16;
    case dl::nne::kINT32: return TensorDataType::Int32;
    case dl::nne::kINT64: return TensorDataType::Int64;
    case dl::nne::kUINT8: return TensorDataType::UInt8;
    case dl::nne::kUINT16: return TensorDataType::UInt16;
    case dl::nne::kUINT32: return TensorDataType::UInt32;
    case dl::nne::kUINT64: return TensorDataType::UInt64;
    case dl::nne::kBOOL: return TensorDataType::Bool;
    case dl::nne::kINT4: return TensorDataType::Int4;
    case dl::nne::kUINT4: return TensorDataType::UInt4;
    case dl::nne::kBFLOAT16: return TensorDataType::BFloat16;
    default: return TensorDataType::Unknown;
    }
}

std::vector<std::uint8_t> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Unable to open Denglin model: " + path);
    const auto end = file.tellg();
    const auto size = static_cast<std::streamsize>(end);
    if (size <= 0) throw std::runtime_error("Denglin model is empty: " + path);
    file.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Unable to read Denglin model: " + path);
    }
    return data;
}

struct NneHostMemoryDeleter {
    void operator()(dl::nne::HostMemory* memory) const noexcept {
        if (memory) memory->Destroy();
    }
};

using HostMemoryPtr = std::unique_ptr<dl::nne::HostMemory, NneHostMemoryDeleter>;

constexpr char kDenglinCacheMagic[] = "CVINFER-DL-CACHE";
constexpr std::uint32_t kDenglinCacheVersion = 2;
constexpr std::size_t kMaxCacheMetadataBytes = 16U * 1024U * 1024U;

struct CacheMetadata {
    std::string sdk_version{"unknown"};
    std::string device;
    std::string model_format;
    std::string model_hash;
    std::string input_name;
    std::vector<std::int64_t> input_shape;
    std::vector<DenglinOptimizationProfile> optimization_profiles;
    TensorDataType input_type{TensorDataType::Unknown};
    int max_batch_size{1};
    int batch_size{1};
    int weight_share{1};
    int cluster{0};
    std::uint32_t builder_flags{};
    int input_layout{1};
    std::vector<std::string> output_names;
    std::string plugin_library;
    std::string plugin_module;
    std::string plugin_first_op;
};

bool same_profile(const DenglinOptimizationProfile& left,
                  const DenglinOptimizationProfile& right) {
    return left.min_shape == right.min_shape &&
           left.opt_shape == right.opt_shape &&
           left.max_shape == right.max_shape;
}

bool same_profiles(const std::vector<DenglinOptimizationProfile>& left,
                   const std::vector<DenglinOptimizationProfile>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!same_profile(left[i], right[i])) return false;
    }
    return true;
}

void validate_profile_shape(const std::vector<int64_t>& shape,
                            const std::vector<int64_t>& input_shape,
                            const char* label) {
    if (shape.empty() || shape.size() > static_cast<std::size_t>(dl::nne::Dims::MAX_DIMS) ||
        shape.size() != input_shape.size()) {
        throw std::invalid_argument(std::string("Denglin ") + label + " shape is invalid");
    }
    for (std::size_t i = 0; i < shape.size(); ++i) {
        const auto value = shape[i];
        if (value <= 0 || value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(std::string("Denglin ") + label +
                                        " dimensions must be positive");
        }
        if (input_shape[i] > 0 && input_shape[i] != value) {
            throw std::invalid_argument(std::string("Denglin ") + label +
                                        " does not match the static input dimension");
        }
    }
}

void validate_profiles(const DenglinSessionOptions& options) {
    if (options.optimization_profile < -1) {
        throw std::invalid_argument("Denglin optimization profile index is invalid");
    }
    if (options.optimization_profiles.size() > 1024U) {
        throw std::invalid_argument("Denglin optimization profile list is too large");
    }
    const bool dynamic_input = std::any_of(
        options.input_shape.begin(), options.input_shape.end(),
        [](std::int64_t dimension) { return dimension < 0; });
    if (dynamic_input && options.optimization_profiles.empty()) {
        throw std::invalid_argument(
            "Denglin dynamic input requires at least one optimization profile");
    }
    for (const auto& profile : options.optimization_profiles) {
        validate_profile_shape(profile.min_shape, options.input_shape, "profile min");
        validate_profile_shape(profile.opt_shape, options.input_shape, "profile opt");
        validate_profile_shape(profile.max_shape, options.input_shape, "profile max");
        for (std::size_t i = 0; i < profile.min_shape.size(); ++i) {
            if (profile.min_shape[i] > profile.opt_shape[i] ||
                profile.opt_shape[i] > profile.max_shape[i]) {
                throw std::invalid_argument(
                    "Denglin optimization profile requires min <= opt <= max");
            }
        }
        if (!profile.max_shape.empty() &&
            profile.max_shape.front() > options.max_batch_size) {
            throw std::invalid_argument(
                "Denglin optimization profile max batch exceeds max_batch_size");
        }
    }
    if (!options.optimization_profiles.empty() &&
        options.optimization_profile >=
            static_cast<int>(options.optimization_profiles.size())) {
        throw std::invalid_argument("Denglin optimization profile index is out of range");
    }
}

struct ByteWriter {
    std::vector<std::uint8_t> bytes;

    void u32(std::uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
        }
    }

    void u64(std::uint64_t value) {
        for (int i = 0; i < 8; ++i) {
            bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
        }
    }

    void i64(std::int64_t value) {
        u64(static_cast<std::uint64_t>(value));
    }

    void string(const std::string& value) {
        u64(static_cast<std::uint64_t>(value.size()));
        bytes.insert(bytes.end(), value.begin(), value.end());
    }

    void raw(const void* data, std::size_t size) {
        if (size == 0) return;
        const auto* first = static_cast<const std::uint8_t*>(data);
        bytes.insert(bytes.end(), first, first + size);
    }
};

struct ByteReader {
    const std::uint8_t* data{};
    std::size_t size{};
    std::size_t offset{};

    explicit ByteReader(const std::vector<std::uint8_t>& bytes)
        : data(bytes.data()), size(bytes.size()) {}

    void require(std::size_t count) {
        if (count > size - offset) throw std::runtime_error("Denglin cache is truncated");
    }

    std::uint32_t u32() {
        require(4);
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            value |= static_cast<std::uint32_t>(data[offset++]) << (i * 8);
        }
        return value;
    }

    std::uint64_t u64() {
        require(8);
        std::uint64_t value = 0;
        for (int i = 0; i < 8; ++i) {
            value |= static_cast<std::uint64_t>(data[offset++]) << (i * 8);
        }
        return value;
    }

    std::int64_t i64() {
        return static_cast<std::int64_t>(u64());
    }

    std::string string() {
        const auto length = u64();
        if (length > size - offset || length > kMaxCacheMetadataBytes) {
            throw std::runtime_error("Denglin cache string is invalid");
        }
        std::string value(reinterpret_cast<const char*>(data + offset),
                          static_cast<std::size_t>(length));
        offset += static_cast<std::size_t>(length);
        return value;
    }

    std::vector<std::uint8_t> bytes(std::size_t count) {
        require(count);
        std::vector<std::uint8_t> value(data + offset, data + offset + count);
        offset += count;
        return value;
    }
};

std::uint64_t fnv1a(const std::uint8_t* data, std::size_t size) {
    constexpr std::uint64_t offset_basis = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    auto hash = offset_basis;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= prime;
    }
    return hash;
}

std::string hex_u64(std::uint64_t value) {
    std::ostringstream result;
    result << std::hex << value;
    return result.str();
}

std::string file_hash(const std::string& path) {
    const auto data = read_binary_file(path);
    return hex_u64(fnv1a(data.data(), data.size()));
}

std::string device_fingerprint() {
    CUdevice device{};
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return "unknown";

    char name[256]{};
    if (cuDeviceGetName(name, static_cast<int>(sizeof(name)), device) != CUDA_SUCCESS) {
        return "unknown";
    }
    int major = 0;
    int minor = 0;
    (void)cuDeviceComputeCapability(&major, &minor, device);
    std::ostringstream result;
    result << name << "|cc=" << major << '.' << minor;
    return result.str();
}

std::string sdk_version() {
#ifdef CVINFER_DENGLIN_SDK_VERSION
    return CVINFER_DENGLIN_SDK_VERSION;
#else
    return "unknown";
#endif
}

void write_cache_metadata(ByteWriter& writer, const CacheMetadata& metadata) {
    writer.string(metadata.sdk_version);
    writer.string(metadata.device);
    writer.string(metadata.model_format);
    writer.string(metadata.model_hash);
    writer.string(metadata.input_name);
    writer.u32(static_cast<std::uint32_t>(metadata.input_shape.size()));
    for (const auto dimension : metadata.input_shape) writer.i64(dimension);
    writer.u32(static_cast<std::uint32_t>(metadata.optimization_profiles.size()));
    for (const auto& profile : metadata.optimization_profiles) {
        for (const auto* shape : {&profile.min_shape, &profile.opt_shape,
                                  &profile.max_shape}) {
            writer.u32(static_cast<std::uint32_t>(shape->size()));
            for (const auto dimension : *shape) writer.i64(dimension);
        }
    }
    writer.u32(static_cast<std::uint32_t>(metadata.input_type));
    writer.u32(static_cast<std::uint32_t>(metadata.max_batch_size));
    writer.u32(static_cast<std::uint32_t>(metadata.batch_size));
    writer.u32(static_cast<std::uint32_t>(metadata.weight_share));
    writer.u32(static_cast<std::uint32_t>(metadata.cluster));
    writer.u32(metadata.builder_flags);
    writer.u32(static_cast<std::uint32_t>(metadata.input_layout));
    writer.u32(static_cast<std::uint32_t>(metadata.output_names.size()));
    for (const auto& name : metadata.output_names) writer.string(name);
    writer.string(metadata.plugin_library);
    writer.string(metadata.plugin_module);
    writer.string(metadata.plugin_first_op);
}

CacheMetadata read_cache_metadata(const std::vector<std::uint8_t>& bytes) {
    ByteReader reader(bytes);
    CacheMetadata metadata;
    metadata.sdk_version = reader.string();
    metadata.device = reader.string();
    metadata.model_format = reader.string();
    metadata.model_hash = reader.string();
    metadata.input_name = reader.string();
    const auto shape_count = reader.u32();
    if (shape_count > dl::nne::Dims::MAX_DIMS) {
        throw std::runtime_error("Denglin cache input shape is invalid");
    }
    metadata.input_shape.reserve(shape_count);
    for (std::uint32_t i = 0; i < shape_count; ++i) {
        const auto dimension = reader.i64();
        if (dimension == 0 || dimension < -1) {
            throw std::runtime_error("Denglin cache input shape is invalid");
        }
        metadata.input_shape.push_back(dimension);
    }
    const auto profile_count = reader.u32();
    if (profile_count > 1024U) {
        throw std::runtime_error("Denglin cache optimization profile list is invalid");
    }
    metadata.optimization_profiles.reserve(profile_count);
    for (std::uint32_t profile_index = 0; profile_index < profile_count;
         ++profile_index) {
        DenglinOptimizationProfile profile;
        for (auto* shape : {&profile.min_shape, &profile.opt_shape,
                            &profile.max_shape}) {
            const auto count = reader.u32();
            if (count == 0 || count > dl::nne::Dims::MAX_DIMS) {
                throw std::runtime_error("Denglin cache profile shape is invalid");
            }
            shape->reserve(count);
            for (std::uint32_t dimension_index = 0; dimension_index < count;
                 ++dimension_index) {
                const auto dimension = reader.i64();
                if (dimension <= 0 || dimension > std::numeric_limits<int>::max()) {
                    throw std::runtime_error("Denglin cache profile dimension is invalid");
                }
                shape->push_back(dimension);
            }
        }
        metadata.optimization_profiles.push_back(std::move(profile));
    }
    const auto input_type = reader.u32();
    if (input_type > static_cast<std::uint32_t>(TensorDataType::Unknown)) {
        throw std::runtime_error("Denglin cache input dtype is invalid");
    }
    metadata.input_type = static_cast<TensorDataType>(input_type);
    metadata.max_batch_size = static_cast<int>(reader.u32());
    metadata.batch_size = static_cast<int>(reader.u32());
    metadata.weight_share = static_cast<int>(reader.u32());
    metadata.cluster = static_cast<int>(reader.u32());
    metadata.builder_flags = reader.u32();
    metadata.input_layout = static_cast<int>(reader.u32());
    if (metadata.max_batch_size <= 0 || metadata.batch_size <= 0 ||
        metadata.batch_size > metadata.max_batch_size ||
        (metadata.weight_share != 1 && metadata.weight_share != 2 &&
         metadata.weight_share != 4) ||
        metadata.cluster < 0 || metadata.cluster > 10 ||
        metadata.input_layout < 0 || metadata.input_layout > 2) {
        throw std::runtime_error("Denglin cache build configuration is invalid");
    }
    const auto output_count = reader.u32();
    if (output_count > 1U << 20) throw std::runtime_error("Denglin cache output list is invalid");
    metadata.output_names.reserve(output_count);
    for (std::uint32_t i = 0; i < output_count; ++i) {
        metadata.output_names.push_back(reader.string());
    }
    metadata.plugin_library = reader.string();
    metadata.plugin_module = reader.string();
    metadata.plugin_first_op = reader.string();
    if (reader.offset != reader.size) throw std::runtime_error("Denglin cache metadata has trailing data");
    return metadata;
}

struct CacheEnvelope {
    CacheMetadata metadata;
    std::vector<std::uint8_t> payload;
};

bool has_cache_magic(const std::vector<std::uint8_t>& data) {
    constexpr auto magic_size = sizeof(kDenglinCacheMagic) - 1;
    return data.size() >= magic_size &&
           std::memcmp(data.data(), kDenglinCacheMagic, magic_size) == 0;
}

std::vector<std::uint8_t> make_cache_blob(const CacheMetadata& metadata,
                                          const std::vector<std::uint8_t>& payload) {
    ByteWriter metadata_writer;
    write_cache_metadata(metadata_writer, metadata);
    if (metadata_writer.bytes.size() > kMaxCacheMetadataBytes) {
        throw std::overflow_error("Denglin cache metadata is too large");
    }

    ByteWriter writer;
    writer.raw(kDenglinCacheMagic, sizeof(kDenglinCacheMagic) - 1);
    writer.u32(kDenglinCacheVersion);
    writer.u64(static_cast<std::uint64_t>(metadata_writer.bytes.size()));
    writer.u64(static_cast<std::uint64_t>(payload.size()));
    writer.u64(fnv1a(metadata_writer.bytes.data(), metadata_writer.bytes.size()));
    writer.u64(fnv1a(payload.data(), payload.size()));
    writer.raw(metadata_writer.bytes.data(), metadata_writer.bytes.size());
    writer.raw(payload.data(), payload.size());
    return std::move(writer.bytes);
}

CacheEnvelope parse_cache_blob(const std::vector<std::uint8_t>& data) {
    constexpr auto magic_size = sizeof(kDenglinCacheMagic) - 1;
    if (!has_cache_magic(data) || data.size() < magic_size + 4U + 8U * 4U) {
        throw std::runtime_error("Denglin cache magic is invalid");
    }
    ByteReader reader(data);
    const auto magic = reader.bytes(magic_size);
    if (std::memcmp(magic.data(), kDenglinCacheMagic, magic_size) != 0) {
        throw std::runtime_error("Denglin cache magic is invalid");
    }
    if (reader.u32() != kDenglinCacheVersion) {
        throw std::runtime_error("Denglin cache version is unsupported");
    }
    const auto metadata_size = reader.u64();
    const auto payload_size = reader.u64();
    const auto metadata_hash = reader.u64();
    const auto payload_hash = reader.u64();
    if (metadata_size > kMaxCacheMetadataBytes ||
        metadata_size > reader.size - reader.offset ||
        payload_size > reader.size - reader.offset - static_cast<std::size_t>(metadata_size) ||
        reader.size - reader.offset != static_cast<std::size_t>(metadata_size + payload_size)) {
        throw std::runtime_error("Denglin cache sizes are invalid");
    }
    const auto metadata_bytes = reader.bytes(static_cast<std::size_t>(metadata_size));
    if (fnv1a(metadata_bytes.data(), metadata_bytes.size()) != metadata_hash) {
        throw std::runtime_error("Denglin cache metadata checksum mismatch");
    }
    CacheEnvelope envelope;
    envelope.metadata = read_cache_metadata(metadata_bytes);
    envelope.payload = reader.bytes(static_cast<std::size_t>(payload_size));
    if (fnv1a(envelope.payload.data(), envelope.payload.size()) != payload_hash) {
        throw std::runtime_error("Denglin cache payload checksum mismatch");
    }
    return envelope;
}

void write_file_atomic(const std::string& path, const std::vector<std::uint8_t>& data) {
    if (path.empty()) throw std::invalid_argument("Denglin output path is empty");
    const std::filesystem::path target(path);
    const auto parent = target.parent_path();
    std::error_code error;
    if (!parent.empty()) std::filesystem::create_directories(parent, error);
    if (error) throw std::runtime_error("Unable to create Denglin cache directory: " + error.message());

    auto temporary = target;
    std::ostringstream temporary_suffix;
    temporary_suffix << ".tmp."
                     << std::chrono::high_resolution_clock::now().time_since_epoch().count()
                     << '.' << std::hash<std::thread::id>{}(std::this_thread::get_id());
    temporary += temporary_suffix.str();
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("Unable to open Denglin output: " + temporary.string());
        if (!data.empty()) {
            file.write(reinterpret_cast<const char*>(data.data()),
                       static_cast<std::streamsize>(data.size()));
        }
        file.flush();
        if (!file) throw std::runtime_error("Unable to write Denglin output: " + temporary.string());
    }

    std::filesystem::remove(target, error);
    error.clear();
    std::filesystem::rename(temporary, target, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw std::runtime_error("Unable to replace Denglin output: " + path);
    }
}

bool same_path(const std::string& left, const std::string& right) {
    std::error_code error_left;
    std::error_code error_right;
    const auto left_absolute = std::filesystem::absolute(left, error_left);
    const auto right_absolute = std::filesystem::absolute(right, error_right);
    if (error_left || error_right) return left == right;
    return left_absolute.lexically_normal() == right_absolute.lexically_normal();
}

CacheMetadata make_cache_metadata(const DenglinSessionOptions& options,
                                  const std::string& model_format,
                                  const std::string& model_hash,
                                  const dl::nne::Engine* engine) {
    CacheMetadata metadata;
    metadata.sdk_version = sdk_version();
    metadata.device = device_fingerprint();
    metadata.model_format = model_format;
    metadata.model_hash = model_hash;
    metadata.input_name = options.input_name;
    metadata.input_shape = options.input_shape;
    metadata.optimization_profiles = options.optimization_profiles;
    metadata.max_batch_size = options.max_batch_size;
    metadata.batch_size = options.batch_size;
    metadata.weight_share = static_cast<int>(options.weight_share);
    metadata.cluster = static_cast<int>(options.cluster);
    metadata.builder_flags = options.builder_flags;
    metadata.input_layout = static_cast<int>(options.input_layout);
    metadata.output_names = options.output_names;
    metadata.plugin_library = options.plugin_library;
    metadata.plugin_module = options.plugin_module;
    metadata.plugin_first_op = options.plugin_first_op;
    if (engine) {
        int input_index = options.input_name.empty() ? -1 : engine->GetBindingIndex(options.input_name.c_str());
        if (input_index < 0) {
            for (int index = 0; index < engine->GetNbBindings(); ++index) {
                if (engine->BindingIsInput(index)) {
                    input_index = index;
                    break;
                }
            }
        }
        if (input_index >= 0) metadata.input_type = to_data_type(engine->GetBindingDataType(input_index));
        if (metadata.output_names.empty()) {
            for (int index = 0; index < engine->GetNbBindings(); ++index) {
                if (!engine->BindingIsInput(index)) {
                    const auto* name = engine->GetBindingName(index);
                    if (name && *name) metadata.output_names.emplace_back(name);
                }
            }
        }
    }
    return metadata;
}

bool cache_metadata_matches(const CacheMetadata& actual,
                            const CacheMetadata& expected) {
    const auto compatible_runtime_value = [](const std::string& left,
                                              const std::string& right) {
        return left == right || left == "unknown" || right == "unknown";
    };
    if (!compatible_runtime_value(actual.sdk_version, expected.sdk_version) ||
        !compatible_runtime_value(actual.device, expected.device) ||
        actual.model_format != expected.model_format ||
        actual.model_hash != expected.model_hash ||
        actual.input_name != expected.input_name ||
        actual.input_shape != expected.input_shape ||
        !same_profiles(actual.optimization_profiles, expected.optimization_profiles) ||
        actual.max_batch_size != expected.max_batch_size ||
        actual.batch_size != expected.batch_size ||
        actual.weight_share != expected.weight_share ||
        actual.cluster != expected.cluster ||
        actual.builder_flags != expected.builder_flags ||
        actual.input_layout != expected.input_layout ||
        actual.plugin_library != expected.plugin_library ||
        actual.plugin_module != expected.plugin_module ||
        actual.plugin_first_op != expected.plugin_first_op) {
        return false;
    }
    if (!expected.output_names.empty() && actual.output_names != expected.output_names) return false;
    return true;
}

std::vector<std::uint8_t> serialize_engine_bytes(const dl::nne::Engine& engine) {
    HostMemoryPtr memory(engine.Serialize());
    if (!memory || !memory->Data() || memory->Size() == 0) {
        throw std::runtime_error("Denglin engine serialization failed");
    }
    const auto* data = static_cast<const std::uint8_t*>(memory->Data());
    return std::vector<std::uint8_t>(data, data + memory->Size());
}

bool try_load_cache(const std::string& path,
                    const CacheMetadata& expected,
                    EnginePtr& engine,
                    std::string& reason) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
        reason = error ? error.message() : "cache file does not exist";
        return false;
    }
    try {
        const auto bytes = read_binary_file(path);
        if (!has_cache_magic(bytes)) {
            reason = "cache file has no cvinfer metadata envelope";
            return false;
        }
        const auto envelope = parse_cache_blob(bytes);
        if (!cache_metadata_matches(envelope.metadata, expected)) {
            reason = "cache metadata does not match the requested model or runtime";
            return false;
        }
        EnginePtr candidate(dl::nne::Deserialize(envelope.payload.data(), envelope.payload.size()));
        if (!candidate) {
            reason = "Denglin rejected the cached engine payload";
            return false;
        }
        const auto input_index = candidate->GetBindingIndex(
            envelope.metadata.input_name.c_str());
        if (input_index < 0 || !candidate->BindingIsInput(input_index)) {
            reason = "cached engine input binding does not match cache metadata";
            return false;
        }
        if (envelope.metadata.input_type != TensorDataType::Unknown &&
            to_data_type(candidate->GetBindingDataType(input_index)) !=
                envelope.metadata.input_type) {
            reason = "cached engine input dtype does not match cache metadata";
            return false;
        }
        if (!envelope.metadata.optimization_profiles.empty()) {
            if (candidate->GetNbOptimizationProfiles() !=
                static_cast<int>(envelope.metadata.optimization_profiles.size())) {
                reason = "cached engine optimization profiles do not match cache metadata";
                return false;
            }
            for (std::size_t profile_index = 0;
                 profile_index < envelope.metadata.optimization_profiles.size();
                 ++profile_index) {
                const auto& expected_profile = envelope.metadata.optimization_profiles[profile_index];
                if (to_shape(candidate->GetProfileDimensions(
                        input_index, static_cast<int32_t>(profile_index),
                        dl::nne::kOptProfileMin)) != expected_profile.min_shape ||
                    to_shape(candidate->GetProfileDimensions(
                        input_index, static_cast<int32_t>(profile_index),
                        dl::nne::kOptProfileOpt)) != expected_profile.opt_shape ||
                    to_shape(candidate->GetProfileDimensions(
                        input_index, static_cast<int32_t>(profile_index),
                        dl::nne::kOptProfileMax)) != expected_profile.max_shape) {
                    reason = "cached engine optimization profile shapes do not match cache metadata";
                    return false;
                }
            }
        }
        engine = std::move(candidate);
        return true;
    } catch (const std::exception& error_value) {
        reason = error_value.what();
        return false;
    }
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return value;
}

DenglinModelFormat resolve_model_format(const std::string& path,
                                        DenglinModelFormat requested) {
    if (requested != DenglinModelFormat::Auto) return requested;
    const auto lower = lowercase(path);
    if (lower.size() >= 5 && lower.compare(lower.size() - 5, 5, ".onnx") == 0) {
        return DenglinModelFormat::Onnx;
    }
    return DenglinModelFormat::SerializedEngine;
}

dl::nne::Format to_sdk_format(DenglinInputLayout layout) {
    switch (layout) {
    case DenglinInputLayout::Linear: return dl::nne::kLINEAR;
    case DenglinInputLayout::NHWC: return dl::nne::kNHWC;
    default: return dl::nne::kNCHW;
    }
}

dl::nne::WeightShareMode to_sdk_weight_share(DenglinWeightShareMode mode) {
    switch (mode) {
    case DenglinWeightShareMode::Share2: return dl::nne::kShare2;
    case DenglinWeightShareMode::Share4: return dl::nne::kShare4;
    default: return dl::nne::kSingle;
    }
}

dl::nne::ClusterConfig to_sdk_cluster(DenglinClusterConfig cluster) {
    switch (cluster) {
    case DenglinClusterConfig::Cluster1: return dl::nne::kCluster1;
    case DenglinClusterConfig::Cluster2: return dl::nne::kCluster2;
    case DenglinClusterConfig::Cluster3: return dl::nne::kCluster3;
    case DenglinClusterConfig::Cluster01: return dl::nne::kCluster01;
    case DenglinClusterConfig::Cluster23: return dl::nne::kCluster23;
    case DenglinClusterConfig::Cluster02: return dl::nne::kCluster02;
    case DenglinClusterConfig::Cluster13: return dl::nne::kCluster13;
    case DenglinClusterConfig::Cluster03: return dl::nne::kCluster03;
    case DenglinClusterConfig::Cluster12: return dl::nne::kCluster12;
    case DenglinClusterConfig::Cluster0123: return dl::nne::kCluster0123;
    default: return dl::nne::kCluster0;
    }
}

} // namespace

struct DenglinSession::Impl {
    DenglinSessionOptions options;
    EnginePtr engine;
    ContextPtr context;
    cudaStream_t stream{};

    std::vector<std::string> binding_names;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::unordered_map<std::string, int> binding_indices;

    std::vector<void*> bindings;
    std::vector<void*> owned_device_buffers;
    std::vector<std::size_t> buffer_bytes;

    std::unordered_map<std::string, std::vector<std::uint8_t>> host_inputs;
    std::unordered_map<std::string, const void*> device_inputs;
    std::unordered_map<std::string, std::size_t> device_input_bytes;
    std::unordered_map<std::string, void*> device_outputs;
    std::unordered_map<std::string, std::size_t> device_output_bytes;
    std::unordered_map<std::string, Tensor> outputs;
    std::unordered_map<std::string, std::vector<int64_t>> output_shapes;
    bool use_tensor_api{};
    bool output_host_copy{true};
    bool cache_hit{};
    bool pending{};

    ~Impl() {
        if (stream) cudaStreamSynchronize(stream);
        pending = false;
        context.reset();
        for (void* buffer : owned_device_buffers) {
            if (buffer) cudaFree(buffer);
        }
        if (stream) cudaStreamDestroy(stream);
        engine.reset();
    }

    int binding(const std::string& name) const {
        const auto it = binding_indices.find(name);
        if (it == binding_indices.end()) {
            throw std::invalid_argument("Unknown Denglin tensor: " + name);
        }
        return it->second;
    }

    void ensure_idle(const char* operation) const {
        if (pending) {
            throw std::runtime_error(std::string("Denglin ") + operation +
                                     " is unavailable while async inference is pending");
        }
    }

    dl::nne::Dims binding_dimensions(int index) const {
        if (use_tensor_api && index >= 0 &&
            index < static_cast<int>(binding_names.size())) {
            const auto tensor_dims = context->GetTensorShape(
                binding_names[static_cast<std::size_t>(index)].c_str());
            if (!has_unresolved_dimension(tensor_dims)) return tensor_dims;
        }
        auto dims = context->GetBindingDimensions(index);
        if (!has_unresolved_dimension(dims)) return dims;

        const auto engine_dims = engine->GetBindingDimensions(index);
        if (!has_unresolved_dimension(engine_dims)) return engine_dims;

        if (engine->GetNbOptimizationProfiles() > 0) {
            const auto active_profile = context->GetOptimizationProfile();
            const int profile = active_profile < 0 ? 0 : static_cast<int>(active_profile);
            const auto profile_dims = engine->GetProfileDimensions(
                index, profile, dl::nne::kOptProfileOpt);
            if (!has_unresolved_dimension(profile_dims)) return profile_dims;
        }
        return dims;
    }

    dl::nne::Dims allocation_dimensions(int index) const {
        auto dims = binding_dimensions(index);
        if (!has_unresolved_dimension(dims)) return dims;
        if (engine->GetNbOptimizationProfiles() > 0) {
            const auto active_profile = context->GetOptimizationProfile();
            const int profile = active_profile < 0 ? 0 : static_cast<int>(active_profile);
            const auto profile_dims = engine->GetProfileDimensions(
                index, profile, dl::nne::kOptProfileMax);
            if (!has_unresolved_dimension(profile_dims)) return profile_dims;
        }
        throw std::runtime_error("Denglin tensor shape is unresolved: " +
                                 binding_names[static_cast<std::size_t>(index)]);
    }

    std::size_t binding_bytes(int index, const dl::nne::Dims& dims) const {
        auto bytes = tensor_bytes(dims, engine->GetBindingDataType(index));
        // The legacy Enqueue API carries batch_size separately for implicit
        // batch engines. Match the SDK samples by accounting for that batch
        // multiplier when sizing input and output bindings.
        if (engine->HasImplicitBacthDimension()) {
            const auto batch = static_cast<std::size_t>(options.batch_size);
            if (batch > std::numeric_limits<std::size_t>::max() / bytes) {
                throw std::overflow_error("Denglin batch byte size overflow");
            }
            bytes *= batch;
        }
        return bytes;
    }

    void ensure_device_buffer(int index, std::size_t bytes) {
        if (bytes == 0) throw std::runtime_error("Denglin tensor has zero bytes");
        auto& owned = owned_device_buffers[static_cast<std::size_t>(index)];
        auto& capacity = buffer_bytes[static_cast<std::size_t>(index)];
        if (owned && capacity >= bytes) return;
        if (owned) check_cuda(cudaFree(owned), "cudaFree(Denglin buffer)");
        owned = nullptr;
        check_cuda(cudaMalloc(&owned, bytes), "cudaMalloc(Denglin buffer)");
        capacity = bytes;
    }

    void set_tensor_address(const std::string& name, void* address) {
        if (!use_tensor_api) return;
        if (!context->SetTensorAddress(name.c_str(), address)) {
            throw std::runtime_error("Denglin SetTensorAddress failed: " + name);
        }
    }
};

DenglinSession::DenglinSession() = default;
DenglinSession::~DenglinSession() = default;
DenglinSession::DenglinSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DenglinSession::DenglinSession(DenglinSession&&) noexcept = default;
DenglinSession& DenglinSession::operator=(DenglinSession&&) noexcept = default;

DenglinSession DenglinSession::load(const std::string& model_path,
                                    DenglinSessionOptions options) {
    if (model_path.empty()) throw std::invalid_argument("Denglin model path is empty");
    if (options.max_batch_size <= 0 || options.batch_size <= 0 ||
        options.batch_size > options.max_batch_size) {
        throw std::invalid_argument("Denglin batch size is invalid");
    }
    if (options.optimization_profile < -1) {
        throw std::invalid_argument("Denglin optimization profile index is invalid");
    }
    if (options.cache_mode != DenglinCacheMode::Disabled &&
        options.engine_cache_path.empty()) {
        throw std::invalid_argument(
            "Denglin cache mode requires engine_cache_path");
    }

    check_driver_init();
    // Establish the CUDA runtime primary context on the calling thread.
    check_cuda(cudaSetDevice(0), "cudaSetDevice(Denglin)");
    auto impl = std::make_unique<Impl>();
    impl->options = std::move(options);
    const auto format = resolve_model_format(model_path, impl->options.model_format);
    if (format == DenglinModelFormat::Onnx &&
        impl->options.cache_mode != DenglinCacheMode::Disabled &&
        same_path(model_path, impl->options.engine_cache_path)) {
        throw std::invalid_argument(
            "Denglin engine_cache_path must differ from the ONNX model path");
    }
    bool built_from_onnx = false;
    std::string model_hash;

    if (format == DenglinModelFormat::Onnx) {
        if (impl->options.input_name.empty() || impl->options.input_shape.empty()) {
            throw std::invalid_argument(
                "Denglin ONNX loading requires input_name and input_shape");
        }
        (void)to_dims(impl->options.input_shape, true);
        validate_profiles(impl->options);
        for (const auto& output_name : impl->options.output_names) {
            if (output_name.empty()) {
                throw std::invalid_argument("Denglin output name is empty");
            }
        }
        if (impl->options.plugin_library.empty() !=
            impl->options.plugin_module.empty()) {
            throw std::invalid_argument(
                "Denglin plugin_library and plugin_module must be provided together");
        }

        if (impl->options.cache_mode != DenglinCacheMode::Disabled) {
            model_hash = file_hash(model_path);
            const auto expected_metadata = make_cache_metadata(
                impl->options, "onnx", model_hash, nullptr);
            std::string cache_reason;
            if (try_load_cache(impl->options.engine_cache_path,
                               expected_metadata,
                               impl->engine,
                               cache_reason)) {
                impl->cache_hit = true;
            } else if (!impl->options.build_on_cache_miss) {
                throw std::runtime_error(
                    "Denglin engine cache miss and build_on_cache_miss is disabled: " +
                    cache_reason);
            }
        }

        if (!impl->engine) {
            BuilderPtr builder(dl::nne::CreateInferBuilder());
            if (!builder) throw std::runtime_error("Denglin builder creation failed");

            const bool dynamic_input = std::any_of(
                impl->options.input_shape.begin(), impl->options.input_shape.end(),
                [](std::int64_t dimension) { return dimension < 0; });
            NetworkPtr network(dynamic_input ||
                                   !impl->options.optimization_profiles.empty()
                               ? builder->CreateNetworkV2(dl::nne::kEXPLICIT_BATCH)
                               : builder->CreateNetwork());
            if (!network) throw std::runtime_error("Denglin network creation failed");

            dl::nne::BuilderConfig builder_config{};
            builder_config.max_batch_size = impl->options.max_batch_size;
            builder_config.ws_mode = to_sdk_weight_share(impl->options.weight_share);
            builder_config.flags = impl->options.builder_flags;
            builder->SetBuilderConfig(builder_config);
            if (impl->options.builder_flags != 0) {
                builder->SetFlags(impl->options.builder_flags);
            }

            ParserPtr parser(dl::nne::CreateParser());
            if (!parser) throw std::runtime_error("Denglin parser creation failed");

            if (!impl->options.plugin_library.empty() ||
                !impl->options.plugin_module.empty()) {
                if (impl->options.plugin_library.empty() ||
                    impl->options.plugin_module.empty()) {
                    throw std::invalid_argument(
                        "Denglin plugin_library and plugin_module must be provided together");
                }
                const char* first_op = impl->options.plugin_first_op.empty()
                    ? nullptr : impl->options.plugin_first_op.c_str();
                if (!parser->RegisterUserOp(impl->options.plugin_library.c_str(),
                                            impl->options.plugin_module.c_str(),
                                            first_op)) {
                    throw std::runtime_error("Denglin user op registration failed");
                }
            }

            const auto input_dims = to_dims(impl->options.input_shape, true);
            if (!parser->RegisterInput(impl->options.input_name.c_str(),
                                       input_dims,
                                       to_sdk_format(impl->options.input_layout))) {
                throw std::runtime_error("Denglin input registration failed: " +
                                         impl->options.input_name);
            }
            for (const auto& output_name : impl->options.output_names) {
                if (output_name.empty() || !parser->RegisterOutput(output_name.c_str())) {
                    throw std::runtime_error("Denglin output registration failed: " +
                                             output_name);
                }
            }
            if (!parser->Parse(model_path.c_str(), *network)) {
                throw std::runtime_error("Denglin ONNX parse failed: " + model_path);
            }

            for (const auto& profile_options : impl->options.optimization_profiles) {
                // OptimizationProfile has a protected destructor and no
                // Destroy() entry in this SDK. Ownership is transferred to
                // Network::AddOptimizationProfile().
                auto* profile = builder->CreateOptimizationProfile();
                if (!profile) {
                    throw std::runtime_error(
                        "Denglin optimization profile creation failed");
                }
                if (!profile->SetDimensions(
                        impl->options.input_name.c_str(), dl::nne::kOptProfileMin,
                        dl::nne::kInput, to_dims(profile_options.min_shape)) ||
                    !profile->SetDimensions(
                        impl->options.input_name.c_str(), dl::nne::kOptProfileOpt,
                        dl::nne::kInput, to_dims(profile_options.opt_shape)) ||
                    !profile->SetDimensions(
                        impl->options.input_name.c_str(), dl::nne::kOptProfileMax,
                        dl::nne::kInput, to_dims(profile_options.max_shape)) ||
                    !profile->IsValid()) {
                    throw std::runtime_error(
                        "Denglin optimization profile configuration failed");
                }
                if (network->AddOptimizationProfile(profile) < 0) {
                    throw std::runtime_error(
                        "Denglin optimization profile registration failed");
                }
            }

            impl->engine.reset(builder->BuildEngine(*network));
            if (!impl->engine) throw std::runtime_error("Denglin engine build failed");
            built_from_onnx = true;
        }
    } else if (format == DenglinModelFormat::SerializedEngine) {
        if (impl->options.cache_mode != DenglinCacheMode::Disabled) {
            throw std::invalid_argument(
                "Denglin engine cache mode is only valid when loading ONNX");
        }
        const auto serialized = read_binary_file(model_path);
        std::vector<std::uint8_t> payload = serialized;
        if (has_cache_magic(serialized)) {
            const auto envelope = parse_cache_blob(serialized);
            const auto runtime_sdk = sdk_version();
            const auto runtime_device = device_fingerprint();
            if ((envelope.metadata.sdk_version != runtime_sdk &&
                 envelope.metadata.sdk_version != "unknown" && runtime_sdk != "unknown") ||
                (envelope.metadata.device != runtime_device &&
                 envelope.metadata.device != "unknown" && runtime_device != "unknown")) {
                throw std::runtime_error(
                    "Denglin cache runtime metadata does not match the current SDK/device");
            }
            if (envelope.metadata.cluster != static_cast<int>(impl->options.cluster) ||
                envelope.metadata.batch_size != impl->options.batch_size ||
                envelope.metadata.max_batch_size != impl->options.max_batch_size ||
                envelope.metadata.weight_share != static_cast<int>(impl->options.weight_share) ||
                envelope.metadata.builder_flags != impl->options.builder_flags ||
                envelope.metadata.input_layout != static_cast<int>(impl->options.input_layout) ||
                (impl->options.input_name.size() > 0 &&
                 envelope.metadata.input_name != impl->options.input_name) ||
                (!impl->options.input_shape.empty() &&
                 envelope.metadata.input_shape != impl->options.input_shape) ||
                (!impl->options.input_shape.empty() &&
                 !same_profiles(envelope.metadata.optimization_profiles,
                                impl->options.optimization_profiles)) ||
                (!impl->options.output_names.empty() &&
                 envelope.metadata.output_names != impl->options.output_names) ||
                envelope.metadata.plugin_library != impl->options.plugin_library ||
                envelope.metadata.plugin_module != impl->options.plugin_module ||
                envelope.metadata.plugin_first_op != impl->options.plugin_first_op) {
                throw std::runtime_error(
                    "Denglin cache engine metadata does not match the requested configuration");
            }
            payload = envelope.payload;
            impl->cache_hit = true;
        }
        impl->engine.reset(dl::nne::Deserialize(payload.data(), payload.size()));
        if (!impl->engine) {
            throw std::runtime_error("Denglin engine deserialization failed: " + model_path);
        }
    } else {
        throw std::invalid_argument("Unsupported Denglin model format");
    }

    impl->context.reset(impl->engine->CreateExecutionContext(
        to_sdk_cluster(impl->options.cluster)));
    if (!impl->context) throw std::runtime_error("Denglin execution context creation failed");
    if (impl->options.optimization_profile >= 0) {
        if (impl->options.optimization_profile >=
            impl->engine->GetNbOptimizationProfiles()) {
            throw std::invalid_argument("Denglin optimization profile index is out of range");
        }
        if (!impl->context->SetOptimizationProfile(
                impl->options.optimization_profile)) {
            throw std::runtime_error("Denglin optimization profile selection failed");
        }
    }
    check_cuda(cudaStreamCreate(&impl->stream), "cudaStreamCreate(Denglin)");

    const int binding_count = impl->engine->GetNbBindings();
    if (binding_count <= 0) throw std::runtime_error("Denglin engine has no bindings");
    impl->binding_names.resize(static_cast<std::size_t>(binding_count));
    impl->bindings.resize(static_cast<std::size_t>(binding_count), nullptr);
    impl->owned_device_buffers.resize(static_cast<std::size_t>(binding_count), nullptr);
    impl->buffer_bytes.resize(static_cast<std::size_t>(binding_count), 0);

    for (int index = 0; index < binding_count; ++index) {
        const char* name = impl->engine->GetBindingName(index);
        if (!name || !*name) throw std::runtime_error("Denglin binding has no name");
        impl->binding_names[static_cast<std::size_t>(index)] = name;
        if (impl->binding_indices.emplace(name, index).second) {
            if (impl->engine->BindingIsInput(index)) {
                impl->input_names.emplace_back(name);
            } else {
                impl->output_names.emplace_back(name);
            }
        }
    }

    const bool tensor_api_available = impl->engine->GetNbIOTensors() > 0;
    if (impl->options.execution_mode == DenglinExecutionMode::TensorApi &&
        !tensor_api_available) {
        throw std::runtime_error(
            "Denglin TensorApi execution was requested but the engine exposes no IO tensors");
    }
    // Auto prefers the modern tensor-address API. If an older serialized engine
    // exposes no IO tensors, it transparently uses the legacy binding API.
    // TensorApi remains strict and requires EnqueueV3 support.
    impl->use_tensor_api = tensor_api_available &&
        impl->options.execution_mode != DenglinExecutionMode::Bindings;

    if (built_from_onnx &&
        impl->options.cache_mode == DenglinCacheMode::ReadWrite) {
        const auto payload = serialize_engine_bytes(*impl->engine);
        const auto metadata = make_cache_metadata(
            impl->options, "onnx", model_hash, impl->engine.get());
        write_file_atomic(impl->options.engine_cache_path,
                          make_cache_blob(metadata, payload));
    }

    return DenglinSession(std::move(impl));
}

void DenglinSession::serialize_engine(const std::string& engine_path) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("serialize_engine");
    write_file_atomic(engine_path, serialize_engine_bytes(*impl_->engine));
}

bool DenglinSession::cache_hit() const noexcept {
    return impl_ && impl_->cache_hit;
}

std::vector<std::string> DenglinSession::input_names() const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    return impl_->input_names;
}

std::vector<std::string> DenglinSession::output_names() const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    return impl_->output_names;
}

std::vector<int64_t> DenglinSession::tensor_shape(const std::string& name) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    return to_shape(impl_->binding_dimensions(impl_->binding(name)));
}

void DenglinSession::set_shape(const std::string& name,
                               const std::vector<int64_t>& shape) {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("set_shape");
    const int index = impl_->binding(name);
    if (!impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Only Denglin input tensors can have shapes set");
    }
    const auto dims = to_dims(shape);
    const auto engine_dims = impl_->engine->GetBindingDimensions(index);
    if (!has_unresolved_dimension(engine_dims) && to_shape(engine_dims) == shape) {
        impl_->outputs.clear();
        impl_->output_shapes.clear();
        return;
    }
    bool accepted = false;
    if (impl_->use_tensor_api) {
        accepted = impl_->context->SetInputShape(name.c_str(), dims);
    }
    if (!accepted) {
        accepted = impl_->context->SetBindingDimensions(index, dims);
    }
    if (!accepted && !impl_->use_tensor_api) {
        accepted = impl_->context->SetInputShape(name.c_str(), dims);
    }
    if (!accepted) {
        throw std::runtime_error("Denglin rejected input shape: " + name);
    }
    impl_->outputs.clear();
    impl_->output_shapes.clear();
}

void DenglinSession::set_optimization_profile(int profile_index) {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("set_optimization_profile");
    if (profile_index < 0 ||
        profile_index >= impl_->engine->GetNbOptimizationProfiles()) {
        throw std::invalid_argument("Denglin optimization profile index is out of range");
    }
    if (!impl_->context->SetOptimizationProfile(profile_index)) {
        throw std::runtime_error("Denglin optimization profile selection failed");
    }
    impl_->outputs.clear();
    impl_->output_shapes.clear();
}

int DenglinSession::optimization_profile() const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    return impl_->context->GetOptimizationProfile();
}

int DenglinSession::optimization_profile_count() const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    return impl_->engine->GetNbOptimizationProfiles();
}

DenglinOptimizationProfile DenglinSession::optimization_profile_shapes(
    int profile_index) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    if (profile_index < 0 ||
        profile_index >= impl_->engine->GetNbOptimizationProfiles()) {
        throw std::invalid_argument("Denglin optimization profile index is out of range");
    }
    if (impl_->input_names.empty()) {
        throw std::runtime_error("Denglin engine has no input");
    }
    const int index = impl_->binding(impl_->input_names.front());
    DenglinOptimizationProfile result;
    result.min_shape = to_shape(impl_->engine->GetProfileDimensions(
        index, profile_index, dl::nne::kOptProfileMin));
    result.opt_shape = to_shape(impl_->engine->GetProfileDimensions(
        index, profile_index, dl::nne::kOptProfileOpt));
    result.max_shape = to_shape(impl_->engine->GetProfileDimensions(
        index, profile_index, dl::nne::kOptProfileMax));
    return result;
}

void DenglinSession::set_input(const std::string& name,
                               const void* data,
                               std::size_t bytes) {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("set_input");
    if (!data && bytes != 0) throw std::invalid_argument("Denglin input data is null");
    const int index = impl_->binding(name);
    if (!impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Denglin tensor is not an input: " + name);
    }
    auto& input = impl_->host_inputs[name];
    input.resize(bytes);
    if (bytes > 0) std::memcpy(input.data(), data, bytes);
    impl_->device_inputs.erase(name);
    impl_->device_input_bytes.erase(name);
    (void)index;
}

void DenglinSession::set_input_device(const std::string& name,
                                      const void* device_data,
                                      std::size_t bytes) {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("set_input_device");
    if (!device_data && bytes != 0) {
        throw std::invalid_argument("Denglin device input data is null");
    }
    const int index = impl_->binding(name);
    if (!impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Denglin tensor is not an input: " + name);
    }
    impl_->device_inputs[name] = device_data;
    impl_->device_input_bytes[name] = bytes;
    impl_->host_inputs.erase(name);
    (void)index;
}

void DenglinSession::set_output_device(const std::string& name,
                                       void* device_data,
                                       std::size_t bytes) {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("set_output_device");
    if (!device_data && bytes != 0) {
        throw std::invalid_argument("Denglin device output data is null");
    }
    const int index = impl_->binding(name);
    if (impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Denglin tensor is not an output: " + name);
    }
    if (!device_data || bytes == 0) {
        impl_->device_outputs.erase(name);
        impl_->device_output_bytes.erase(name);
        return;
    }
    impl_->device_outputs[name] = device_data;
    impl_->device_output_bytes[name] = bytes;
}

void DenglinSession::set_output_host_copy(bool enabled) {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("set_output_host_copy");
    impl_->output_host_copy = enabled;
}

void DenglinSession::infer_async() {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("infer_async");
    if (impl_->input_names.empty()) throw std::runtime_error("Denglin engine has no input");

    impl_->outputs.clear();
    impl_->output_shapes.clear();
    try {
        const auto process_input = [&](const std::string& name) {
            const int index = impl_->binding(name);
            auto context_dims = impl_->context->GetBindingDimensions(index);
            if (impl_->use_tensor_api) {
                context_dims = impl_->context->GetTensorShape(name.c_str());
            }
            if (has_unresolved_dimension(context_dims)) {
                throw std::runtime_error(
                    "Denglin dynamic input dimensions were not specified: " + name);
            }
            const auto dims = impl_->binding_dimensions(index);
            const auto bytes = impl_->binding_bytes(index, dims);
            const auto device_input = impl_->device_inputs.find(name);
            void* address = nullptr;
            if (device_input != impl_->device_inputs.end()) {
                const auto size = impl_->device_input_bytes.find(name);
                if (size == impl_->device_input_bytes.end() || size->second != bytes) {
                    throw std::runtime_error(
                        "Denglin device input byte size mismatch: " + name);
                }
                address = const_cast<void*>(device_input->second);
            } else {
                const auto input = impl_->host_inputs.find(name);
                if (input == impl_->host_inputs.end() || input->second.size() != bytes) {
                    throw std::runtime_error("Denglin input byte size mismatch: " + name);
                }
                impl_->ensure_device_buffer(index, bytes);
                address = impl_->owned_device_buffers[static_cast<std::size_t>(index)];
                check_cuda(cudaMemcpyAsync(address, input->second.data(), bytes,
                                           cudaMemcpyHostToDevice, impl_->stream),
                           "cudaMemcpyAsync(Denglin H2D)");
            }
            impl_->bindings[static_cast<std::size_t>(index)] = address;
            impl_->set_tensor_address(name, address);
        };

        for (const auto& name : impl_->input_names) process_input(name);

        const auto prepare_output = [&](const std::string& name) {
            const int index = impl_->binding(name);
            const auto dims = impl_->allocation_dimensions(index);
            const auto bytes = impl_->binding_bytes(index, dims);
            void* address = nullptr;
            const auto external = impl_->device_outputs.find(name);
            if (external != impl_->device_outputs.end()) {
                const auto size = impl_->device_output_bytes.find(name);
                if (size == impl_->device_output_bytes.end() || size->second < bytes) {
                    throw std::runtime_error(
                        "Denglin device output buffer is too small: " + name);
                }
                address = external->second;
            } else {
                impl_->ensure_device_buffer(index, bytes);
                address = impl_->owned_device_buffers[static_cast<std::size_t>(index)];
            }
            impl_->bindings[static_cast<std::size_t>(index)] = address;
            impl_->set_tensor_address(name, address);
        };

        for (const auto& name : impl_->output_names) prepare_output(name);

        bool enqueued = false;
        const auto stream = reinterpret_cast<void*>(impl_->stream);
        if (impl_->use_tensor_api) {
            enqueued = impl_->context->EnqueueV3(stream);
            if (!enqueued &&
                impl_->options.execution_mode == DenglinExecutionMode::TensorApi) {
                throw std::runtime_error(
                    "Denglin EnqueueV3 failed for TensorApi execution mode");
            }
        }
        if (!enqueued && impl_->engine->HasImplicitBacthDimension()) {
            enqueued = impl_->context->Enqueue(
                impl_->options.batch_size, impl_->bindings.data(), stream, nullptr);
        } else if (!enqueued) {
            enqueued = impl_->context->EnqueueV2(
                impl_->bindings.data(), stream, nullptr);
        }
        if (!enqueued) throw std::runtime_error("Denglin inference execution failed");

        for (const auto& name : impl_->output_names) {
            const int index = impl_->binding(name);
            auto dims = impl_->context->GetBindingDimensions(index);
            if (impl_->use_tensor_api) {
                dims = impl_->context->GetTensorShape(name.c_str());
            }
            if (has_unresolved_dimension(dims)) dims = impl_->allocation_dimensions(index);
            const auto type = impl_->engine->GetBindingDataType(index);
            const auto bytes = impl_->binding_bytes(index, dims);
            const auto external = impl_->device_outputs.find(name);
            if (external == impl_->device_outputs.end() &&
                impl_->buffer_bytes[static_cast<std::size_t>(index)] < bytes) {
                throw std::runtime_error(
                    "Denglin output buffer is smaller than resolved output: " + name);
            }
            impl_->output_shapes[name] = to_shape(dims);

            if (!impl_->output_host_copy) continue;
            Tensor output;
            output.name = name;
            output.shape = to_shape(dims);
            output.data_type = to_data_type(type);
            output.data.resize(bytes);
            auto [it, inserted] = impl_->outputs.emplace(name, std::move(output));
            (void)inserted;
            check_cuda(cudaMemcpyAsync(
                           it->second.data.data(),
                           impl_->bindings[static_cast<std::size_t>(index)], bytes,
                           cudaMemcpyDeviceToHost, impl_->stream),
                       "cudaMemcpyAsync(Denglin D2H)");
        }
        impl_->pending = true;
    } catch (...) {
        (void)cudaStreamSynchronize(impl_->stream);
        impl_->pending = false;
        throw;
    }
}

void DenglinSession::enqueue() {
    infer_async();
}

void DenglinSession::synchronize() {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    if (!impl_->pending) return;
    try {
        check_cuda(cudaStreamSynchronize(impl_->stream),
                   "cudaStreamSynchronize(Denglin infer)");
        impl_->pending = false;
    } catch (...) {
        impl_->pending = false;
        throw;
    }
}

bool DenglinSession::is_pending() const noexcept {
    return impl_ && impl_->pending;
}

void DenglinSession::infer() {
    infer_async();
    synchronize();
}

Tensor DenglinSession::get_output(const std::string& name) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("get_output");
    const auto it = impl_->outputs.find(name);
    if (it == impl_->outputs.end()) {
        throw std::invalid_argument("Denglin output tensor is unavailable: " + name);
    }
    return it->second;
}

const Tensor& DenglinSession::get_output_ref(const std::string& name) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("get_output_ref");
    const auto it = impl_->outputs.find(name);
    if (it == impl_->outputs.end()) {
        throw std::invalid_argument("Denglin output tensor is unavailable: " + name);
    }
    return it->second;
}

std::vector<Tensor> DenglinSession::outputs() const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("outputs");
    std::vector<Tensor> result;
    result.reserve(impl_->outputs.size());
    for (const auto& name : impl_->output_names) {
        const auto it = impl_->outputs.find(name);
        if (it != impl_->outputs.end()) result.push_back(it->second);
    }
    return result;
}

std::vector<const Tensor*> DenglinSession::output_refs() const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    impl_->ensure_idle("output_refs");
    std::vector<const Tensor*> result;
    result.reserve(impl_->outputs.size());
    for (const auto& name : impl_->output_names) {
        const auto it = impl_->outputs.find(name);
        if (it != impl_->outputs.end()) result.push_back(&it->second);
    }
    return result;
}

const void* DenglinSession::device_output(const std::string& name) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    const int index = impl_->binding(name);
    if (impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Denglin tensor is not an output: " + name);
    }
    if (impl_->output_shapes.find(name) == impl_->output_shapes.end()) {
        throw std::invalid_argument("Denglin output tensor is unavailable: " + name);
    }
    return impl_->bindings[static_cast<std::size_t>(index)];
}

std::vector<int64_t> DenglinSession::device_output_shape(const std::string& name) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    const int index = impl_->binding(name);
    if (impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Denglin tensor is not an output: " + name);
    }
    const auto it = impl_->output_shapes.find(name);
    if (it == impl_->output_shapes.end()) {
        throw std::invalid_argument("Denglin output tensor is unavailable: " + name);
    }
    return it->second;
}

TensorDataType DenglinSession::device_output_type(const std::string& name) const {
    if (!impl_) throw std::runtime_error("Denglin session is not initialized");
    const int index = impl_->binding(name);
    if (impl_->engine->BindingIsInput(index)) {
        throw std::invalid_argument("Denglin tensor is not an output: " + name);
    }
    if (impl_->output_shapes.find(name) == impl_->output_shapes.end()) {
        throw std::invalid_argument("Denglin output tensor is unavailable: " + name);
    }
    return to_data_type(impl_->engine->GetBindingDataType(index));
}

void* DenglinSession::cuda_stream() const noexcept {
    return impl_ ? reinterpret_cast<void*>(impl_->stream) : nullptr;
}

} // namespace cvinfer::core
