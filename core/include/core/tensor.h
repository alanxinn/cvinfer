#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cvinfer::core {

// Tensor metadata shared by all inference backends. Backend-specific headers
// should include this file instead of depending on another backend's session
// header.
enum class TensorDataType {
    Float32,
    Float16,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    Int4,
    UInt4,
    BFloat16,
    Unknown
};

inline const char* tensor_data_type_name(TensorDataType type) noexcept {
    switch (type) {
    case TensorDataType::Float32: return "FP32";
    case TensorDataType::Float16: return "FP16";
    case TensorDataType::Float64: return "FP64";
    case TensorDataType::Int8: return "INT8";
    case TensorDataType::Int16: return "INT16";
    case TensorDataType::Int32: return "INT32";
    case TensorDataType::Int64: return "INT64";
    case TensorDataType::UInt8: return "UINT8";
    case TensorDataType::UInt16: return "UINT16";
    case TensorDataType::UInt32: return "UINT32";
    case TensorDataType::UInt64: return "UINT64";
    case TensorDataType::Bool: return "BOOL";
    case TensorDataType::Int4: return "INT4";
    case TensorDataType::UInt4: return "UINT4";
    case TensorDataType::BFloat16: return "BF16";
    default: return "UNKNOWN";
    }
}

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    TensorDataType data_type{TensorDataType::Unknown};
    std::vector<std::uint8_t> data;
};

} // namespace cvinfer::core
