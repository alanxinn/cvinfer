#pragma once

#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::core {

enum class DenglinModelFormat {
    Auto,
    Onnx,
    SerializedEngine
};

// Cache policy for ONNX -> Denglin engine builds. Disabled preserves the
// stage-one behavior; ReadOnly can consume an existing cache, and ReadWrite
// creates/refreshes the cache after a successful build.
enum class DenglinCacheMode {
    Disabled,
    ReadOnly,
    ReadWrite
};

enum class DenglinInputLayout {
    Linear,
    NCHW,
    NHWC
};

enum class DenglinWeightShareMode {
    Single = 1,
    Share2 = 2,
    Share4 = 4
};

enum class DenglinClusterConfig {
    Cluster0,
    Cluster1,
    Cluster2,
    Cluster3,
    Cluster01,
    Cluster23,
    Cluster02,
    Cluster13,
    Cluster03,
    Cluster12,
    Cluster0123
};

// A dynamic input profile describes the legal min/opt/max dimensions for the
// input registered in DenglinSessionOptions::input_name.  The input_shape
// field uses -1 for wildcard dimensions when one or more profiles are used.
struct DenglinOptimizationProfile {
    std::vector<int64_t> min_shape;
    std::vector<int64_t> opt_shape;
    std::vector<int64_t> max_shape;
};

// Auto uses the modern tensor-address API when the loaded SDK engine exposes
// it and falls back to binding pointers for older engines.  Bindings keeps the
// legacy path available for compatibility; TensorApi requires EnqueueV3.
enum class DenglinExecutionMode {
    Auto,
    Bindings,
    TensorApi
};

struct DenglinSessionOptions {
    DenglinModelFormat model_format{DenglinModelFormat::Auto};

    // Optional serialized Denglin engine cache used only when loading ONNX.
    // The cache contains a versioned metadata envelope and is not a TensorRT
    // engine. ReadWrite saves a cache after an ONNX build; ReadOnly never
    // writes it.
    DenglinCacheMode cache_mode{DenglinCacheMode::Disabled};
    std::string engine_cache_path;
    bool build_on_cache_miss{true};

    // ONNX parsing requires an explicit input name and shape. Use -1 for
    // dynamic dimensions and provide optimization_profiles in that case. The
    // default layout is deliberately NCHW because the current YOLO/RF-DETR
    // pipeline produces CHW tensors.
    std::string input_name;
    std::vector<int64_t> input_shape;
    DenglinInputLayout input_layout{DenglinInputLayout::NCHW};
    std::vector<std::string> output_names;

    // Optional optimization profiles for dynamic input_shape dimensions.
    // Each profile contains min <= opt <= max shapes.  Runtime shape changes
    // are applied with set_shape() after loading the session.
    std::vector<DenglinOptimizationProfile> optimization_profiles;
    // -1 keeps the SDK-selected profile (normally profile 0 for a new
    // context).  Set a non-negative value to select a profile explicitly.
    int optimization_profile{-1};

    DenglinExecutionMode execution_mode{DenglinExecutionMode::Auto};

    int max_batch_size{1};
    int batch_size{1};
    DenglinWeightShareMode weight_share{DenglinWeightShareMode::Single};
    DenglinClusterConfig cluster{DenglinClusterConfig::Cluster0};
    std::uint32_t builder_flags{};

    // Optional user-op registration for models requiring a custom plugin.
    std::string plugin_library;
    std::string plugin_module;
    std::string plugin_first_op;
};

class DenglinSession {
public:
    DenglinSession();
    ~DenglinSession();
    DenglinSession(DenglinSession&&) noexcept;
    DenglinSession& operator=(DenglinSession&&) noexcept;
    DenglinSession(const DenglinSession&) = delete;
    DenglinSession& operator=(const DenglinSession&) = delete;

    // Loads an ONNX file and builds a Denglin engine, or deserializes a raw
    // Denglin engine/cache envelope when model_format is SerializedEngine/Auto
    // for a non-ONNX path. TensorRT serialized engines are not accepted here.
    static DenglinSession load(
        const std::string& model_path,
        DenglinSessionOptions options = {});

    // Writes the raw SDK Serialize() payload. This file can be loaded again
    // with model_format=SerializedEngine. It is deliberately different from
    // engine_cache_path, which adds metadata for safe cache validation.
    void serialize_engine(const std::string& engine_path) const;
    // True when this session was restored from engine_cache_path or a cache
    // envelope passed as a SerializedEngine model.
    bool cache_hit() const noexcept;

    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;
    std::vector<int64_t> tensor_shape(const std::string& name) const;
    void set_shape(const std::string& name, const std::vector<int64_t>& shape);
    void set_optimization_profile(int profile_index);
    int optimization_profile() const;
    int optimization_profile_count() const;
    DenglinOptimizationProfile optimization_profile_shapes(int profile_index) const;

    void set_input(const std::string& name, const void* data, std::size_t bytes);
    void set_input_device(const std::string& name,
                          const void* device_data,
                          std::size_t bytes);
    // Optional caller-owned device output. Pass nullptr/0 to restore an
    // internally allocated output buffer. The buffer must remain valid until
    // synchronize() returns.
    void set_output_device(const std::string& name,
                           void* device_data,
                           std::size_t bytes);
    void set_output_host_copy(bool enabled);

    // Enqueues H2D copies, dlNNE Enqueue/EnqueueV3 and optional D2H copies on
    // the session stream, then waits for completion.
    void infer();
    // Asynchronous counterpart.  Host outputs and caller-owned device input/
    // output buffers become usable after synchronize().  While pending, do
    // not replace input/output pointers or read host outputs.
    void infer_async();
    // Alias emphasizing the SDK operation used by this backend.
    void enqueue();
    void synchronize();
    bool is_pending() const noexcept;

    Tensor get_output(const std::string& name) const;
    // The returned view is valid until the next infer()/enqueue() call.
    const Tensor& get_output_ref(const std::string& name) const;
    std::vector<Tensor> outputs() const;
    // Returned views are valid until the next infer()/enqueue() call.
    std::vector<const Tensor*> output_refs() const;

    // The device pointer remains valid until the next infer/enqueue or shape
    // change.  For async execution, wait for synchronize() before consuming
    // its contents unless the consumer is queued on cuda_stream().
    const void* device_output(const std::string& name) const;
    std::vector<int64_t> device_output_shape(const std::string& name) const;
    TensorDataType device_output_type(const std::string& name) const;
    void* cuda_stream() const noexcept;

private:
    struct Impl;
    explicit DenglinSession(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace cvinfer::core
