#include "tensorrt_builder/tensorrt_builder_api.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_last_error;

void clear_error() { g_last_error.clear(); }
void set_error(const std::string& message) { g_last_error = message; }

class ProgressReporter {
public:
    ProgressReporter(trt_builder_progress_callback_t callback, void* user_data)
        : callback_(callback), user_data_(user_data) {}

    ~ProgressReporter() { stop_heartbeat(); }

    void report(int32_t stage, int32_t percent, const char* message, bool heartbeat = false) const noexcept {
        if (!callback_) return;
        try {
            callback_(stage, percent, heartbeat ? 1 : 0, message ? message : "", user_data_);
        } catch (...) {
            // Never allow exceptions from a managed callback to cross the C ABI.
        }
    }

    void start_heartbeat(int32_t stage, int32_t percent, const char* message) {
        if (!callback_ || heartbeat_running_.exchange(true)) return;
        heartbeat_stage_ = stage;
        heartbeat_percent_ = percent;
        heartbeat_message_ = message ? message : "";
        heartbeat_started_ = std::chrono::steady_clock::now();
        heartbeat_thread_ = std::thread([this]() {
            while (heartbeat_running_.load()) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - heartbeat_started_).count();
                std::ostringstream text;
                text << heartbeat_message_ << " (elapsed " << elapsed << " s)";
                report(heartbeat_stage_, heartbeat_percent_, text.str().c_str(), true);
                std::unique_lock<std::mutex> lock(heartbeat_mutex_);
                heartbeat_cv_.wait_for(lock, std::chrono::seconds(1), [this]() {
                    return !heartbeat_running_.load();
                });
            }
        });
    }

    void stop_heartbeat() noexcept {
        if (!heartbeat_running_.exchange(false)) return;
        heartbeat_cv_.notify_all();
        if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
    }

private:
    trt_builder_progress_callback_t callback_{};
    void* user_data_{};
    std::atomic<bool> heartbeat_running_{false};
    std::thread heartbeat_thread_;
    std::chrono::steady_clock::time_point heartbeat_started_{};
    int32_t heartbeat_stage_{};
    int32_t heartbeat_percent_{};
    std::string heartbeat_message_;
    std::mutex heartbeat_mutex_;
    std::condition_variable heartbeat_cv_;
};

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (message && severity <= Severity::kERROR) {
            last_error_ = message;
        }
    }

    std::string last_error_;
};

template <typename T>
struct TrtDestroyer {
    void operator()(T* object) const noexcept {
        if (object) object->destroy();
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDestroyer<T>>;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

std::string trim(const std::string& value) {
    std::size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    std::size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
    return value.substr(first, last - first);
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> result;
    std::size_t start = 0;
    while (start <= value.size()) {
        const auto end = value.find(separator, start);
        result.push_back(trim(value.substr(start, end == std::string::npos ? std::string::npos : end - start)));
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

struct ShapeSpec {
    std::string name;
    std::vector<int64_t> dimensions;
};

std::vector<int64_t> parse_dimensions(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("Tensor shape is empty");
    std::vector<int64_t> dimensions;
    for (const auto& token : split(text, 'x')) {
        if (token.empty()) throw std::invalid_argument("Tensor shape contains an empty dimension");
        std::size_t position = 0;
        const auto value = std::stoll(token, &position);
        if (position != token.size() || value <= 0) {
            throw std::invalid_argument("Tensor shape dimensions must be positive integers");
        }
        dimensions.push_back(value);
    }
    return dimensions;
}

std::vector<ShapeSpec> parse_shapes(const char* text, const char* label) {
    if (!text || !*text) throw std::invalid_argument(std::string(label) + " is required");
    std::vector<ShapeSpec> result;
    std::map<std::string, bool> names;
    for (const auto& item : split(text, ',')) {
        if (item.empty()) throw std::invalid_argument(std::string(label) + " contains an empty item");
        const auto separator = item.find(':');
        if (separator == std::string::npos) {
            throw std::invalid_argument(std::string(label) + " must use name:dimxdim syntax");
        }
        ShapeSpec shape;
        shape.name = trim(item.substr(0, separator));
        if (shape.name.empty()) throw std::invalid_argument(std::string(label) + " contains an empty tensor name");
        if (names.emplace(shape.name, true).second == false)
            throw std::invalid_argument(std::string(label) + " contains a duplicate tensor name: " + shape.name);
        shape.dimensions = parse_dimensions(trim(item.substr(separator + 1)));
        result.push_back(std::move(shape));
    }
    if (result.empty()) throw std::invalid_argument(std::string(label) + " is empty");
    return result;
}

using ShapeMap = std::map<std::string, std::vector<int64_t>>;

template <typename TConfig>
auto add_optimization_profile(TConfig& config, nvinfer1::IOptimizationProfile* profile, int)
    -> decltype(config.addOptimizationProfile(profile)) {
    return config.addOptimizationProfile(profile);
}

template <typename TConfig>
auto add_optimization_profile(TConfig& config, nvinfer1::IOptimizationProfile* profile, long)
    -> decltype(config.addOptimizationProfile(*profile)) {
    return config.addOptimizationProfile(*profile);
}

ShapeMap make_shape_map(const std::vector<ShapeSpec>& shapes) {
    ShapeMap result;
    for (const auto& shape : shapes) result.emplace(shape.name, shape.dimensions);
    return result;
}

nvinfer1::Dims to_dims(const std::vector<int64_t>& shape) {
    if (shape.empty() || shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS))
        throw std::invalid_argument("Tensor shape rank is invalid");
    nvinfer1::Dims result{};
    result.nbDims = static_cast<int>(shape.size());
    for (int i = 0; i < result.nbDims; ++i) {
        const auto value = shape[static_cast<std::size_t>(i)];
        if (value <= 0 || value > std::numeric_limits<int>::max())
            throw std::invalid_argument("Tensor shape dimensions are out of range");
        result.d[i] = static_cast<int>(value);
    }
    return result;
}

void validate_profile(const trt_builder_profile_t& profile) {
    const auto min_shapes = make_shape_map(parse_shapes(profile.min_shapes, "min_shapes"));
    const auto opt_shapes = make_shape_map(parse_shapes(profile.opt_shapes, "opt_shapes"));
    const auto max_shapes = make_shape_map(parse_shapes(profile.max_shapes, "max_shapes"));
    if (min_shapes.size() != opt_shapes.size() || min_shapes.size() != max_shapes.size())
        throw std::invalid_argument("min, opt and max shapes must contain the same tensor names");
    for (const auto& item : min_shapes) {
        const auto opt = opt_shapes.find(item.first);
        const auto max = max_shapes.find(item.first);
        if (opt == opt_shapes.end() || max == max_shapes.end())
            throw std::invalid_argument("min, opt and max shapes must contain the same tensor names");
        if (item.second.size() != opt->second.size() || item.second.size() != max->second.size())
            throw std::invalid_argument("min, opt and max shapes must have the same rank");
        for (std::size_t i = 0; i < item.second.size(); ++i) {
            if (item.second[i] > opt->second[i] || opt->second[i] > max->second[i])
                throw std::invalid_argument("Shape dimensions must satisfy min <= opt <= max");
        }
    }
}

void configure_profiles(
    nvinfer1::IBuilder& builder,
    nvinfer1::INetworkDefinition& network,
    nvinfer1::IBuilderConfig& config,
    const trt_builder_profile_t* profiles,
    std::size_t profile_count) {
    bool has_dynamic_input = false;
    for (int i = 0; i < network.getNbInputs(); ++i) {
        const auto* input = network.getInput(i);
        const auto dims = input->getDimensions();
        for (int dimension = 0; dimension < dims.nbDims; ++dimension) {
            if (dims.d[dimension] < 0) {
                has_dynamic_input = true;
                break;
            }
        }
    }
    if (has_dynamic_input && profile_count == 0)
        throw std::invalid_argument("Dynamic ONNX inputs require at least one optimization profile");

    for (std::size_t profile_index = 0; profile_index < profile_count; ++profile_index) {
        if (!profiles) throw std::invalid_argument("Optimization profile array is null");
        validate_profile(profiles[profile_index]);
        const auto min_shapes = make_shape_map(parse_shapes(profiles[profile_index].min_shapes, "min_shapes"));
        const auto opt_shapes = make_shape_map(parse_shapes(profiles[profile_index].opt_shapes, "opt_shapes"));
        const auto max_shapes = make_shape_map(parse_shapes(profiles[profile_index].max_shapes, "max_shapes"));
        for (const auto& shape : min_shapes) {
            bool known_input = false;
            for (int input_index = 0; input_index < network.getNbInputs(); ++input_index) {
                const auto* input = network.getInput(input_index);
                if (input->getName() && shape.first == input->getName()) {
                    known_input = true;
                    break;
                }
            }
            if (!known_input) throw std::invalid_argument("Optimization profile contains unknown input: " + shape.first);
        }
        // TensorRT 8.6 的 IOptimizationProfile 没有 destroy()；其生命周期由
        // IBuilderConfig 接管，不能套用通用 TrtPtr 删除器。
        auto* optimization_profile = builder.createOptimizationProfile();
        if (!optimization_profile) throw std::runtime_error("TensorRT optimization profile creation failed");
        for (int input_index = 0; input_index < network.getNbInputs(); ++input_index) {
            auto* input = network.getInput(input_index);
            const std::string name = input->getName() ? input->getName() : "";
            const auto min = min_shapes.find(name);
            const auto opt = opt_shapes.find(name);
            const auto max = max_shapes.find(name);
            if (min == min_shapes.end() || opt == opt_shapes.end() || max == max_shapes.end()) {
                throw std::invalid_argument("Optimization profile is missing input: " + name);
            }
            if (!optimization_profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, to_dims(min->second)) ||
                !optimization_profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, to_dims(opt->second)) ||
                !optimization_profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, to_dims(max->second))) {
                throw std::runtime_error("TensorRT rejected optimization profile input: " + name);
            }
        }
        if (!optimization_profile->isValid()) {
            throw std::invalid_argument("TensorRT optimization profile is invalid");
        }
        if (add_optimization_profile(config, optimization_profile, 0) < 0)
            throw std::runtime_error("TensorRT rejected optimization profile");
    }
}

void write_engine(const char* path, const nvinfer1::IHostMemory& serialized) {
    std::ofstream output(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(std::string("Unable to open engine output: ") + path);
    output.write(static_cast<const char*>(serialized.data()), static_cast<std::streamsize>(serialized.size()));
    if (!output) throw std::runtime_error(std::string("Unable to write engine output: ") + path);
}

std::vector<std::uint8_t> read_onnx(const char* path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error(std::string("Unable to open ONNX file: ") + path);
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error(std::string("ONNX file is empty: ") + path);
    input.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    if (!input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size)))
        throw std::runtime_error(std::string("Unable to read ONNX file: ") + path);
    return data;
}

std::string parser_errors(nvonnxparser::IParser& parser) {
    std::ostringstream message;
    for (int i = 0; i < parser.getNbErrors(); ++i) {
        const auto* error = parser.getError(i);
        if (error) message << (i == 0 ? "" : "\n") << error->desc();
    }
    return message.str();
}

} // namespace

extern "C" {

TENSORRT_BUILDER_API const char* TENSORRT_BUILDER_CALL trt_builder_last_error(void) {
    return g_last_error.c_str();
}

TENSORRT_BUILDER_API void TENSORRT_BUILDER_CALL trt_builder_config_init(
    trt_builder_config_t* config) {
    if (!config) return;
    config->precision = TRT_BUILDER_PRECISION_FP32;
    config->workspace_mib = 4096;
    config->device_id = -1;
    config->enable_tf32 = 1;
    config->enable_sparse_weights = 0;
    config->strict_types = 0;
}

TENSORRT_BUILDER_API trt_builder_status_t TENSORRT_BUILDER_CALL
trt_builder_build_onnx_with_progress(
    const char* onnx_path,
    const char* engine_path,
    const trt_builder_config_t* source_config,
    const trt_builder_profile_t* profiles,
    size_t profile_count,
    trt_builder_progress_callback_t progress_callback,
    void* progress_user_data) {
    clear_error();
    ProgressReporter progress(progress_callback, progress_user_data);
    progress.report(0, 0, "Starting ONNX to TensorRT conversion");
    if (!onnx_path || !*onnx_path || !engine_path || !*engine_path)
        return set_error("ONNX path and engine path are required"), TRT_BUILDER_STATUS_INVALID_ARGUMENT;
    if (profile_count > 0 && !profiles)
        return set_error("Optimization profile array is required"), TRT_BUILDER_STATUS_INVALID_ARGUMENT;

    trt_builder_config_t defaults{};
    trt_builder_config_init(&defaults);
    const auto& config = source_config ? *source_config : defaults;
    if (config.precision < TRT_BUILDER_PRECISION_FP32 || config.precision > TRT_BUILDER_PRECISION_INT8)
        return set_error("Unsupported TensorRT precision"), TRT_BUILDER_STATUS_INVALID_ARGUMENT;
    if (config.workspace_mib <= 0)
        return set_error("workspace_mib must be positive"), TRT_BUILDER_STATUS_INVALID_ARGUMENT;
    if (static_cast<std::uint64_t>(config.workspace_mib) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) / (1024U * 1024U))
        return set_error("workspace_mib is too large"), TRT_BUILDER_STATUS_INVALID_ARGUMENT;
    try {
        if (config.device_id >= 0) check_cuda(cudaSetDevice(config.device_id), "cudaSetDevice");
        progress.report(1, 10, "CUDA device initialized");
        Logger logger;
        if (!initLibNvInferPlugins(&logger, ""))
            throw std::runtime_error("TensorRT plugin registry initialization failed");
        progress.report(2, 15, "TensorRT plugins initialized");
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) throw std::runtime_error("TensorRT builder creation failed");
        progress.report(3, 20, "TensorRT builder created");
        const auto network_flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(network_flags));
        if (!network) {
            throw std::runtime_error("TensorRT network creation failed");
        }
        progress.report(4, 25, "Network created");
        TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser) {
            throw std::runtime_error("TensorRT ONNX parser creation failed");
        }
        progress.report(5, 30, "ONNX parser created");
        const auto onnx_data = read_onnx(onnx_path);
        progress.report(6, 35, "ONNX file loaded");
        if (!parser->parse(onnx_data.data(), onnx_data.size(), onnx_path)) {
            const auto details = parser_errors(*parser);
            throw std::runtime_error(details.empty() ? "TensorRT ONNX parsing failed" : details);
        }
        progress.report(7, 45, "ONNX parsed");
        TrtPtr<nvinfer1::IBuilderConfig> builder_config(builder->createBuilderConfig());
        if (!builder_config) {
            throw std::runtime_error("TensorRT builder config creation failed");
        }
        progress.report(8, 50, "Builder configuration created");
        builder_config->setMaxWorkspaceSize(static_cast<std::size_t>(config.workspace_mib) * 1024U * 1024U);
        // TensorRT 8.6 can build this transformer graph with FP16, but the
        // resulting RF-DETR classification logits are numerically corrupted
        // (all scores collapse below the postprocess threshold).  Detect the
        // exported RF-DETR contract and keep it in FP32 for correctness.
        bool is_rf_detr = false;
        for (int output_index = 0; output_index < network->getNbOutputs(); ++output_index) {
            const auto* output = network->getOutput(output_index);
            const auto dims = output ? output->getDimensions() : nvinfer1::Dims{};
            const int last = dims.nbDims > 0 ? dims.d[dims.nbDims - 1] : 0;
            if ((output && output->getName() && std::string(output->getName()) == "labels") || last == 91) {
                is_rf_detr = true;
                break;
            }
        }
        const bool use_fp16 = config.precision == TRT_BUILDER_PRECISION_FP16 && !is_rf_detr;
        if (use_fp16)
            builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        else if (config.precision == TRT_BUILDER_PRECISION_INT8)
            builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        if (config.enable_tf32) builder_config->setFlag(nvinfer1::BuilderFlag::kTF32);
        else builder_config->clearFlag(nvinfer1::BuilderFlag::kTF32);
        if (config.enable_sparse_weights) builder_config->setFlag(nvinfer1::BuilderFlag::kSPARSE_WEIGHTS);
        if (config.strict_types) builder_config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        configure_profiles(*builder, *network, *builder_config, profiles, profile_count);
        progress.report(9, 60, "Optimization profiles configured");
        progress.start_heartbeat(10, 65, "Building TensorRT engine");
        TrtPtr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *builder_config));
        progress.stop_heartbeat();
        if (!engine) {
            throw std::runtime_error(logger.last_error_.empty() ? "TensorRT engine build failed" : logger.last_error_);
        }
        TrtPtr<nvinfer1::IHostMemory> serialized(engine->serialize());
        if (!serialized) {
            throw std::runtime_error("TensorRT engine serialization failed");
        }
        progress.report(11, 92, "Engine serialized");
        write_engine(engine_path, *serialized);
        progress.report(12, 98, "Engine file written");
        progress.report(13, 100, "Conversion completed");
        return TRT_BUILDER_STATUS_OK;
    } catch (const std::invalid_argument& error) {
        set_error(error.what());
        return TRT_BUILDER_STATUS_INVALID_ARGUMENT;
    } catch (const std::ios_base::failure& error) {
        set_error(error.what());
        return TRT_BUILDER_STATUS_IO_ERROR;
    } catch (const std::exception& error) {
        set_error(error.what());
        return TRT_BUILDER_STATUS_BUILD_ERROR;
    } catch (...) {
        set_error("Unknown TensorRT ONNX build failure");
        return TRT_BUILDER_STATUS_BUILD_ERROR;
    }
}

TENSORRT_BUILDER_API trt_builder_status_t TENSORRT_BUILDER_CALL trt_builder_build_onnx(
    const char* onnx_path,
    const char* engine_path,
    const trt_builder_config_t* config,
    const trt_builder_profile_t* profiles,
    size_t profile_count) {
    return trt_builder_build_onnx_with_progress(
        onnx_path, engine_path, config, profiles, profile_count, nullptr, nullptr);
}

} // extern "C"
