#include "core/denglin_core.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::int64_t> parse_shape(const std::string& text) {
    if (text.empty()) throw std::invalid_argument("shape is empty");

    std::vector<std::int64_t> result;
    std::string value;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const char character = i < text.size() ? text[i] : 'x';
        if (character == 'x' || character == 'X' || character == ',') {
            if (value.empty()) throw std::invalid_argument("shape contains an empty dimension");
            const auto dimension = std::stoll(value);
            if (dimension <= 0) throw std::invalid_argument("shape dimensions must be positive");
            result.push_back(dimension);
            value.clear();
        } else {
            value.push_back(character);
        }
    }
    if (result.empty()) throw std::invalid_argument("shape is empty");
    return result;
}

std::size_t volume(const std::vector<std::int64_t>& shape) {
    std::size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            result > static_cast<std::size_t>(-1) / static_cast<std::size_t>(dimension)) {
            throw std::overflow_error("tensor volume overflow");
        }
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

std::string shape_string(const std::vector<std::int64_t>& shape) {
    std::ostringstream result;
    result << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) result << ',';
        result << shape[i];
    }
    result << ']';
    return result.str();
}

void validate_host_outputs(const std::vector<cvinfer::core::Tensor>& outputs) {
    if (outputs.empty()) throw std::runtime_error("Denglin returned no host outputs");
    for (const auto& output : outputs) {
        if (output.name.empty() || output.shape.empty() || output.data.empty()) {
            throw std::runtime_error("Denglin returned an incomplete output: " + output.name);
        }
        for (const auto dimension : output.shape) {
            if (dimension <= 0) {
                throw std::runtime_error("Denglin returned an unresolved output shape: " +
                                         output.name);
            }
        }
        if (output.data_type == cvinfer::core::TensorDataType::Float32 &&
            output.data.size() % sizeof(float) == 0) {
            const auto count = output.data.size() / sizeof(float);
            for (std::size_t i = 0; i < count; ++i) {
                float value = 0.0F;
                std::memcpy(&value, output.data.data() + i * sizeof(float), sizeof(value));
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Denglin output contains NaN/Inf: " + output.name);
                }
            }
        }
    }
}

void print_output_summary(const std::vector<cvinfer::core::Tensor>& outputs) {
    for (const auto& output : outputs) {
        std::cout << "output=" << output.name << " shape=" << shape_string(output.shape)
                  << " type=" << cvinfer::core::tensor_data_type_name(output.data_type)
                  << " bytes=" << output.data.size();
        if (output.data_type == cvinfer::core::TensorDataType::Float32 &&
            output.data.size() >= sizeof(float)) {
            const auto* values = reinterpret_cast<const float*>(output.data.data());
            std::cout << " first=";
            const auto count = (std::min<std::size_t>)(8, output.data.size() / sizeof(float));
            for (std::size_t i = 0; i < count; ++i) {
                if (i != 0) std::cout << ',';
                std::cout << values[i];
            }
            if (output.shape.size() == 3 && output.shape[1] > 4) {
                const auto boxes = static_cast<std::size_t>(output.shape[2]);
                float class_min = values[4 * boxes];
                float class_max = class_min;
                for (std::size_t i = 4 * boxes; i < output.data.size() / sizeof(float); ++i) {
                    class_min = (std::min)(class_min, values[i]);
                    class_max = (std::max)(class_max, values[i]);
                }
                std::cout << " class0=" << values[4 * boxes]
                          << " class1=" << values[5 * boxes]
                          << " class2=" << values[6 * boxes]
                          << " class3=" << values[7 * boxes]
                          << " class_min=" << class_min << " class_max=" << class_max;
                std::cout << " samples=" << values[4 * boxes + 100]
                          << "," << values[4 * boxes + 1000]
                          << "," << values[4 * boxes + 4000];
            }
        }
        std::cout << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: alanxinn_cvinfer_denglin_smoke <model.onnx> "
                     "[iterations] [input_name] [shape] [cache_path] [input_f32]\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const int iterations = argc > 2 ? (std::max)(1, std::stoi(argv[2])) : 1;
        const std::string input_name = argc > 3 ? argv[3] : "images";
        const auto requested_shape = argc > 4
            ? parse_shape(argv[4])
            : std::vector<std::int64_t>{1, 3, 640, 640};
        const std::string cache_argument = argc > 5 ? argv[5] : std::string{};
        const std::string cache_path = cache_argument == "-" ? std::string{} : cache_argument;
        const std::string input_path = argc > 6 ? argv[6] : std::string{};

        cvinfer::core::DenglinSessionOptions options;
        options.input_name = input_name;
        options.input_shape = requested_shape;
        if (!cache_path.empty()) {
            options.cache_mode = cvinfer::core::DenglinCacheMode::ReadWrite;
            options.engine_cache_path = cache_path;
        }

        auto session = cvinfer::core::DenglinSession::load(model_path, options);
        const auto input_names = session.input_names();
        const auto output_names = session.output_names();
        if (input_names.empty() || output_names.empty()) {
            throw std::runtime_error("Denglin model has no input or output IO tensor");
        }

        const auto actual_shape = session.tensor_shape(input_name);
        if (actual_shape.empty()) throw std::runtime_error("Denglin input shape is empty");
        const auto input_elements = volume(actual_shape);
        std::vector<float> input(input_elements);
        if (!input_path.empty()) {
            std::ifstream file(input_path, std::ios::binary);
            if (!file) throw std::runtime_error("unable to open input_f32: " + input_path);
            file.read(reinterpret_cast<char*>(input.data()),
                      static_cast<std::streamsize>(input.size() * sizeof(float)));
            if (file.gcount() != static_cast<std::streamsize>(input.size() * sizeof(float)))
                throw std::runtime_error("input_f32 has an unexpected size");
        } else {
            for (std::size_t i = 0; i < input.size(); ++i) {
                input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 17.0F;
            }
        }

        session.set_input(input_name, input.data(), input.size() * sizeof(float));
        for (int i = 0; i < iterations; ++i) session.infer();
        const auto outputs = session.outputs();
        validate_host_outputs(outputs);
        print_output_summary(outputs);

        // Stage-three coverage: re-apply the current shape, enqueue without
        // synchronizing, then explicitly wait before reading host outputs.
        session.set_shape(input_name, actual_shape);
        session.infer_async();
        if (!session.is_pending()) {
            throw std::runtime_error("Denglin async inference was not marked pending");
        }
        session.synchronize();
        validate_host_outputs(session.outputs());

        session.set_output_host_copy(false);
        session.infer();
        for (const auto& name : output_names) {
            if (!session.device_output(name)) {
                throw std::runtime_error("Denglin device output is null: " + name);
            }
            if (session.device_output_shape(name).empty()) {
                throw std::runtime_error("Denglin device output shape is empty: " + name);
            }
        }

        if (!cache_path.empty()) {
            const auto raw_engine_path = cache_path + ".raw.engine";
            session.serialize_engine(raw_engine_path);

            auto restored_options = options;
            restored_options.model_format = cvinfer::core::DenglinModelFormat::SerializedEngine;
            restored_options.cache_mode = cvinfer::core::DenglinCacheMode::Disabled;
            restored_options.engine_cache_path.clear();
            auto restored = cvinfer::core::DenglinSession::load(
                raw_engine_path, restored_options);
            restored.set_input(input_name, input.data(), input.size() * sizeof(float));
            restored.infer();
            validate_host_outputs(restored.outputs());

            auto cached_options = options;
            cached_options.cache_mode = cvinfer::core::DenglinCacheMode::ReadOnly;
            cached_options.build_on_cache_miss = false;
            auto cached = cvinfer::core::DenglinSession::load(model_path, cached_options);
            if (!cached.cache_hit()) throw std::runtime_error("Denglin cache was not used");
            cached.set_input(input_name, input.data(), input.size() * sizeof(float));
            cached.infer();
            validate_host_outputs(cached.outputs());
        }

        std::cout << "Denglin Core smoke PASS\n"
                  << "model=" << model_path << "\n"
                  << "input=" << input_name << " shape=" << shape_string(actual_shape) << "\n"
                  << "outputs=" << output_names.size() << " iterations=" << iterations
                  << (cache_path.empty() ? "" : " cache=PASS") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Denglin Core smoke FAIL: " << error.what() << '\n';
        return 1;
    }
}
