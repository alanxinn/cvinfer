#include "core/trt_core.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

// Dump named outputs for independent ONNX Runtime/PyTorch numerical comparison.
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: trt_raw_outputs engine input.bin output-directory");
        auto session = cvinfer::core::TrtSession::load(argv[1]);
        const auto names = session.input_names();
        if (names.size() != 1) throw std::runtime_error("raw-output test requires one input");
        const auto shape = session.tensor_shape(names[0]);
        session.set_shape(names[0], shape);
        std::ifstream file(std::filesystem::u8path(argv[2]), std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0) throw std::runtime_error("input file is missing or empty");
        std::vector<char> data(static_cast<std::size_t>(file.tellg()));
        file.seekg(0);
        if (!file.read(data.data(), static_cast<std::streamsize>(data.size()))) throw std::runtime_error("input read failed");
        session.set_input(names[0], data.data(), data.size());
        session.infer();
        const auto directory = std::filesystem::u8path(argv[3]);
        std::filesystem::create_directories(directory);
        std::ofstream metadata(directory / "outputs.tsv");
        for (const auto& output : session.outputs()) {
            const auto filename = std::to_string(metadata.tellp()) + ".bin";
            std::ofstream stream(directory / filename, std::ios::binary);
            stream.write(reinterpret_cast<const char*>(output.data.data()), static_cast<std::streamsize>(output.data.size()));
            metadata << output.name << '\t' << filename << '\t' << cvinfer::core::tensor_data_type_name(output.data_type);
            for (const auto dimension : output.shape) metadata << '\t' << dimension;
            metadata << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
