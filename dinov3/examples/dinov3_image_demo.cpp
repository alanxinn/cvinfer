#include "dinov3/dinov3.h"
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::invalid_argument("usage: alanxinn_dinov3_demo model.engine image.jpg output-directory");
        const auto image = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (image.empty() || !image.isContinuous()) throw std::runtime_error("Cannot read a contiguous BGR image");
        cvinfer::dinov3::FeatureExtractor model(argv[1]);
        const auto result = model.infer_bgr(image.data, image.cols, image.rows);
        const auto directory = std::filesystem::u8path(argv[3]);
        std::filesystem::create_directories(directory);
        std::ofstream metadata(directory / "tensors.tsv");
        if (!metadata) throw std::runtime_error("Cannot open tensor metadata output");
        for (const auto* tensor : {&result.global_embedding, &result.normalized_embedding, &result.patch_tokens, &result.feature_map}) {
            std::ofstream file(directory / (tensor->name + ".bin"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(tensor->data.data()), static_cast<std::streamsize>(tensor->data.size()));
            if (!file) throw std::runtime_error("Cannot write DINOv3 tensor output");
            metadata << tensor->name << '\t' << "FP32";
            std::cout << tensor->name << ":";
            for (const auto dimension : tensor->shape) { metadata << '\t' << dimension; std::cout << ' ' << dimension; }
            metadata << '\n'; std::cout << '\n';
        }
        if (!metadata) throw std::runtime_error("Cannot write tensor metadata");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
