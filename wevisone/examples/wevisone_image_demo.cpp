#include "wevisone/wevisone.h"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 4) { std::cerr << "Usage: alanxinn_wevisone_demo <engine-directory> <image> <task> [query] [x1 y1 x2 y2]\n"; return 1; }
        auto image = cv::imread(argv[2]);
        if (image.empty()) throw std::runtime_error("Cannot read image");
        cvinfer::wevisone::Model model(argv[1]);
        cvinfer::wevisone::Request r; r.task = cvinfer::wevisone::parse_task(argv[3]);
        if (argc > 4) r.query = {argv[4]};
        if (argc == 9) r.reference_boxes.push_back({std::stof(argv[5]), std::stof(argv[6]), std::stof(argv[7]), std::stof(argv[8])});
        const auto result = model.predict_bgr(image.data, image.cols, image.rows, r);
        std::cout << result.response << "\ncount=" << result.count << " rejected=" << result.rejected
                  << " truncated=" << result.truncated << " latency_ms=" << result.latency_ms << '\n';
        for (size_t i = 0; i < result.masks.size(); ++i) {
            const auto& m = result.masks[i]; cv::Mat alpha(m.height, m.width, CV_32F, const_cast<float*>(m.alpha.data()));
            cv::Mat png; alpha.convertTo(png, CV_8U, 255); cv::imwrite("wevisone_alpha_" + std::to_string(i) + ".png", png);
        }
        return result.truncated ? 3 : 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
