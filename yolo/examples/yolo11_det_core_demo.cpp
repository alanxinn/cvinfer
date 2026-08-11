#include "yolo/yolo.h"
#include "demo_benchmark.h"
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: alanxinn_yolo11_det_core_demo <engine> <image> [input_w] [input_h] [classes] [preprocessed_output] [result_output] [warmup] [iterations] [preprocess_backend] [postprocess_backend]\n";
        return 2;
    }
    try {
        cvinfer::yolo::Config config;
        const auto probe = cvinfer::core::TrtSession::load(argv[1]);
        const auto input_names = probe.input_names();
        if (!input_names.empty()) {
            const auto shape = probe.tensor_shape(input_names.front());
            if (shape.size() == 4 && shape[2] > 0 && shape[3] > 0) {
                config.input_height = static_cast<int>(shape[2]);
                config.input_width = static_cast<int>(shape[3]);
            }
        }
        if (argc > 3) config.input_width = std::stoi(argv[3]);
        if (argc > 4) config.input_height = std::stoi(argv[4]);
        if (argc > 5) config.class_count = std::stoi(argv[5]);
        if (argc > 10) config.preprocess_backend = cvinfer::yolo::demo::parse_backend(argv[10]);
        if (argc > 11) config.postprocess_backend = cvinfer::yolo::demo::parse_backend(argv[11]);
        config.enable_timing = true;
        cvinfer::yolo::Detector detector(argv[1], config);
        const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "failed to read image: " << argv[2] << '\n';
            return 3;
        }
        cvinfer::yolo::Letterbox transform;
        const auto chw = cvinfer::yolo::preprocess_bgr(image.data, image.cols, image.rows, image.channels(), config, &transform);
        cv::Mat preprocessed(config.input_height, config.input_width, CV_8UC3);
        const std::size_t plane = static_cast<std::size_t>(config.input_width) * config.input_height;
        for (int y = 0; y < config.input_height; ++y) for (int x = 0; x < config.input_width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * config.input_width + x;
            auto& pixel = preprocessed.at<cv::Vec3b>(y, x);
            pixel[0] = cv::saturate_cast<unsigned char>(chw[2 * plane + index] * 255.0F);
            pixel[1] = cv::saturate_cast<unsigned char>(chw[plane + index] * 255.0F);
            pixel[2] = cv::saturate_cast<unsigned char>(chw[index] * 255.0F);
        }
        const bool benchmark_mode = argc > 9;
        const std::string preprocess_path = argc > 6 ? argv[6] : "preprocessed_det.jpg";
        if (!benchmark_mode) {
            if (!cv::imwrite(preprocess_path, preprocessed)) throw std::runtime_error("failed to save preprocessed image");
            std::cout << "preprocessed image saved: " << preprocess_path << " scale=" << transform.scale
                      << " pad=" << transform.pad_x << "," << transform.pad_y << "\n";
        } else {
            std::cout << "benchmark mode: skip intermediate image output\n";
        }
        const int warmup = argc > 8 ? (std::max)(0, std::stoi(argv[8])) : 5;
        const int iterations = argc > 9 ? (std::max)(1, std::stoi(argv[9])) : 50;
        for (int i = 0; i < warmup; ++i) detector.infer_image(image.data, image.cols, image.rows, image.channels());
        cvinfer::yolo::demo::BenchmarkStats stats;
        std::vector<cvinfer::yolo::Detection> results;
        for (int i = 0; i < iterations; ++i) {
            results = detector.infer_image(image.data, image.cols, image.rows, image.channels());
            stats.add(detector.last_timing());
        }
        stats.print("detector", warmup, iterations);
        std::cout << "detections=" << results.size() << '\n';
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& box = results[i].box;
            std::cout << i << ": class=" << box.class_id << " score=" << box.score
                      << " box=" << box.x1 << ',' << box.y1 << ',' << box.x2 << ',' << box.y2 << '\n';
        }
        cv::Mat result_image = image.clone();
        for (const auto& item : results) {
            const auto& b = item.box;
            const cv::Rect rect(static_cast<int>(b.x1), static_cast<int>(b.y1),
                                (std::max)(1, static_cast<int>(b.x2 - b.x1)),
                                (std::max)(1, static_cast<int>(b.y2 - b.y1)));
            cv::rectangle(result_image, rect, cv::Scalar(0, 255, 0), 2);
            cv::putText(result_image, std::to_string(b.class_id) + ":" + std::to_string(b.score),
                        cv::Point(rect.x, (std::max)(15, rect.y - 4)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        const std::string result_path = argc > 7 ? argv[7] : "result_det.jpg";
        if (!cv::imwrite(result_path, result_image)) throw std::runtime_error("failed to save detection result");
        std::cout << "result image saved: " << result_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "detector error: " << error.what() << '\n';
        return 1;
    }
}
