#pragma once

#include "yolo/yolo.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <string>

namespace cvinfer::yolo::demo {
inline ComputeBackend parse_backend(const std::string& value) {
    const std::string normalized = value;
    if (normalized == "auto") return ComputeBackend::Auto;
    if (normalized == "cpu") return ComputeBackend::Cpu;
    if (normalized == "cuda" || normalized == "gpu") return ComputeBackend::Cuda;
    throw std::invalid_argument("backend must be auto, cpu or cuda: " + value);
}
struct BenchmarkStats {
    int count{};
    double sum_preprocess{};
    double sum_inference{};
    double sum_postprocess{};
    double sum_total{};
    double min_preprocess{std::numeric_limits<double>::infinity()};
    double min_inference{std::numeric_limits<double>::infinity()};
    double min_postprocess{std::numeric_limits<double>::infinity()};
    double min_total{std::numeric_limits<double>::infinity()};
    double max_preprocess{};
    double max_inference{};
    double max_postprocess{};
    double max_total{};
    void add(const Timing& timing) {
        ++count;
        sum_preprocess += timing.preprocess_ms; sum_inference += timing.inference_ms;
        sum_postprocess += timing.postprocess_ms; sum_total += timing.total_ms;
        min_preprocess = (std::min)(min_preprocess, timing.preprocess_ms); min_inference = (std::min)(min_inference, timing.inference_ms);
        min_postprocess = (std::min)(min_postprocess, timing.postprocess_ms); min_total = (std::min)(min_total, timing.total_ms);
        max_preprocess = (std::max)(max_preprocess, timing.preprocess_ms); max_inference = (std::max)(max_inference, timing.inference_ms);
        max_postprocess = (std::max)(max_postprocess, timing.postprocess_ms); max_total = (std::max)(max_total, timing.total_ms);
    }
    void print(const std::string& name, int warmup, int iterations) const {
        if (count == 0) return;
        std::cout << name << " warmup=" << warmup << " iterations=" << iterations << " measured=" << count << '\n';
        std::cout << "preprocess avg=" << sum_preprocess / count << " min=" << min_preprocess << " max=" << max_preprocess << " ms\n";
        std::cout << "inference avg=" << sum_inference / count << " min=" << min_inference << " max=" << max_inference << " ms\n";
        std::cout << "postprocess avg=" << sum_postprocess / count << " min=" << min_postprocess << " max=" << max_postprocess << " ms\n";
        std::cout << "total avg=" << sum_total / count << " min=" << min_total << " max=" << max_total << " ms\n";
    }
};
}
