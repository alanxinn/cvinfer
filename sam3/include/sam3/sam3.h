#pragma once
#include "core/tensor.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::sam3 {
struct Box { float x1{}, y1{}, x2{}, y2{}; };
struct Point { float x{}, y{}; bool positive{true}; };
struct Mask {
    Box box;
    float score{};
    int width{}, height{};
    // Full original-image mask, row-major, values 0/255.
    std::vector<std::uint8_t> pixels;
};
struct Config {
    float confidence_threshold{0.3F};
    float mask_threshold{0.5F};
    int max_points{32};
};

// Engine bundle produced by scripts/export_sam3.py. Sessions are independent;
// a session must not be used concurrently by multiple threads.
class Segmentor {
public:
    explicit Segmentor(const std::string& engine_directory, Config config = {});
    ~Segmentor();
    Segmentor(Segmentor&&) noexcept;
    Segmentor& operator=(Segmentor&&) noexcept;
    Segmentor(const Segmentor&) = delete;
    Segmentor& operator=(const Segmentor&) = delete;
    void set_image(const std::uint8_t* bgr, int width, int height, int channels = 3);
    void set_box(Box box);
    void add_point(Point point);
    void clear_points();
    Mask predict();
    std::vector<Mask> predict_text(const std::string& text);
    // Allows an application's official Hugging Face tokenizer to supply any
    // Unicode input; tokens include BOS/EOS and are padded to the engine size.
    std::vector<Mask> predict_tokens(const std::vector<std::int64_t>& ids,
                                     const std::vector<std::int64_t>& attention_mask);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cvinfer::sam3
