#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::sam2 {
struct Box { float x1{}, y1{}, x2{}, y2{}; };
struct Point { float x{}, y{}; bool positive{true}; };
struct Mask {
    Box box; // Tight nonzero-mask bounds, XYXY; empty masks use an all-zero box.
    float score{};
    int width{}, height{};
    std::vector<std::uint8_t> pixels; // Original-image size, row-major 0/255.
};
struct Config { float mask_threshold{0.5F}; int max_points{32}; };

// One instance per image/session; not thread-safe. Bundle: export_sam2.py.
class Segmentor {
public:
    explicit Segmentor(const std::string& engine_directory, Config config = {});
    ~Segmentor();
    Segmentor(Segmentor&&) noexcept;
    Segmentor& operator=(Segmentor&&) noexcept;
    Segmentor(const Segmentor&) = delete;
    Segmentor& operator=(const Segmentor&) = delete;
    void set_image(const std::uint8_t* bgr, int width, int height, int channels = 3);
    void set_box(Box box); // Resets points and previous-mask refinement state.
    void clear_box();
    void add_point(Point point); // Also supports point-only prompting.
    void remove_point(std::size_t index); // Zero-based; clears previous mask.
    void clear_points(); // Preserves the box; clears previous mask.
    void clear_prompts(); // Preserves cached image features.
    // Selects the highest-quality of three candidates. Refinement is explicit;
    // true requires a preceding successful prediction for this prompt history.
    Mask predict(bool use_previous_mask = false);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cvinfer::sam2
