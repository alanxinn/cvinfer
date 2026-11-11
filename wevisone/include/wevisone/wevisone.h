#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvinfer::wevisone {
enum class Task { Detection, Referring, Point, CountBox, CountPoint, Ocr,
                  OcrGrounding, Segmentation, Matting, VisualPrompt, Freeform };
Task parse_task(const std::string& name);
const char* task_name(Task task);
struct Box { float x1{}, y1{}, x2{}, y2{}; };
struct Options {
    int max_new_tokens{1024};
    float temperature{0};
    float top_p{.9F};
    int top_k{50};
    float repetition_penalty{1};
    float mask_threshold{.5F};
    uint64_t seed{0};
    bool decode_pixels{true};
};
struct Request {
    Task task{Task::Detection};
    std::vector<std::string> query;
    std::string prompt; // Nonempty overrides the task template.
    std::vector<Box> reference_boxes; // Original-image XYXY; VisualPrompt only.
    Options options;
};
struct Annotation {
    std::string label; // Category, referring phrase or OCR text; UTF-8.
    Box box;
    float x{}, y{};
    bool is_point{false};
    int mask_index{-1};
};
struct Mask {
    Box box;
    int width{}, height{};
    std::vector<float> alpha; // Original-image row-major [0,1], retained for both tasks.
    std::vector<uint8_t> binary; // 0/255, populated for Segmentation only.
};
struct Result {
    std::string prompt, response;
    std::vector<Annotation> annotations;
    std::vector<Mask> masks;
    std::vector<std::string> absent_labels;
    int count{-1}; // -1 for non-count tasks or malformed/truncated count output.
    bool rejected{false}, truncated{false};
    double latency_ms{};
};
// One instance per concurrent caller. Results own their storage.
class Model {
public:
    explicit Model(const std::string& bundle_directory);
    ~Model();
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    void set_image(const uint8_t* bgr, int width, int height, int channels = 3);
    Result predict(const Request& request);
    Result predict_bgr(const uint8_t* bgr, int width, int height, const Request& request);
    // Teacher-force official box+OBJ responses, using exact unquantized boxes for decoding.
    Result matte_boxes(const std::vector<Box>& boxes, const Options& options = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cvinfer::wevisone
