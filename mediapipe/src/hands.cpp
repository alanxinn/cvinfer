#include "mediapipe/hands.h"
#include "core/trt_core.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

static_assert(sizeof(hands_config_t) == 16, "MediaPipe config ABI changed");
static_assert(sizeof(hands_hand_t) == 532, "MediaPipe hand ABI changed");

namespace cvinfer::mediapipe {
namespace {
constexpr double pi = 3.14159265358979323846;
struct Palm { cv::Rect2d box; cv::Point2d points[7]; float score; };
void validate_config(const hands_config_t& config) {
    for (float threshold : {config.palm_threshold, config.hand_threshold, config.nms_threshold})
        if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
            throw std::invalid_argument("MediaPipe thresholds must be finite and in [0,1]");
    if (config.max_hands < 1 || config.max_hands > 64)
        throw std::invalid_argument("MediaPipe max_hands must be in [1,64]");
}
void validate_engine(core::TrtSession& session, const char* input, int side,
                     std::initializer_list<std::pair<const char*, std::vector<int64_t>>> outputs) {
    if (session.input_names() != std::vector<std::string>{input} ||
        session.tensor_shape(input) != std::vector<int64_t>{1, side, side, 3})
        throw std::runtime_error("MediaPipe engine has invalid NHWC input; use prepare_mediapipe_hands.py");
    const auto names = session.output_names();
    if (names.size() != outputs.size()) throw std::runtime_error("Invalid MediaPipe output count");
    for (const auto& output : outputs)
        if (std::find(names.begin(), names.end(), output.first) == names.end() ||
            session.tensor_shape(output.first) != output.second)
            throw std::runtime_error(std::string("Invalid MediaPipe output: ") + output.first);
}
std::vector<float> output(core::TrtSession& session, const char* name, std::size_t count) {
    const auto& tensor = session.get_output_ref(name);
    if (tensor.data_type != core::TensorDataType::Float32 || tensor.data.size() != count * sizeof(float))
        throw std::runtime_error(std::string("MediaPipe requires FP32 output: ") + name);
    std::vector<float> values(count);
    std::memcpy(values.data(), tensor.data.data(), tensor.data.size());
    for (float value : values)
        if (!std::isfinite(value)) throw std::runtime_error(std::string("Non-finite MediaPipe output: ") + name);
    return values;
}
void run(core::TrtSession& session, const char* name, const cv::Mat& bgr) {
    cv::Mat rgb, values;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(values, CV_32FC3, 1. / 255.);
    session.set_shape(name, {1, values.rows, values.cols, 3});
    session.set_input(name, values.ptr<float>(), values.total() * values.elemSize());
    session.infer();
}
cv::Point2d rotate(const cv::Point2d& p, double cosine, double sine) {
    return {cosine * p.x + sine * p.y, -sine * p.x + cosine * p.y};
}
}
struct HandDetector::Impl {
    hands_config_t config;
    core::TrtSession palm, hand;
    std::vector<cv::Point2d> anchors;
    Impl(const std::string& p, const std::string& h, hands_config_t c)
        : config(c), palm(core::TrtSession::load(p)), hand(core::TrtSession::load(h)) {
        validate_engine(palm, "palm_input", 192, {{"palm_boxes", {1, 2016, 18}}, {"palm_scores", {1, 2016, 1}}});
        validate_engine(hand, "hand_input", 224, {{"landmarks", {1, 63}}, {"confidence", {1, 1}},
            {"handedness", {1, 1}}, {"world_landmarks", {1, 63}}});
        for (const auto& grid : {std::pair<int,int>{24, 2}, {12, 6}})
            for (int y = 0; y < grid.first; ++y) for (int x = 0; x < grid.first; ++x)
                for (int k = 0; k < grid.second; ++k)
                    anchors.emplace_back((x + .5) * 192 / grid.first, (y + .5) * 192 / grid.first);
    }
    std::vector<Palm> detect(const cv::Mat& image) {
        const double scale = 192. / std::max(image.cols, image.rows);
        const int width = std::max(1, int(image.cols * scale)), height = std::max(1, int(image.rows * scale));
        const int left = (192 - width) / 2, top = (192 - height) / 2;
        cv::Mat canvas = cv::Mat::zeros(192, 192, CV_8UC3);
        cv::Mat resized;
        cv::resize(image, resized, {width, height}, 0, 0, cv::INTER_LINEAR);
        resized.copyTo(canvas(cv::Rect(left, top, width, height)));
        run(palm, "palm_input", canvas);
        const auto boxes = output(palm, "palm_boxes", 2016 * 18), logits = output(palm, "palm_scores", 2016);
        std::vector<int> indices(2016);
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        std::vector<Palm> palms;
        for (int i : indices) {
            const float score = 1.F / (1.F + std::exp(-std::clamp(logits[i], -80.F, 80.F)));
            if (score < config.palm_threshold) continue;
            const float* v = boxes.data() + i * 18;
            const auto center = (cv::Point2d(v[0], v[1]) + anchors[i] - cv::Point2d(left, top)) * (1. / scale);
            const double w = v[2] / scale, h = v[3] / scale;
            if (w <= 0 || h <= 0) continue;
            Palm candidate{};
            candidate.box = {center.x - w / 2, center.y - h / 2, w, h}; candidate.score = score;
            bool keep = true;
            for (const auto& other : palms) {
                const double intersection = (candidate.box & other.box).area();
                const double union_area = candidate.box.area() + other.box.area() - intersection;
                if (intersection / std::max(union_area, 1e-12) > config.nms_threshold) { keep = false; break; }
            }
            if (!keep) continue;
            for (int k = 0; k < 7; ++k)
                candidate.points[k] = (cv::Point2d(v[4 + k * 2], v[5 + k * 2]) + anchors[i] -
                                       cv::Point2d(left, top)) * (1. / scale);
            palms.push_back(candidate);
            if (palms.size() >= static_cast<std::size_t>(config.max_hands)) break;
        }
        return palms;
    }
    bool estimate(const cv::Mat& image, const Palm& detection, hands_hand_t& result) {
        const auto direction = detection.points[2] - detection.points[0];
        const double angle = pi / 2 + std::atan2(direction.y, direction.x);
        const double cosine = std::cos(angle), sine = std::sin(angle);
        cv::Point2d lo(std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        cv::Point2d hi(-lo.x, -lo.y);
        for (const auto& p : detection.points) {
            const auto r = rotate(p, cosine, sine);
            lo.x = std::min(lo.x, r.x); lo.y = std::min(lo.y, r.y);
            hi.x = std::max(hi.x, r.x); hi.y = std::max(hi.y, r.y);
        }
        const auto center = (lo + hi) * .5 + cv::Point2d(0, -.4 * (hi.y - lo.y));
        const double side = std::max(hi.x - lo.x, hi.y - lo.y) * 3;
        if (!std::isfinite(side) || side < 1e-3) return false;
        const double factor = 224. / side;
        const cv::Matx23d affine(cosine * factor, sine * factor, 112 - center.x * factor,
                                -sine * factor, cosine * factor, 112 - center.y * factor);
        cv::Mat crop;
        cv::warpAffine(image, crop, affine, {224, 224}, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
        run(hand, "hand_input", crop);
        const auto confidence = output(hand, "confidence", 1);
        if (confidence[0] < config.hand_threshold) return false;
        const auto landmarks = output(hand, "landmarks", 63), world = output(hand, "world_landmarks", 63);
        result.confidence = confidence[0]; result.palm_score = detection.score;
        result.handedness = output(hand, "handedness", 1)[0];
        result.box[0] = result.box[1] = std::numeric_limits<float>::max();
        result.box[2] = result.box[3] = -std::numeric_limits<float>::max();
        for (int i = 0; i < 21; ++i) {
            const auto aligned = cv::Point2d((landmarks[i * 3] - 112) / factor + center.x,
                                             (landmarks[i * 3 + 1] - 112) / factor + center.y);
            const auto p = rotate(aligned, cosine, -sine);
            result.landmarks[i * 3] = static_cast<float>(p.x);
            result.landmarks[i * 3 + 1] = static_cast<float>(p.y);
            result.landmarks[i * 3 + 2] = static_cast<float>(landmarks[i * 3 + 2] / factor);
            const auto wp = rotate({world[i * 3], world[i * 3 + 1]}, cosine, -sine);
            result.world_landmarks[i * 3] = static_cast<float>(wp.x);
            result.world_landmarks[i * 3 + 1] = static_cast<float>(wp.y);
            result.world_landmarks[i * 3 + 2] = world[i * 3 + 2];
            result.box[0] = std::min(result.box[0], static_cast<float>(p.x));
            result.box[1] = std::min(result.box[1], static_cast<float>(p.y));
            result.box[2] = std::max(result.box[2], static_cast<float>(p.x));
            result.box[3] = std::max(result.box[3], static_cast<float>(p.y));
        }
        return true;
    }
};
HandDetector::HandDetector(const std::string& palm, const std::string& hand, hands_config_t config) {
    validate_config(config);
    impl_ = std::make_unique<Impl>(palm, hand, config);
}
HandDetector::~HandDetector() = default;
HandDetector::HandDetector(HandDetector&&) noexcept = default;
HandDetector& HandDetector::operator=(HandDetector&&) noexcept = default;
std::vector<hands_hand_t> HandDetector::infer_bgr(const uint8_t* bgr, int width, int height, int channels) {
    if (!impl_) throw std::runtime_error("MediaPipe detector is not initialized");
    if (!bgr || width <= 0 || height <= 0 || channels != 3 ||
        static_cast<std::size_t>(width) > std::numeric_limits<std::size_t>::max() / 3 / height)
        throw std::invalid_argument("MediaPipe requires a contiguous 3-channel BGR image");
    const cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(bgr));
    std::vector<hands_hand_t> result;
    for (const auto& palm : impl_->detect(image)) {
        hands_hand_t hand{};
        if (impl_->estimate(image, palm, hand)) result.push_back(hand);
    }
    return result;
}
} // namespace cvinfer::mediapipe

struct hands_model {
    cvinfer::mediapipe::HandDetector detector;
    hands_model(const char* palm, const char* hand, hands_config_t config) : detector(palm, hand, config) {}
};
namespace {
thread_local std::string last_error;
template<class F> int guarded(F&& fn) noexcept {
    try { last_error.clear(); return fn(); }
    catch (const std::invalid_argument& e) { last_error = e.what(); return 1; }
    catch (const std::exception& e) { last_error = e.what(); return 2; }
    catch (...) { last_error = "Unknown MediaPipe error"; return 2; }
}
}
extern "C" {
const char* hands_last_error(void) { return last_error.c_str(); }
void hands_config_init(hands_config_t* config) { if (config) *config = {.5F, .5F, .3F, 2}; }
int hands_model_create(const char* palm, const char* hand, const hands_config_t* config, hands_model_t** model) {
    if (model) *model = nullptr;
    return guarded([&] {
        if (!model || !palm || !hand || !*palm || !*hand) throw std::invalid_argument("Engine paths and model pointer are required");
        hands_config_t c; hands_config_init(&c); if (config) c = *config;
        auto owned = std::make_unique<hands_model>(palm, hand, c);
        *model = owned.release(); return 0;
    });
}
void hands_model_destroy(hands_model_t* model) { delete model; }
int hands_model_infer_bgr(hands_model_t* model, const uint8_t* bgr, int32_t width, int32_t height,
                         int32_t channels, hands_hand_t* output, size_t capacity, size_t* count) {
    if (count) *count = 0;
    return guarded([&] {
        if (!model || !count || (!output && capacity)) throw std::invalid_argument("Invalid MediaPipe output buffer/model/count");
        const auto result = model->detector.infer_bgr(bgr, width, height, channels);
        *count = result.size();
        if (capacity < result.size()) { last_error = "MediaPipe output buffer is too small"; return 4; }
        if (!result.empty()) std::copy(result.begin(), result.end(), output);
        return 0;
    });
}
}
