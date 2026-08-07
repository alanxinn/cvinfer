#include "yolo/yolo_cuda.h"

#include <stdexcept>

namespace cvinfer::yolo::cuda_backend {

struct PreprocessContext {};
struct SegmentationPostprocessContext {};

namespace {
[[noreturn]] void unavailable() {
    throw std::runtime_error(
        "CUDA backend is not included in this build; install "
        "alanxinn_cvinfer_cuda.dll or rebuild with BUILD_CUDA_BACKEND=ON");
}
} // namespace

std::shared_ptr<PreprocessContext> create_preprocess_context() {
    unavailable();
}

const float* preprocess_bgr(PreprocessContext&, const std::uint8_t*, int, int, int,
                            const Config&, void*, Letterbox*) {
    unavailable();
}

std::size_t preprocess_bytes(const Config&) {
    unavailable();
}

std::vector<Detection> decode_detections_device(
    const void*, const std::vector<std::int64_t>&, core::TensorDataType,
    const Config&, int, int, void*) {
    unavailable();
}

std::shared_ptr<SegmentationPostprocessContext>
create_segmentation_postprocess_context() {
    unavailable();
}

std::vector<Segmentation> decode_segmentations_device(
    const void*, const std::vector<std::int64_t>&, core::TensorDataType,
    const void*, const std::vector<std::int64_t>&, core::TensorDataType,
    const Config&, int, int, void*, SegmentationPostprocessContext&) {
    unavailable();
}

} // namespace cvinfer::yolo::cuda_backend
