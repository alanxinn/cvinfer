#pragma once

#include "yolo/yolo.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(_WIN32)
  #if defined(ALANXINN_CVINFER_CUDA_BUILD)
    #define ALANXINN_CUDA_API __declspec(dllexport)
  #elif defined(CVINFER_CUDA_STATIC)
    #define ALANXINN_CUDA_API
  #else
    #define ALANXINN_CUDA_API __declspec(dllimport)
  #endif
#else
  #define ALANXINN_CUDA_API
#endif

// Public CUDA backend ABI declaration.
//
// The implementation is intentionally kept in the private CUDA backend
// repository. The public YOLO library only depends on this interface and can
// be built in CPU-only backend mode.
namespace cvinfer::yolo::cuda_backend {
struct PreprocessContext;
struct SegmentationPostprocessContext;

ALANXINN_CUDA_API std::shared_ptr<PreprocessContext> create_preprocess_context();
ALANXINN_CUDA_API const float* preprocess_bgr(PreprocessContext& context,
                                               const std::uint8_t* data,
                                               int width,
                                               int height,
                                               int channels,
                                               const Config& config,
                                               void* cuda_stream,
                                               Letterbox* transform);
ALANXINN_CUDA_API std::size_t preprocess_bytes(const Config& config);

ALANXINN_CUDA_API std::vector<Detection> decode_detections_device(
    const void* device_data,
    const std::vector<std::int64_t>& shape,
    core::TensorDataType data_type,
    const Config& config,
    int image_width,
    int image_height,
    void* cuda_stream);

ALANXINN_CUDA_API std::shared_ptr<SegmentationPostprocessContext>
create_segmentation_postprocess_context();
ALANXINN_CUDA_API std::vector<Segmentation> decode_segmentations_device(
    const void* detection_device,
    const std::vector<std::int64_t>& detection_shape,
    core::TensorDataType detection_type,
    const void* prototype_device,
    const std::vector<std::int64_t>& prototype_shape,
    core::TensorDataType prototype_type,
    const Config& config,
    int image_width,
    int image_height,
    void* cuda_stream,
    SegmentationPostprocessContext& context);
} // namespace cvinfer::yolo::cuda_backend
