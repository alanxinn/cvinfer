#pragma once

#include "rf_detr/rf_detr.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#if defined(_WIN32)
  #if defined(ALANXINN_CVINFER_CUDA_BUILD)
    #define ALANXINN_RF_DETR_CUDA_API __declspec(dllexport)
  #elif defined(CVINFER_RF_DETR_CUDA_STATIC)
    #define ALANXINN_RF_DETR_CUDA_API
  #else
    #define ALANXINN_RF_DETR_CUDA_API __declspec(dllimport)
  #endif
#else
  #define ALANXINN_RF_DETR_CUDA_API
#endif

namespace cvinfer::rf_detr::cuda_backend {

struct PreprocessContext;
struct DetectionPostprocessContext;
struct SegmentationPostprocessContext;

ALANXINN_RF_DETR_CUDA_API std::shared_ptr<PreprocessContext> create_preprocess_context();
ALANXINN_RF_DETR_CUDA_API const float* preprocess_bgr(
    PreprocessContext& context,
    const std::uint8_t* data,
    int width,
    int height,
    int channels,
    const Config& config,
    void* cuda_stream);
ALANXINN_RF_DETR_CUDA_API std::size_t preprocess_bytes(const Config& config);

ALANXINN_RF_DETR_CUDA_API std::shared_ptr<DetectionPostprocessContext>
create_detection_postprocess_context();
ALANXINN_RF_DETR_CUDA_API std::vector<Detection> decode_detections_device(
    const void* boxes_device,
    const std::vector<std::int64_t>& boxes_shape,
    core::TensorDataType boxes_type,
    const void* logits_device,
    const std::vector<std::int64_t>& logits_shape,
    core::TensorDataType logits_type,
    const Config& config,
    int image_width,
    int image_height,
    void* cuda_stream,
    DetectionPostprocessContext& context);

ALANXINN_RF_DETR_CUDA_API std::shared_ptr<SegmentationPostprocessContext>
create_segmentation_postprocess_context();
ALANXINN_RF_DETR_CUDA_API std::vector<Segmentation> decode_segmentations_device(
    const void* boxes_device,
    const std::vector<std::int64_t>& boxes_shape,
    core::TensorDataType boxes_type,
    const void* labels_device,
    const std::vector<std::int64_t>& labels_shape,
    core::TensorDataType labels_type,
    const void* masks_device,
    const std::vector<std::int64_t>& masks_shape,
    core::TensorDataType masks_type,
    const Config& config,
    int image_width,
    int image_height,
    void* cuda_stream,
    SegmentationPostprocessContext& context);

} // namespace cvinfer::rf_detr::cuda_backend
