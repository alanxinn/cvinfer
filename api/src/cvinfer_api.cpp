#include "cvinfer/cvinfer_api.h"

#include "yolo/yolo.h"
#include "rf_detr/rf_detr.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct cvinfer_model {
    cvinfer_model_type_t type{};
    bool timing_enabled{};
    std::unique_ptr<cvinfer::yolo::Detector> detector;
    std::unique_ptr<cvinfer::yolo::Segmentor> segmentor;
    std::unique_ptr<cvinfer::yolo::PoseEstimator> pose_estimator;
    std::unique_ptr<cvinfer::rf_detr::Detector> rf_detr_detector;
    std::unique_ptr<cvinfer::rf_detr::Segmentor> rf_detr_segmentor;
};

struct cvinfer_result {
    cvinfer_model_type_t type{};
    std::vector<cvinfer::yolo::Detection> detections;
    std::vector<cvinfer::yolo::Segmentation> segmentations;
    std::vector<cvinfer::yolo::Pose> poses;
    std::vector<cvinfer::rf_detr::Detection> rf_detr_detections;
    std::vector<cvinfer::rf_detr::Segmentation> rf_detr_segmentations;
    cvinfer_timing_t timing{};
};

namespace {
thread_local std::string g_last_error;

void clear_error() { g_last_error.clear(); }
void set_error(const char* message) { g_last_error = message ? message : "Unknown error"; }
void set_error(const std::exception& error) { g_last_error = error.what(); }

cvinfer_status_t invalid_argument(const char* message) {
    set_error(message);
    return CVINFER_STATUS_INVALID_ARGUMENT;
}

cvinfer::yolo::Config make_config(const cvinfer_model_config_t* source) {
    cvinfer::yolo::Config config;
    if (!source) return config;
    if (source->input_width > 0) config.input_width = source->input_width;
    if (source->input_height > 0) config.input_height = source->input_height;
    if (source->class_count > 0) config.class_count = source->class_count;
    if (source->confidence_threshold >= 0.0F) config.confidence_threshold = source->confidence_threshold;
    if (source->nms_threshold >= 0.0F) config.nms_threshold = source->nms_threshold;
    if (source->mask_threshold >= 0.0F) config.mask_threshold = source->mask_threshold;
    config.class_agnostic_nms = source->class_agnostic_nms != 0;
    if (source->input_name) config.input_name = source->input_name;
    if (source->output_name) config.output_name = source->output_name;
    if (source->prototype_name) config.prototype_name = source->prototype_name;
    return config;
}

cvinfer::yolo::ComputeBackend parse_backend(int32_t value) {
    switch (value) {
    case CVINFER_BACKEND_AUTO: return cvinfer::yolo::ComputeBackend::Auto;
    case CVINFER_BACKEND_CPU: return cvinfer::yolo::ComputeBackend::Cpu;
    case CVINFER_BACKEND_CUDA: return cvinfer::yolo::ComputeBackend::Cuda;
    default: throw std::invalid_argument("Unsupported compute backend");
    }
}

cvinfer::rf_detr::ComputeBackend parse_rf_detr_backend(int32_t value) {
    switch (value) {
    case CVINFER_BACKEND_AUTO: return cvinfer::rf_detr::ComputeBackend::Auto;
    case CVINFER_BACKEND_CPU: return cvinfer::rf_detr::ComputeBackend::Cpu;
    case CVINFER_BACKEND_CUDA: return cvinfer::rf_detr::ComputeBackend::Cuda;
    default: throw std::invalid_argument("Unsupported compute backend");
    }
}

cvinfer::yolo::Config make_config_v2(const cvinfer_model_config_v2_t* source) {
    cvinfer::yolo::Config config;
    if (!source) return config;
    if (source->input_width > 0) config.input_width = source->input_width;
    if (source->input_height > 0) config.input_height = source->input_height;
    if (source->class_count > 0) config.class_count = source->class_count;
    if (source->confidence_threshold >= 0.0F) config.confidence_threshold = source->confidence_threshold;
    if (source->nms_threshold >= 0.0F) config.nms_threshold = source->nms_threshold;
    if (source->mask_threshold >= 0.0F) config.mask_threshold = source->mask_threshold;
    config.class_agnostic_nms = source->class_agnostic_nms != 0;
    config.preprocess_backend = parse_backend(source->preprocess_backend);
    config.postprocess_backend = parse_backend(source->postprocess_backend);
    if (source->keypoint_count > 0) config.keypoint_count = source->keypoint_count;
    if (source->keypoint_dims > 0) config.keypoint_dims = source->keypoint_dims;
    if (source->input_name) config.input_name = source->input_name;
    if (source->output_name) config.output_name = source->output_name;
    if (source->prototype_name) config.prototype_name = source->prototype_name;
    return config;
}

cvinfer::rf_detr::Config make_rf_detr_config(const cvinfer_model_config_t* source) {
    cvinfer::rf_detr::Config config;
    if (!source) return config;
    if (source->input_width > 0) config.input_width = source->input_width;
    if (source->input_height > 0) config.input_height = source->input_height;
    if (source->class_count > 0) config.class_count = source->class_count;
    if (source->confidence_threshold >= 0.0F) config.confidence_threshold = source->confidence_threshold;
    if (source->nms_threshold >= 0.0F) config.nms_threshold = source->nms_threshold;
    if (source->mask_threshold >= 0.0F) config.mask_threshold = source->mask_threshold;
    config.class_agnostic_nms = source->class_agnostic_nms != 0;
    if (source->input_name) config.input_name = source->input_name;
    if (source->output_name) config.output_name = source->output_name;
    return config;
}

cvinfer::rf_detr::Config make_rf_detr_config_v2(const cvinfer_model_config_v2_t* source) {
    auto config = make_rf_detr_config(reinterpret_cast<const cvinfer_model_config_t*>(source));
    if (source) {
        config.preprocess_backend = parse_rf_detr_backend(source->preprocess_backend);
        config.postprocess_backend = parse_rf_detr_backend(source->postprocess_backend);
    }
    return config;
}
cvinfer_status_t handle_exception(const std::exception& error) {
    set_error(error);
    return CVINFER_STATUS_RUNTIME_ERROR;
}
} // namespace

extern "C" {

CVINFER_API const char* CVINFER_CALL cvinfer_last_error(void) {
    return g_last_error.c_str();
}

CVINFER_API void CVINFER_CALL cvinfer_model_config_init(cvinfer_model_config_t* config) {
    if (!config) return;
    config->input_width = 0;
    config->input_height = 0;
    config->class_count = 0;
    config->confidence_threshold = 0.25F;
    config->nms_threshold = 0.45F;
    config->mask_threshold = 0.5F;
    config->class_agnostic_nms = 0;
    config->input_name = nullptr;
    config->output_name = nullptr;
    config->prototype_name = nullptr;
}

CVINFER_API void CVINFER_CALL cvinfer_model_config_v2_init(cvinfer_model_config_v2_t* config) {
    if (!config) return;
    config->input_width = 0;
    config->input_height = 0;
    config->class_count = 0;
    config->confidence_threshold = 0.25F;
    config->nms_threshold = 0.45F;
    config->mask_threshold = 0.5F;
    config->class_agnostic_nms = 0;
    config->input_name = nullptr;
    config->output_name = nullptr;
    config->prototype_name = nullptr;
    config->preprocess_backend = CVINFER_BACKEND_AUTO;
    config->postprocess_backend = CVINFER_BACKEND_AUTO;
    config->keypoint_count = 17;
    config->keypoint_dims = 3;
}
CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_create(
    const char* engine_path,
    cvinfer_model_type_t model_type,
    const cvinfer_model_config_t* config,
    cvinfer_model_t** model) {
    clear_error();
    if (!engine_path || !*engine_path || !model) return invalid_argument("engine_path, model and output handle are required");
    *model = nullptr;
    try {
        auto instance = std::make_unique<cvinfer_model>();
        instance->type = model_type;
        const auto yolo_config = make_config(config);
        if (model_type == CVINFER_MODEL_YOLO11_DET || model_type == CVINFER_MODEL_YOLO26_DET) {
            instance->detector = std::make_unique<cvinfer::yolo::Detector>(engine_path, yolo_config);
        } else if (model_type == CVINFER_MODEL_YOLO11_SEG || model_type == CVINFER_MODEL_YOLO26_SEG) {
            instance->segmentor = std::make_unique<cvinfer::yolo::Segmentor>(engine_path, yolo_config);
        } else if (model_type == CVINFER_MODEL_YOLO11_POSE || model_type == CVINFER_MODEL_YOLO26_POSE) {
            instance->pose_estimator = std::make_unique<cvinfer::yolo::PoseEstimator>(engine_path, yolo_config);
        } else if (model_type == CVINFER_MODEL_RF_DETR_DET) {
            instance->rf_detr_detector = std::make_unique<cvinfer::rf_detr::Detector>(engine_path, make_rf_detr_config(config));
        } else if (model_type == CVINFER_MODEL_RF_DETR_SEG) {
            instance->rf_detr_segmentor = std::make_unique<cvinfer::rf_detr::Segmentor>(engine_path, make_rf_detr_config(config));
        } else {
            set_error("Unsupported model type");
            return CVINFER_STATUS_UNSUPPORTED;
        }
        *model = instance.release();
        return CVINFER_STATUS_OK;
    } catch (const std::exception& error) {
        return handle_exception(error);
    }
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_create_v2(
    const char* engine_path,
    cvinfer_model_type_t model_type,
    const cvinfer_model_config_v2_t* config,
    cvinfer_model_t** model) {
    clear_error();
    if (!engine_path || !*engine_path || !model) return invalid_argument("engine_path, model and output handle are required");
    *model = nullptr;
    try {
        auto instance = std::make_unique<cvinfer_model>();
        instance->type = model_type;
        const auto yolo_config = make_config_v2(config);
        if (model_type == CVINFER_MODEL_YOLO11_DET || model_type == CVINFER_MODEL_YOLO26_DET) {
            instance->detector = std::make_unique<cvinfer::yolo::Detector>(engine_path, yolo_config);
        } else if (model_type == CVINFER_MODEL_YOLO11_SEG || model_type == CVINFER_MODEL_YOLO26_SEG) {
            instance->segmentor = std::make_unique<cvinfer::yolo::Segmentor>(engine_path, yolo_config);
        } else if (model_type == CVINFER_MODEL_YOLO11_POSE || model_type == CVINFER_MODEL_YOLO26_POSE) {
            instance->pose_estimator = std::make_unique<cvinfer::yolo::PoseEstimator>(engine_path, yolo_config);
        } else if (model_type == CVINFER_MODEL_RF_DETR_DET) {
            instance->rf_detr_detector = std::make_unique<cvinfer::rf_detr::Detector>(engine_path, make_rf_detr_config_v2(config));
        } else if (model_type == CVINFER_MODEL_RF_DETR_SEG) {
            instance->rf_detr_segmentor = std::make_unique<cvinfer::rf_detr::Segmentor>(engine_path, make_rf_detr_config_v2(config));
        } else {
            set_error("Unsupported model type");
            return CVINFER_STATUS_UNSUPPORTED;
        }
        *model = instance.release();
        return CVINFER_STATUS_OK;
    } catch (const std::exception& error) {
        return handle_exception(error);
    }
}
CVINFER_API void CVINFER_CALL cvinfer_model_destroy(cvinfer_model_t* model) {
    delete model;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_set_timing_enabled(
    cvinfer_model_t* model,
    int32_t enabled) {
    clear_error();
    if (!model) return invalid_argument("model is required");
    const bool value = enabled != 0;
    if (model->detector) {
        model->detector->set_timing_enabled(value);
    } else if (model->segmentor) {
        model->segmentor->set_timing_enabled(value);
    } else if (model->pose_estimator) {
        model->pose_estimator->set_timing_enabled(value);
    } else if (model->rf_detr_detector) {
        model->rf_detr_detector->set_timing_enabled(value);
    } else if (model->rf_detr_segmentor) {
        model->rf_detr_segmentor->set_timing_enabled(value);
    } else {
        return invalid_argument("model handle is invalid");
    }
    model->timing_enabled = value;
    return CVINFER_STATUS_OK;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_model_infer_bgr(
    cvinfer_model_t* model,
    const uint8_t* bgr,
    int32_t width,
    int32_t height,
    int32_t channels,
    cvinfer_result_t** result) {
    clear_error();
    if (!model || !bgr || width <= 0 || height <= 0 || channels < 3 || !result)
        return invalid_argument("model, BGR image, dimensions and result are required");
    *result = nullptr;
    try {
        auto output = std::make_unique<cvinfer_result>();
        output->type = model->type;
        if ((model->type == CVINFER_MODEL_YOLO11_DET || model->type == CVINFER_MODEL_YOLO26_DET) && model->detector) {
            output->detections = model->detector->infer_image(bgr, width, height, channels);
            if (model->timing_enabled) {
                const auto& timing = model->detector->last_timing();
                output->timing = {timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms, timing.total_ms};
            }
        } else if ((model->type == CVINFER_MODEL_YOLO11_SEG || model->type == CVINFER_MODEL_YOLO26_SEG) && model->segmentor) {
            output->segmentations = model->segmentor->infer_image(bgr, width, height, channels);
            if (model->timing_enabled) {
                const auto& timing = model->segmentor->last_timing();
                output->timing = {timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms, timing.total_ms};
            }
        } else if ((model->type == CVINFER_MODEL_YOLO11_POSE || model->type == CVINFER_MODEL_YOLO26_POSE) && model->pose_estimator) {
            output->poses = model->pose_estimator->infer_image(bgr, width, height, channels);
            if (model->timing_enabled) {
                const auto& timing = model->pose_estimator->last_timing();
                output->timing = {timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms, timing.total_ms};
            }
        } else if (model->type == CVINFER_MODEL_RF_DETR_DET && model->rf_detr_detector) {
            output->rf_detr_detections = model->rf_detr_detector->infer_image(bgr, width, height, channels);
            if (model->timing_enabled) {
                const auto& timing = model->rf_detr_detector->last_timing();
                output->timing = {timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms, timing.total_ms};
            }
        } else if (model->type == CVINFER_MODEL_RF_DETR_SEG && model->rf_detr_segmentor) {
            output->rf_detr_segmentations = model->rf_detr_segmentor->infer_image(bgr, width, height, channels);
            if (model->timing_enabled) {
                const auto& timing = model->rf_detr_segmentor->last_timing();
                output->timing = {timing.preprocess_ms, timing.inference_ms, timing.postprocess_ms, timing.total_ms};
            }
        } else {
            set_error("Invalid model handle");
            return CVINFER_STATUS_INVALID_ARGUMENT;
        }
        *result = output.release();
        return CVINFER_STATUS_OK;
    } catch (const std::exception& error) {
        return handle_exception(error);
    }
}

CVINFER_API void CVINFER_CALL cvinfer_result_destroy(cvinfer_result_t* result) {
    delete result;
}

CVINFER_API cvinfer_model_type_t CVINFER_CALL cvinfer_result_type(const cvinfer_result_t* result) {
    return result ? result->type : static_cast<cvinfer_model_type_t>(0);
}

CVINFER_API size_t CVINFER_CALL cvinfer_result_count(const cvinfer_result_t* result) {
    if (!result) return 0;
    if (result->type == CVINFER_MODEL_YOLO11_SEG || result->type == CVINFER_MODEL_YOLO26_SEG)
        return result->segmentations.size();
    if (result->type == CVINFER_MODEL_YOLO11_POSE || result->type == CVINFER_MODEL_YOLO26_POSE)
        return result->poses.size();
    if (result->type == CVINFER_MODEL_RF_DETR_DET) return result->rf_detr_detections.size();
    if (result->type == CVINFER_MODEL_RF_DETR_SEG) return result->rf_detr_segmentations.size();
    return result->detections.size();
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_detection(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_detection_t* detection) {
    clear_error();
    if (!result || !detection) return invalid_argument("result and detection are required");
    if (result->type != CVINFER_MODEL_YOLO11_DET &&
        result->type != CVINFER_MODEL_YOLO26_DET &&
        result->type != CVINFER_MODEL_RF_DETR_DET)
        return invalid_argument("Result is not a detection result");
    if (result->type == CVINFER_MODEL_RF_DETR_DET) {
        if (index >= result->rf_detr_detections.size()) return invalid_argument("Detection index is out of range");
        const auto& source = result->rf_detr_detections[index].box;
        detection->box = {source.x1, source.y1, source.x2, source.y2, source.score, source.class_id};
    } else {
        if (index >= result->detections.size()) return invalid_argument("Detection index is out of range");
        const auto& source = result->detections[index].box;
        detection->box = {source.x1, source.y1, source.x2, source.y2, source.score, source.class_id};
    }
    return CVINFER_STATUS_OK;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_pose(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_pose_info_t* pose) {
    clear_error();
    if (!result || !pose) return invalid_argument("result and pose are required");
    if (result->type != CVINFER_MODEL_YOLO11_POSE &&
        result->type != CVINFER_MODEL_YOLO26_POSE)
        return invalid_argument("Result is not a pose result");
    if (index >= result->poses.size()) return invalid_argument("Pose index is out of range");
    const auto& source = result->poses[index];
    pose->box = {source.box.x1, source.box.y1, source.box.x2, source.box.y2, source.box.score, source.box.class_id};
    pose->keypoint_count = static_cast<int32_t>(source.keypoints.size());
    pose->keypoint_dims = source.keypoint_dims;
    pose->keypoint_bytes = source.keypoints.size() * sizeof(cvinfer_keypoint_t);
    return CVINFER_STATUS_OK;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_copy_keypoints(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_keypoint_t* destination,
    size_t destination_bytes,
    size_t* copied_bytes) {
    clear_error();
    if (!result) return invalid_argument("result is required");
    if (result->type != CVINFER_MODEL_YOLO11_POSE &&
        result->type != CVINFER_MODEL_YOLO26_POSE)
        return invalid_argument("Result is not a pose result");
    if (index >= result->poses.size()) return invalid_argument("Pose index is out of range");
    const auto& keypoints = result->poses[index].keypoints;
    const size_t bytes = keypoints.size() * sizeof(cvinfer_keypoint_t);
    if (copied_bytes) *copied_bytes = bytes;
    if (destination_bytes < bytes) {
        set_error("Destination keypoint buffer is too small");
        return CVINFER_STATUS_BUFFER_TOO_SMALL;
    }
    if (bytes > 0 && !destination) return invalid_argument("Destination keypoint buffer is required");
    if (bytes > 0) std::memcpy(destination, keypoints.data(), bytes);
    return CVINFER_STATUS_OK;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_segmentation(
    const cvinfer_result_t* result,
    size_t index,
    cvinfer_segmentation_info_t* segmentation) {
    clear_error();
    if (!result || !segmentation) return invalid_argument("result and segmentation are required");
    if (result->type == CVINFER_MODEL_YOLO11_SEG ||
        result->type == CVINFER_MODEL_YOLO26_SEG) {
        if (index >= result->segmentations.size()) return invalid_argument("Segmentation index is out of range");
        const auto& source = result->segmentations[index];
        segmentation->box = {source.box.x1, source.box.y1, source.box.x2, source.box.y2, source.box.score, source.box.class_id};
        segmentation->mask_width = source.mask_width;
        segmentation->mask_height = source.mask_height;
        segmentation->mask_bytes = source.mask.size();
    } else if (result->type == CVINFER_MODEL_RF_DETR_SEG) {
        if (index >= result->rf_detr_segmentations.size()) return invalid_argument("Segmentation index is out of range");
        const auto& source = result->rf_detr_segmentations[index];
        segmentation->box = {source.box.x1, source.box.y1, source.box.x2, source.box.y2, source.box.score, source.box.class_id};
        segmentation->mask_width = source.mask_width;
        segmentation->mask_height = source.mask_height;
        segmentation->mask_bytes = source.mask.size();
    } else {
        return invalid_argument("Result is not a segmentation result");
    }
    return CVINFER_STATUS_OK;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_copy_mask(
    const cvinfer_result_t* result,
    size_t index,
    uint8_t* destination,
    size_t destination_bytes,
    size_t* copied_bytes) {
    clear_error();
    if (!result) return invalid_argument("result is required");
    const std::vector<std::uint8_t>* mask = nullptr;
    if (result->type == CVINFER_MODEL_YOLO11_SEG ||
        result->type == CVINFER_MODEL_YOLO26_SEG) {
        if (index >= result->segmentations.size()) return invalid_argument("Segmentation index is out of range");
        mask = &result->segmentations[index].mask;
    } else if (result->type == CVINFER_MODEL_RF_DETR_SEG) {
        if (index >= result->rf_detr_segmentations.size()) return invalid_argument("Segmentation index is out of range");
        mask = &result->rf_detr_segmentations[index].mask;
    } else {
        return invalid_argument("Result is not a segmentation result");
    }
    if (copied_bytes) *copied_bytes = mask->size();
    if (destination_bytes < mask->size()) {
        set_error("Destination mask buffer is too small");
        return CVINFER_STATUS_BUFFER_TOO_SMALL;
    }
    if (!mask->empty() && !destination) return invalid_argument("Destination mask buffer is required");
    if (!mask->empty()) std::memcpy(destination, mask->data(), mask->size());
    return CVINFER_STATUS_OK;
}

CVINFER_API cvinfer_status_t CVINFER_CALL cvinfer_result_get_timing(
    const cvinfer_result_t* result,
    cvinfer_timing_t* timing) {
    clear_error();
    if (!result || !timing) return invalid_argument("result and timing are required");
    timing->preprocess_ms = result->timing.preprocess_ms;
    timing->inference_ms = result->timing.inference_ms;
    timing->postprocess_ms = result->timing.postprocess_ms;
    timing->total_ms = result->timing.total_ms;
    return CVINFER_STATUS_OK;
}

} // extern "C"
