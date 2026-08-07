#include "vp_ocr_job.h"

#if defined(VP_WITH_ASYNC_OCR) || defined(VP_ASYNC_OCR_CORE_ONLY)

#include <cmath>
#include <stdexcept>
#include <utility>

namespace vp_objects {
    vp_ocr_job::vp_ocr_job(
        int channel_index,
        int frame_index,
        int track_id,
        std::int64_t timestamp_ms,
        double fps,
        const cv::Mat& crop,
        vp_rect source_box,
        std::string evidence_image_path,
        const cv::Mat& padded_crop):
        channel_index_(channel_index),
        frame_index_(frame_index),
        track_id_(track_id),
        timestamp_ms_(timestamp_ms),
        fps_(fps),
        crop_(crop.clone()),
        source_box_(source_box),
        evidence_image_path_(std::move(evidence_image_path)),
        padded_crop_(padded_crop.clone()) {
        if (channel_index < 0 || frame_index < 0 || track_id < 0 || timestamp_ms < 0) {
            throw std::invalid_argument("OCR job indices and timestamp must be non-negative");
        }
        if (!std::isfinite(fps) || fps <= 0.0) {
            throw std::invalid_argument("OCR job FPS must be positive and finite");
        }
        if (crop.empty()) {
            throw std::invalid_argument("OCR job crop must not be empty");
        }
        if (source_box.width <= 0 || source_box.height <= 0) {
            throw std::invalid_argument("OCR job source box must have a positive size");
        }
    }

    int vp_ocr_job::channel_index() const {
        return channel_index_;
    }

    int vp_ocr_job::frame_index() const {
        return frame_index_;
    }

    int vp_ocr_job::track_id() const {
        return track_id_;
    }

    std::int64_t vp_ocr_job::timestamp_ms() const {
        return timestamp_ms_;
    }

    double vp_ocr_job::fps() const {
        return fps_;
    }

    cv::Mat vp_ocr_job::crop() const {
        return crop_.clone();
    }

    const vp_rect& vp_ocr_job::source_box() const {
        return source_box_;
    }

    const std::string& vp_ocr_job::evidence_image_path() const {
        return evidence_image_path_;
    }

    cv::Mat vp_ocr_job::padded_crop() const {
        return padded_crop_.clone();
    }
}

#endif
