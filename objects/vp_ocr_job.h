#pragma once

#if defined(VP_WITH_ASYNC_OCR) || defined(VP_ASYNC_OCR_CORE_ONLY)

#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

#include "shapes/vp_rect.h"

namespace vp_objects {
    struct vp_ocr_candidate {
        std::string text;
        float confidence = 0.0f;
    };

    class vp_ocr_job {
    private:
        const int channel_index_;
        const int frame_index_;
        const int track_id_;
        const std::int64_t timestamp_ms_;
        const double fps_;
        const cv::Mat crop_;
        const vp_rect source_box_;
        const std::string evidence_image_path_;
        const cv::Mat padded_crop_;

    public:
        vp_ocr_job(
            int channel_index,
            int frame_index,
            int track_id,
            std::int64_t timestamp_ms,
            double fps,
            const cv::Mat& crop,
            vp_rect source_box,
            std::string evidence_image_path,
            const cv::Mat& padded_crop = cv::Mat());

        int channel_index() const;
        int frame_index() const;
        int track_id() const;
        std::int64_t timestamp_ms() const;
        double fps() const;
        cv::Mat crop() const;
        const vp_rect& source_box() const;
        const std::string& evidence_image_path() const;
        cv::Mat padded_crop() const;
    };
}

#endif
