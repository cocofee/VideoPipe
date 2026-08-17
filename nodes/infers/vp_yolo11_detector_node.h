#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "../vp_primary_infer_node.h"

namespace vp_nodes {
    struct vp_yolo11_letterbox {
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        int original_width = 0;
        int original_height = 0;
    };

    class vp_yolo11_detector_node final: public vp_primary_infer_node {
    private:
        float confidence_threshold;
        float nms_threshold;
        std::vector<vp_yolo11_letterbox> current_letterboxes;

        static void validate_thresholds(float confidence_threshold, float nms_threshold);

    protected:
        virtual void preprocess(const std::vector<cv::Mat>& mats_to_infer,
                                cv::Mat& blob_to_infer) override;
        virtual void postprocess(
            const std::vector<cv::Mat>& raw_outputs,
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    public:
        vp_yolo11_detector_node(std::string node_name,
                                std::string model_path,
                                std::string labels_path = "",
                                int input_width = 640,
                                int input_height = 640,
                                float confidence_threshold = 0.25f,
                                float nms_threshold = 0.45f);
        ~vp_yolo11_detector_node();
    };
}
