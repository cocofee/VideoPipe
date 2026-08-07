#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

#if defined(VP_WITH_YOLO11) && !defined(VP_YOLO11_POSTPROCESS_ONLY)
#include "../vp_primary_infer_node.h"
#endif

namespace vp_objects {
    class vp_frame_target;
}

namespace vp_nodes {

    struct vp_yolo11_letterbox {
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        int original_width = 0;
        int original_height = 0;
    };

    struct vp_yolo11_detection {
        cv::Rect box;
        int class_id = -1;
        float confidence = 0.0f;
    };

    std::vector<vp_yolo11_detection> decode_yolo11_output(
        const cv::Mat& output,
        const vp_yolo11_letterbox& letterbox,
        float confidence_threshold,
        float nms_threshold,
        int class_count_hint = 0);

    void require_yolo11_model_loaded(
        bool model_loaded,
        const std::string& model_path);

    std::shared_ptr<vp_objects::vp_frame_target> make_yolo11_frame_target(
        const vp_yolo11_detection& detection,
        int class_id_offset,
        int frame_index,
        int channel_index,
        const std::string& label);

#if defined(VP_WITH_YOLO11) && !defined(VP_YOLO11_POSTPROCESS_ONLY)
    class vp_yolo11_detector_node: public vp_primary_infer_node {
    private:
        float confidence_threshold;
        float nms_threshold;
        std::vector<vp_yolo11_letterbox> current_letterboxes;

    protected:
        virtual void preprocess(
            const std::vector<cv::Mat>& mats_to_infer,
            cv::Mat& blob_to_infer) override;
        virtual void postprocess(
            const std::vector<cv::Mat>& raw_outputs,
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    public:
        vp_yolo11_detector_node(
            std::string node_name,
            std::string model_path,
            std::string labels_path,
            int input_width = 640,
            int input_height = 640,
            float confidence_threshold = 0.25f,
            float nms_threshold = 0.45f);
        ~vp_yolo11_detector_node();
    };
#endif
}
