#pragma once

#include <opencv2/videoio.hpp>

#include "vp_src_node.h"

namespace vp_nodes {
    // Camera source node for local USB/capture devices. Frames are normalized
    // to BGR8 before entering the pipeline.
    class vp_camera_src_node: public vp_src_node {
    private:
        cv::VideoCapture camera_capture;
        int device_index;
        int requested_width;
        int requested_height;
        int requested_fps;
        int effective_fps = 30;
        int skip_interval;
        int backend;

    protected:
        virtual void handle_run() override;

    public:
        vp_camera_src_node(std::string node_name,
                           int channel_index,
                           int device_index = 0,
                           float resize_ratio = 1.0,
                           int width = 0,
                           int height = 0,
                           int fps = 30,
                           int skip_interval = 0,
                           int backend = cv::CAP_ANY);
        ~vp_camera_src_node();

        virtual std::string to_string() override;
    };
}
