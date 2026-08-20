
#pragma once

#include <cstdint>

#include "vp_src_node.h"

namespace vp_nodes {
    // app src node, receive image data from external host code.
    class vp_app_src_node: public vp_src_node
    {
    private:
    protected:
        // just call vp_node::handle_run to ignore vp_src_node::handle_run
        virtual void handle_run() override;
        // just call vp_node::handle_frame_meta to ignore vp_src_node::handle_frame_meta
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override; 
        // just call vp_node::handle_control_meta to ignore vp_src_node::handle_control_meta
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override;

    public:
        vp_app_src_node(std::string node_name, 
                        int channel_index);
        ~vp_app_src_node();

        // push frames into pipeline
        // size of frame MUST be the same as the first time pushing to pipeline
        bool push_frames(std::vector<cv::Mat> frames);

        // Push frames with source timestamps supplied by the host capture API.
        // source_pts_us must be empty or have the same size as frames.
        bool push_frames(std::vector<cv::Mat> frames,
                         std::vector<std::int64_t> source_pts_us,
                         int source_session = 0);
    };
}
