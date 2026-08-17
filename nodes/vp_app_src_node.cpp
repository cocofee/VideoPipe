#include "vp_app_src_node.h"

#include <stdexcept>

namespace vp_nodes {
    vp_app_src_node::vp_app_src_node(std::string node_name,
                        int channel_index):vp_app_src_node(node_name, channel_index, 30) {
    }

    vp_app_src_node::vp_app_src_node(std::string node_name,
                        int channel_index,
                        int source_fps):vp_src_node(node_name, channel_index, 1.0), source_fps(source_fps) {
        if (source_fps <= 0) {
            throw std::invalid_argument("app source FPS must be greater than zero");
        }
        this->initialized();
    }

    vp_app_src_node::~vp_app_src_node() {
        deinitialized();
    }

    // host code acts as previous node, call vp_node::meta_flow(...)
    bool vp_app_src_node::push_frames(std::vector<cv::Mat> frames) {
        // vp_app_src_node not working
        if (!gate.is_open()) {
            VP_WARN(vp_utils::string_format("[%s] is not working!", node_name.c_str()));
            return false;
        }
        
        if (frames.size() == 0) {
            return false;
        }

        // MUST have the same size
        auto size_warn = [this]() {
            VP_WARN(vp_utils::string_format("[%s] frames to be pushed MUST have the same size!", this->node_name.c_str()));
        };
        auto w = frames[0].cols;
        auto h = frames[0].rows;
        if (w <= 0 || h <= 0) {
            return false;
        }
        for (auto& f: frames) {
            if (f.empty() || f.cols != w || f.rows != h) {
                size_warn();
                return false;
            }
        }

        if (original_height != 0 && original_height != h) {
            size_warn();
            return false;
        }

        if (original_width != 0 && original_width != w) {
            size_warn();
            return false;
        }

        // initialize video properties
        if (original_width == 0 || original_height == 0 || original_fps == 0) {    
            original_width = w;
            original_height = h;
            original_fps = source_fps;
        }
        // stream_info_hooker activated if need
        vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
        invoke_stream_info_hooker(node_name, stream_info);

        for (auto& f: frames) {
            cv::Mat frame;
            if (!normalize_frame(f, frame)) {
                VP_WARN(vp_utils::string_format("[%s] unsupported frame format, channels==>%d, depth==>%d", node_name.c_str(), f.channels(), f.depth()));
                return false;
            }
            frame_index++;
            // create frame meta and meta flow like previous node
            auto in_meta = std::make_shared<vp_objects::vp_frame_meta>(frame, frame_index, channel_index, original_width, original_height, original_fps, true);

            vp_node::meta_flow(in_meta);
        }
        return true;
    }

    void vp_app_src_node::handle_run() {
        // call vp_node::handle_run() since we assume vp_app_src_node has virtual previous node (from host code)
        vp_node::handle_run();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_app_src_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        return vp_node::handle_frame_meta(meta);
    }

    std::shared_ptr<vp_objects::vp_meta> vp_app_src_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return vp_node::handle_control_meta(meta);
    }
}
