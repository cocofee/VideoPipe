
#include "vp_screen_des_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
    vp_screen_des_node::vp_screen_des_node(std::string node_name, 
                                            int channel_index, 
                                            bool osd,
                                            vp_objects::vp_size display_w_h):
                                            vp_des_node(node_name, channel_index),
                                            osd(osd),
                                            display_w_h(display_w_h) {
        this->gst_template = vp_utils::string_format(this->gst_template, node_name.c_str());
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_screen_des_node::~vp_screen_des_node() {
        deinitialized();
        screen_writer.release();
        if (opencv_window_enabled) {
            try {
                cv::destroyWindow(node_name);
            }
            catch (const cv::Exception&) {
                // GUI teardown is best effort.
            }
        }
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_screen_des_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            VP_DEBUG(vp_utils::string_format("[%s] received frame meta, channel_index=>%d, frame_index=>%d", node_name.c_str(), meta->channel_index, meta->frame_index));
            
            cv::Mat resize_frame;
            if (this->display_w_h.width != 0 && this->display_w_h.height != 0) {                 
                cv::resize((osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame, resize_frame, cv::Size(display_w_h.width, display_w_h.height));
            }
            else {
                resize_frame = (osd && !meta->osd_frame.empty()) ? meta->osd_frame : meta->frame;
            }

            if (resize_frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] received an empty frame, skipping display", node_name.c_str()));
                return vp_des_node::handle_frame_meta(meta);
            }

            if (!output_initialized && !output_disabled) {
                output_initialized = true;
                bool opened = false;
#if defined(_WIN32)
                // The default ximagesink pipeline is Linux-specific. On Windows,
                // bypass it instead of allowing an asynchronous GStreamer error.
                VP_INFO(vp_utils::string_format("[%s] using OpenCV window output on Windows",
                                                node_name.c_str()));
#else
                try {
                    opened = screen_writer.open(this->gst_template,
                                                cv::CAP_GSTREAMER,
                                                0,
                                                meta->fps,
                                                {resize_frame.cols, resize_frame.rows});
                }
                catch (const cv::Exception& e) {
                    VP_WARN(vp_utils::string_format("[%s] GStreamer display initialization failed: %s",
                                                    node_name.c_str(),
                                                    e.what()));
                }
#endif

                if (!opened) {
                    try {
                        cv::namedWindow(node_name, cv::WINDOW_NORMAL);
                        opencv_window_enabled = true;
                        VP_WARN(vp_utils::string_format("[%s] GStreamer display unavailable; using OpenCV window fallback",
                                                        node_name.c_str()));
                    }
                    catch (const cv::Exception& e) {
                        output_disabled = true;
                        VP_WARN(vp_utils::string_format("[%s] display disabled: %s",
                                                        node_name.c_str(),
                                                        e.what()));
                    }
                }
            }

            if (screen_writer.isOpened()) {
                screen_writer.write(resize_frame);
            }
            else if (opencv_window_enabled) {
                try {
                    cv::imshow(node_name, resize_frame);
                    cv::waitKey(1);
                }
                catch (const cv::Exception& e) {
                    opencv_window_enabled = false;
                    output_disabled = true;
                    VP_WARN(vp_utils::string_format("[%s] OpenCV display failed, disabling output: %s",
                                                    node_name.c_str(),
                                                    e.what()));
                }
            }

            // for general works defined in base class
            return vp_des_node::handle_frame_meta(meta);
    }

    // re-implementation, return nullptr.
    std::shared_ptr<vp_objects::vp_meta> 
        vp_screen_des_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
            // for general works defined in base class
            return vp_des_node::handle_control_meta(meta);
    }
}
