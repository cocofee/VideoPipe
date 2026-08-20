

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <thread>

#include "vp_rtmp_src_node.h"
#include "../utils/vp_utils.h"
#include "../utils/race/vp_video_time.h"

namespace vp_nodes {

    namespace {
        constexpr int MAX_CONSECUTIVE_EMPTY_FRAMES = 30;
        constexpr auto RECONNECT_BACKOFF = std::chrono::milliseconds(200);
    }
        
    vp_rtmp_src_node::vp_rtmp_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string rtmp_url, 
                                        float resize_ratio,
                                        std::string gst_decoder_name,
                                        int skip_interval): 
                                        vp_src_node(node_name, channel_index, resize_ratio),
                                        rtmp_url(rtmp_url), gst_decoder_name(gst_decoder_name), skip_interval(skip_interval) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        this->gst_template = vp_utils::string_format(this->gst_template, rtmp_url.c_str(), gst_decoder_name.c_str());
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_rtmp_src_node::~vp_rtmp_src_node() {
        deinitialized();
    }
    
    // define how to read video from rtmp stream, create frame meta etc.
    // please refer to the implementation of vp_node::handle_run.
    void vp_rtmp_src_node::handle_run() {
        cv::Mat frame;
        int video_width = 0;
        int video_height = 0;
        int fps = 0;
        int skip = 0;
        int consecutive_empty_frames = 0;
        while(alive) {
            // check if need work
            gate.knock();
            
            // try to open capture
            if (!rtmp_capture.isOpened()) {
                video_width = video_height = fps = 0;
                original_width = original_height = original_fps = 0;
                if (!rtmp_capture.open(this->gst_template, cv::CAP_GSTREAMER)) {
                    VP_WARN(vp_utils::string_format("[%s] open rtmp failed, try again...", node_name.c_str()));
                    std::this_thread::sleep_for(RECONNECT_BACKOFF);
                    continue;
                }
                consecutive_empty_frames = 0;
                source_session++;
            }

            // video properties
            if (video_width == 0 || video_height == 0 || fps == 0) {
                video_width = rtmp_capture.get(cv::CAP_PROP_FRAME_WIDTH);
                video_height = rtmp_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                fps = rtmp_capture.get(cv::CAP_PROP_FPS);
                
                original_fps = fps;
                original_width = video_width;
                original_height = video_height;

                // set true fps because skip some frames
                fps = fps / (skip_interval + 1);
            }
            // stream_info_hooker activated if need
            vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
            invoke_stream_info_hooker(node_name, stream_info);

            rtmp_capture >> frame;
            if(frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] reading frame empty, total frame==>%d", node_name.c_str(), frame_index));
                if (++consecutive_empty_frames >= MAX_CONSECUTIVE_EMPTY_FRAMES) {
                    VP_WARN(vp_utils::string_format("[%s] too many empty frames, reconnecting...", node_name.c_str()));
                    rtmp_capture.release();
                    consecutive_empty_frames = 0;
                    std::this_thread::sleep_for(RECONNECT_BACKOFF);
                }
                continue;
            }
            consecutive_empty_frames = 0;

            const auto capture_monotonic_us = vp_utils::monotonic_time_us();
            const auto capture_wall_time_ms = vp_utils::wall_time_ms();

            // need skip
            if (skip < skip_interval) {
                skip++;
                continue;
            }
            skip = 0;

            cv::Mat resize_frame;
            if (this->resize_ratio != 1.0f) {                 
                cv::resize(frame, resize_frame, cv::Size(), resize_ratio, resize_ratio);
            }
            else {
                resize_frame = frame.clone(); // clone!;
            }
            // set true size because resize
            video_width = resize_frame.cols;
            video_height = resize_frame.rows;
            
            this->frame_index++;
            // create frame meta
            auto out_meta = 
                std::make_shared<vp_objects::vp_frame_meta>(resize_frame,
                                                            this->frame_index,
                                                            this->channel_index,
                                                            video_width,
                                                            video_height,
                                                            fps,
                                                            -1,
                                                            source_session,
                                                            capture_monotonic_us,
                                                            capture_wall_time_ms);

            if (out_meta != nullptr) {
                pendding_meta(out_meta);
            }  
        }

        // send dead flag for dispatch_thread
        pendding_meta(nullptr);
    }

    // return stream url
    std::string vp_rtmp_src_node::to_string() {
        return rtmp_url;
    }
}
