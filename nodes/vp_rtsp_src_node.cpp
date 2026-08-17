

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "vp_rtsp_src_node.h"
#include "../utils/vp_utils.h"

namespace vp_nodes {
    std::string build_rtsp_gst_pipeline(const std::string& url,
                                        const vp_rtsp_profile& profile,
                                        const std::string& decoder) {
        if (profile.latency_ms < 0) {
            throw std::invalid_argument("RTSP latency must not be negative");
        }

        const auto transport = profile.transport == vp_rtsp_transport::TCP ? "tcp" : "udp";
        const auto depay_and_parse = profile.codec == vp_rtsp_codec::H264
            ? "rtph264depay ! h264parse"
            : "rtph265depay ! h265parse";

        return "rtspsrc location=" + url +
               " protocols=" + transport +
               " latency=" + std::to_string(profile.latency_ms) +
               " ! application/x-rtp,media=video ! " + depay_and_parse +
               " ! " + decoder + " ! videoconvert ! appsink";
    }
        
    vp_rtsp_src_node::vp_rtsp_src_node(std::string node_name, 
                                        int channel_index, 
                                        std::string rtsp_url, 
                                        float resize_ratio,
                                        std::string gst_decoder_name,
                                        int skip_interval):
                                        vp_rtsp_src_node(node_name,
                                                         channel_index,
                                                         rtsp_url,
                                                         vp_rtsp_profile {},
                                                         gst_decoder_name,
                                                         resize_ratio,
                                                         skip_interval) {
    }

    vp_rtsp_src_node::vp_rtsp_src_node(std::string node_name,
                                        int channel_index,
                                        std::string rtsp_url,
                                        vp_rtsp_profile profile,
                                        std::string gst_decoder_name,
                                        float resize_ratio,
                                        int skip_interval):
                                        vp_src_node(node_name, channel_index, resize_ratio),
                                        rtsp_url(rtsp_url), gst_decoder_name(gst_decoder_name), skip_interval(skip_interval) {
        assert(skip_interval >= 0 && skip_interval <= 9);
        auto decoder = gst_decoder_name;
        if (profile.codec == vp_rtsp_codec::H265 && decoder == "avdec_h264") {
            decoder = "avdec_h265";
        }
        this->gst_template = build_rtsp_gst_pipeline(rtsp_url, profile, decoder);
        VP_INFO(vp_utils::string_format("[%s] [%s]", node_name.c_str(), gst_template.c_str()));
        this->initialized();
    }
    
    vp_rtsp_src_node::~vp_rtsp_src_node() {
        deinitialized();
    }
    
    // define how to read video from rtsp stream, create frame meta etc.
    // please refer to the implementation of vp_node::handle_run.
    void vp_rtsp_src_node::handle_run() {
        cv::Mat frame;
        int video_width = 0;
        int video_height = 0;
        int fps = 0;
        int skip = 0;
        bool stream_info_sent = false;
        while(alive) {
            // check if need work
            gate.knock();
            
            // try to open capture
            if (!rtsp_capture.isOpened()) {
                video_width = video_height = fps = 0;
                original_width = original_height = original_fps = 0;
                stream_info_sent = false;
                if (!rtsp_capture.open(this->gst_template, cv::CAP_GSTREAMER)) {
                    VP_WARN(vp_utils::string_format("[%s] open rtsp failed, try again...", node_name.c_str()));
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }
            }

            // video properties
            if (video_width == 0 || video_height == 0 || fps == 0) {
                video_width = rtsp_capture.get(cv::CAP_PROP_FRAME_WIDTH);
                video_height = rtsp_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
                const auto stream_fps = rtsp_capture.get(cv::CAP_PROP_FPS);
                fps = stream_fps > 1.0 ? static_cast<int>(stream_fps + 0.5) : 30;
                
                original_fps = fps;
                original_width = video_width;
                original_height = video_height;

                // set true fps because skip some frames
                fps = std::max(1, fps / (skip_interval + 1));
            }
            rtsp_capture >> frame;
            if(frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] reading frame empty, total frame==>%d", node_name.c_str(), frame_index));
                rtsp_capture.release();
                video_width = video_height = fps = 0;
                original_width = original_height = original_fps = 0;
                stream_info_sent = false;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (video_width <= 0 || video_height <= 0) {
                video_width = frame.cols;
                video_height = frame.rows;
                original_width = video_width;
                original_height = video_height;
            }
            if (original_width <= 0 || original_height <= 0) {
                original_width = frame.cols;
                original_height = frame.rows;
            }
            if (!stream_info_sent) {
                vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
                invoke_stream_info_hooker(node_name, stream_info);
                stream_info_sent = true;
            }

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
                std::make_shared<vp_objects::vp_frame_meta>(resize_frame, this->frame_index, this->channel_index, video_width, video_height, fps, true);

            if (out_meta != nullptr) {
                this->out_queue.push(out_meta);
                
                // handled hooker activated if need
                if (this->meta_handled_hooker) {
                    meta_handled_hooker(node_name, out_queue.size(), out_meta);
                }

                // important! notify consumer of out_queue in case it is waiting.
                this->out_queue_semaphore.signal();
                VP_DEBUG(vp_utils::string_format("[%s] after handling meta, out_queue.size()==>%d", node_name.c_str(), out_queue.size()));
            }  
        }

        // send dead flag for dispatch_thread
        this->out_queue.push(nullptr);
        this->out_queue_semaphore.signal();    
    }

    // return stream url
    std::string vp_rtsp_src_node::to_string() {
        return rtsp_url;
    }
}
