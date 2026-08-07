#pragma once

#include <string>

#include "vp_src_node.h"

namespace vp_nodes {
    enum class vp_rtsp_codec {
        H264,
        H265
    };

    enum class vp_rtsp_transport {
        TCP,
        UDP
    };

    struct vp_rtsp_profile {
        vp_rtsp_codec codec = vp_rtsp_codec::H264;
        vp_rtsp_transport transport = vp_rtsp_transport::TCP;
        int latency_ms = 200;
    };

    std::string build_rtsp_gst_pipeline(const std::string& url,
                                        const vp_rtsp_profile& profile,
                                        const std::string& decoder);

    // rtsp source node, receive video stream via rtsp protocal.
    // example:
    // rtsp://admin:admin12345@192.168.77.110:554/
    class vp_rtsp_src_node: public vp_src_node {
    private:
        /* data */
        std::string gst_template;
        cv::VideoCapture rtsp_capture;
    protected:
        // re-implemetation
        virtual void handle_run() override;
    public:
        vp_rtsp_src_node(std::string node_name, 
                        int channel_index, 
                        std::string rtsp_url, 
                        float resize_ratio = 1.0,
                        std::string gst_decoder_name = "avdec_h264",
                        int skip_interval = 0);
        vp_rtsp_src_node(std::string node_name,
                        int channel_index,
                        std::string rtsp_url,
                        vp_rtsp_profile profile,
                        std::string gst_decoder_name,
                        float resize_ratio = 1.0,
                        int skip_interval = 0);
        ~vp_rtsp_src_node();

        virtual std::string to_string() override;

        std::string rtsp_url;
        // set avdec_h264 as the default decoder, we can use hardware decoder instead.
        std::string gst_decoder_name = "avdec_h264";
        // 0 means no skip
        int skip_interval = 0;
    };
}
