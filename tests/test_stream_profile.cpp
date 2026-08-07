#include <stdexcept>
#include <string>
#include <type_traits>

#include <opencv2/core.hpp>

#include "../nodes/vp_app_src_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../utils/logger/vp_logger.h"

namespace {
    void expect(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void test_h264_tcp_pipeline() {
        const vp_nodes::vp_rtsp_profile profile {};
        const auto pipeline = vp_nodes::build_rtsp_gst_pipeline(
            "rtsp://camera/live", profile, "avdec_h264");

        expect(pipeline.find("rtspsrc location=rtsp://camera/live") != std::string::npos,
               "RTSP URL is missing");
        expect(pipeline.find("protocols=tcp") != std::string::npos,
               "TCP transport is missing");
        expect(pipeline.find("latency=200") != std::string::npos,
               "default latency is missing");
        expect(pipeline.find("rtph264depay ! h264parse ! avdec_h264") != std::string::npos,
               "H264 pipeline is incomplete");
    }

    void test_h265_udp_pipeline() {
        const vp_nodes::vp_rtsp_profile profile {
            vp_nodes::vp_rtsp_codec::H265,
            vp_nodes::vp_rtsp_transport::UDP,
            80
        };
        const auto pipeline = vp_nodes::build_rtsp_gst_pipeline(
            "rtsp://camera/hevc", profile, "avdec_h265");

        expect(pipeline.find("protocols=udp") != std::string::npos,
               "UDP transport is missing");
        expect(pipeline.find("latency=80") != std::string::npos,
               "configured latency is missing");
        expect(pipeline.find("rtph265depay ! h265parse ! avdec_h265") != std::string::npos,
               "H265 pipeline is incomplete");
    }

    void test_negative_latency_is_rejected() {
        auto profile = vp_nodes::vp_rtsp_profile {};
        profile.latency_ms = -1;

        bool rejected = false;
        try {
            vp_nodes::build_rtsp_gst_pipeline("rtsp://camera/live", profile, "avdec_h264");
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "negative RTSP latency must be rejected");
    }

    void test_legacy_app_source_defaults_to_one_fps() {
        vp_nodes::vp_app_src_node source("legacy_app_source", 0);
        source.start();
        expect(source.push_frames({cv::Mat(2, 2, CV_8UC3)}),
               "legacy app source rejected a valid frame");
        expect(source.get_original_fps() == 1,
               "legacy app source must default to 1 FPS");
    }

    void test_app_source_uses_configured_fps() {
        vp_nodes::vp_app_src_node source("fast_app_source", 1, 60);
        source.start();
        expect(source.push_frames({cv::Mat(2, 2, CV_8UC3)}),
               "app source rejected a valid frame");
        expect(source.get_original_fps() == 60,
               "app source did not preserve configured FPS");
    }

    void test_invalid_app_source_fps_is_rejected() {
        for (const auto fps: {0, -1}) {
            bool rejected = false;
            try {
                vp_nodes::vp_app_src_node source("invalid_app_source", 2, fps);
            }
            catch (const std::invalid_argument&) {
                rejected = true;
            }
            expect(rejected, "non-positive app source FPS must be rejected");
        }
    }
}

static_assert(std::is_constructible_v<vp_nodes::vp_rtsp_src_node,
                                      std::string,
                                      int,
                                      std::string>,
              "legacy RTSP constructor defaults must remain available");

static_assert(std::is_constructible_v<vp_nodes::vp_rtsp_src_node,
                                      std::string,
                                      int,
                                      std::string,
                                      float,
                                      std::string,
                                      int>,
              "legacy RTSP constructor signature must remain available");

static_assert(!std::is_constructible_v<vp_nodes::vp_rtsp_src_node,
                                       std::string,
                                       int,
                                       std::string,
                                       vp_nodes::vp_rtsp_profile>,
              "RTSP profile constructor must require an explicit decoder");

static_assert(std::is_constructible_v<vp_nodes::vp_rtsp_src_node,
                                      std::string,
                                      int,
                                      std::string,
                                      vp_nodes::vp_rtsp_profile,
                                      std::string>,
              "RTSP profile constructor must accept an explicit decoder");

int main() {
    auto& logger = vp_utils::vp_logger::get_logger();
    logger.log_to_console = false;
    logger.log_to_file = false;
    logger.init();

    test_h264_tcp_pipeline();
    test_h265_udp_pipeline();
    test_negative_latency_is_rejected();
    test_legacy_app_source_defaults_to_one_fps();
    test_app_source_uses_configured_fps();
    test_invalid_app_source_fps_is_rejected();
    return 0;
}
