#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>

#include <opencv2/videoio.hpp>

namespace vp_utils {

    inline std::int64_t monotonic_time_us() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    inline std::int64_t wall_time_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // OpenCV exposes backend source position in milliseconds. Some live backends do not
    // support it; callers must preserve -1 rather than deriving a fake PTS from frame rate.
    inline std::int64_t capture_source_pts_us(cv::VideoCapture& capture) {
        const auto position_ms = capture.get(cv::CAP_PROP_POS_MSEC);
        if (!std::isfinite(position_ms) || position_ms < 0) {
            return -1;
        }
        return static_cast<std::int64_t>(std::llround(position_ms * 1000.0));
    }

}
