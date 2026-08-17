#include "vp_camera_src_node.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <thread>

#include "../utils/vp_utils.h"
#include "../utils/logger/vp_logger.h"

namespace vp_nodes {
    vp_camera_src_node::vp_camera_src_node(std::string node_name,
                                           int channel_index,
                                           int device_index,
                                           float resize_ratio,
                                           int width,
                                           int height,
                                           int fps,
                                           int skip_interval,
                                           int backend):
                                           vp_src_node(node_name, channel_index, resize_ratio),
                                           device_index(device_index),
                                           requested_width(width),
                                           requested_height(height),
                                           requested_fps(fps),
                                           skip_interval(skip_interval),
                                           backend(backend) {
        assert(device_index >= 0);
        assert(width >= 0 && height >= 0);
        assert(fps > 0);
        assert(skip_interval >= 0 && skip_interval <= 9);
        this->initialized();
    }

    vp_camera_src_node::~vp_camera_src_node() {
        deinitialized();
    }

    void vp_camera_src_node::handle_run() {
        cv::Mat frame;
        int skip = 0;
        bool stream_info_sent = false;

        while (alive) {
            gate.knock();

            if (!camera_capture.isOpened()) {
                stream_info_sent = false;
                original_width = original_height = 0;
                original_fps = -1;

                const bool opened = backend == cv::CAP_ANY
                    ? camera_capture.open(device_index)
                    : camera_capture.open(device_index, backend);
                if (!opened) {
                    VP_WARN(vp_utils::string_format("[%s] open camera %d failed, retrying...", node_name.c_str(), device_index));
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }

                if (requested_width > 0) {
                    camera_capture.set(cv::CAP_PROP_FRAME_WIDTH, requested_width);
                }
                if (requested_height > 0) {
                    camera_capture.set(cv::CAP_PROP_FRAME_HEIGHT, requested_height);
                }
                if (requested_fps > 0) {
                    camera_capture.set(cv::CAP_PROP_FPS, requested_fps);
                }
            }

            if (!camera_capture.read(frame) || frame.empty()) {
                VP_WARN(vp_utils::string_format("[%s] camera %d returned an empty frame, reopening...", node_name.c_str(), device_index));
                camera_capture.release();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            if (skip < skip_interval) {
                ++skip;
                continue;
            }
            skip = 0;

            cv::Mat normalized_frame;
            if (!normalize_frame(frame, normalized_frame)) {
                VP_WARN(vp_utils::string_format("[%s] unsupported camera frame format, channels==>%d, depth==>%d", node_name.c_str(), frame.channels(), frame.depth()));
                continue;
            }

            if (original_width == 0 || original_height == 0) {
                original_width = normalized_frame.cols;
                original_height = normalized_frame.rows;
                const double camera_fps = camera_capture.get(cv::CAP_PROP_FPS);
                effective_fps = camera_fps > 1.0 ? static_cast<int>(camera_fps + 0.5) : requested_fps;
                effective_fps = std::max(effective_fps, 1);
                original_fps = effective_fps;
            }

            vp_stream_info stream_info {channel_index, original_fps, original_width, original_height, to_string()};
            if (!stream_info_sent) {
                invoke_stream_info_hooker(node_name, stream_info);
                stream_info_sent = true;
            }

            cv::Mat output_frame;
            if (resize_ratio != 1.0f) {
                cv::resize(normalized_frame, output_frame, cv::Size(), resize_ratio, resize_ratio);
            }
            else {
                output_frame = normalized_frame;
            }

            ++frame_index;
            auto out_meta = std::make_shared<vp_objects::vp_frame_meta>(
                output_frame, frame_index, channel_index,
                output_frame.cols, output_frame.rows,
                std::max(1, effective_fps / (skip_interval + 1)), true);
            out_queue.push(out_meta);
            if (meta_handled_hooker) {
                meta_handled_hooker(node_name, out_queue.size(), out_meta);
            }
            out_queue_semaphore.signal();
        }

        out_queue.push(nullptr);
        out_queue_semaphore.signal();
    }

    std::string vp_camera_src_node::to_string() {
        return "camera://" + std::to_string(device_index);
    }
}
