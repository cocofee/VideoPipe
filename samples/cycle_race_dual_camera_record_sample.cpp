#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include "../nodes/record/vp_indexed_file_des_node.h"
#include "../nodes/vp_rtsp_src_node.h"
#include "../utils/logger/vp_logger.h"

namespace {

    const char* state_name(vp_nodes::vp_indexed_record_state state) {
        switch (state) {
        case vp_nodes::vp_indexed_record_state::IDLE:
            return "idle";
        case vp_nodes::vp_indexed_record_state::RECORDING:
            return "recording";
        case vp_nodes::vp_indexed_record_state::DISK_LOW:
            return "disk-low";
        case vp_nodes::vp_indexed_record_state::INVALID_FRAME:
            return "invalid-frame";
        case vp_nodes::vp_indexed_record_state::OPEN_FAILED:
            return "open-failed";
        case vp_nodes::vp_indexed_record_state::WRITE_FAILED:
            return "write-failed";
        case vp_nodes::vp_indexed_record_state::INDEX_FAILED:
            return "index-failed";
        }
        return "unknown";
    }

    vp_nodes::vp_indexed_file_record_config make_config(
        std::string camera_id,
        const std::filesystem::path& save_dir) {
        vp_nodes::vp_indexed_file_record_config config;
        config.camera_id = std::move(camera_id);
        config.save_dir = save_dir;
        config.segment_duration_seconds = 5;
        config.bitrate = 4096;
        config.minimum_free_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        return config;
    }

}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: cycle_race_dual_camera_record_sample "
                     "<camera-a-rtsp> <camera-b-rtsp> <save-dir>" << std::endl;
        return 2;
    }

    const auto save_dir = std::filesystem::absolute(argv[3]).lexically_normal();
    std::filesystem::create_directories(save_dir);
    VP_SET_LOG_DIR((save_dir / "log").string());
    VP_LOGGER_INIT();

    auto frame_index_store = std::make_shared<vp_utils::vp_video_frame_index_store>(
        save_dir / "video-frame-index.jsonl");

    auto camera_a = std::make_shared<vp_nodes::vp_rtsp_src_node>(
        "finish_camera_a", 0, argv[1]);
    auto camera_b = std::make_shared<vp_nodes::vp_rtsp_src_node>(
        "finish_camera_b", 1, argv[2]);
    auto recorder_a = std::make_shared<vp_nodes::vp_indexed_file_des_node>(
        "finish_recorder_a", 0, make_config("camera-a", save_dir), frame_index_store);
    auto recorder_b = std::make_shared<vp_nodes::vp_indexed_file_des_node>(
        "finish_recorder_b", 1, make_config("camera-b", save_dir), frame_index_store);

    const auto report_status = [](const vp_nodes::vp_indexed_record_status& status) {
        std::cout << status.camera_id << " channel=" << status.channel_index
                  << " state=" << state_name(status.state)
                  << " path=" << status.video_path
                  << " message=" << status.message << std::endl;
    };
    recorder_a->set_status_hooker(report_status);
    recorder_b->set_status_hooker(report_status);
    recorder_a->attach_to({camera_a});
    recorder_b->attach_to({camera_b});

    camera_a->start();
    camera_b->start();
    std::cout << "recording; press Enter to stop" << std::endl;
    std::cin.get();

    camera_a->detach_recursively();
    camera_b->detach_recursively();
    return 0;
}
