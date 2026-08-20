#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "../vp_des_node.h"
#include "../../utils/race/vp_video_frame_index_writer.h"

namespace vp_nodes {

    enum class vp_indexed_record_state {
        IDLE,
        RECORDING,
        DISK_LOW,
        INVALID_FRAME,
        OPEN_FAILED,
        WRITE_FAILED,
        INDEX_FAILED,
    };

    struct vp_indexed_record_status {
        vp_indexed_record_state state = vp_indexed_record_state::IDLE;
        std::string camera_id;
        int channel_index = -1;
        std::string video_path;
        std::uintmax_t available_bytes = 0;
        std::string message;
    };

    using vp_indexed_record_status_hooker =
        std::function<void(const vp_indexed_record_status&)>;

    struct vp_indexed_file_record_config {
        std::string camera_id;
        std::filesystem::path save_dir;
        int segment_duration_seconds = 5;
        vp_objects::vp_size resolution_w_h = {};
        int bitrate = 4096;
        bool osd = false;
        std::string gst_encoder_name = "x264enc";
        std::uintmax_t minimum_free_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
        std::int64_t index_interval_ms = 250;
    };

    // One instance records one camera/channel. Use separate instances for dual cameras and
    // share the same vp_video_frame_index_store between them.
    class vp_indexed_file_des_node: public vp_des_node {
    private:
        vp_indexed_file_record_config config;
        std::shared_ptr<vp_utils::vp_video_frame_index_store> frame_index_store;
        cv::VideoWriter video_writer;

        std::filesystem::path current_video_path;
        std::filesystem::path current_work_path;
        std::int64_t current_video_frame_index = -1;
        std::int64_t segment_started_monotonic_us = -1;
        std::int64_t last_checkpoint_monotonic_us = -1;
        std::int64_t last_space_check_monotonic_us = -1;
        std::uint64_t segment_sequence = 0;
        int current_source_session = -1;
        int current_width = 0;
        int current_height = 0;
        int current_fps = 0;
        std::vector<vp_utils::vp_video_frame_index_entry> current_segment_checkpoints;
        std::vector<vp_utils::vp_video_frame_index_entry> deferred_index_entries;

        mutable std::mutex status_lock;
        vp_indexed_record_status current_status;
        vp_indexed_record_status_hooker status_hooker;

        bool should_rotate(const vp_objects::vp_frame_meta& meta, const cv::Mat& frame) const;
        bool open_segment(const vp_objects::vp_frame_meta& meta, const cv::Mat& frame);
        bool close_segment();
        bool verify_and_publish_segment();
        void collect_checkpoint(const vp_objects::vp_frame_meta& meta);
        bool check_available_space(std::uintmax_t& available_bytes) const;
        std::filesystem::path make_segment_path(const vp_objects::vp_frame_meta& meta);
        void update_status(
            vp_indexed_record_state state,
            std::string message,
            std::uintmax_t available_bytes = 0);

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(
            std::shared_ptr<vp_objects::vp_frame_meta> meta) override;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(
            std::shared_ptr<vp_objects::vp_control_meta> meta) override;

    public:
        vp_indexed_file_des_node(
            std::string node_name,
            int channel_index,
            vp_indexed_file_record_config config,
            std::shared_ptr<vp_utils::vp_video_frame_index_store> frame_index_store);
        ~vp_indexed_file_des_node();

        void set_status_hooker(vp_indexed_record_status_hooker hooker);
        vp_indexed_record_status get_record_status() const;
    };

}
