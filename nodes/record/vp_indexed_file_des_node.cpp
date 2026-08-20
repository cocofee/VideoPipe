#include "vp_indexed_file_des_node.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vp_nodes {

    namespace {

        std::string sanitize_path_part(std::string value) {
            std::replace_if(
                value.begin(),
                value.end(),
                [](unsigned char character) {
                    return !std::isalnum(character) && character != '-' && character != '_';
                },
                '_');
            return value;
        }

    }

    vp_indexed_file_des_node::vp_indexed_file_des_node(
        std::string node_name,
        int channel_index,
        vp_indexed_file_record_config config,
        std::shared_ptr<vp_utils::vp_video_frame_index_store> frame_index_store):
        vp_des_node(std::move(node_name), channel_index),
        config(std::move(config)),
        frame_index_store(std::move(frame_index_store)) {
        if (this->config.camera_id.empty()) {
            throw std::invalid_argument("indexed recorder camera_id is required");
        }
        if (this->config.save_dir.empty()) {
            throw std::invalid_argument("indexed recorder save_dir is required");
        }
        if (this->config.segment_duration_seconds <= 0) {
            throw std::invalid_argument("indexed recorder segment duration must be positive");
        }
        if (this->config.bitrate <= 0) {
            throw std::invalid_argument("indexed recorder bitrate must be positive");
        }
        if (channel_index < 0) {
            throw std::invalid_argument("indexed recorder channel_index must not be negative");
        }
        if (!this->frame_index_store) {
            throw std::invalid_argument("indexed recorder frame index store is required");
        }

        this->config.save_dir = std::filesystem::absolute(this->config.save_dir).lexically_normal();
        current_status.camera_id = this->config.camera_id;
        current_status.channel_index = channel_index;
        initialized();
    }

    vp_indexed_file_des_node::~vp_indexed_file_des_node() {
        deinitialized();
        close_segment();
    }

    bool vp_indexed_file_des_node::should_rotate(
        const vp_objects::vp_frame_meta& meta,
        const cv::Mat& frame) const {
        if (!video_writer.isOpened()) {
            return true;
        }
        if (meta.source_session != current_source_session ||
            frame.cols != current_width ||
            frame.rows != current_height ||
            meta.fps != current_fps) {
            return true;
        }
        if (meta.capture_monotonic_us < segment_started_monotonic_us) {
            return true;
        }
        const auto segment_duration_us =
            static_cast<std::int64_t>(config.segment_duration_seconds) * 1000000;
        return meta.capture_monotonic_us - segment_started_monotonic_us >= segment_duration_us;
    }

    std::filesystem::path vp_indexed_file_des_node::make_segment_path(
        const vp_objects::vp_frame_meta& meta) {
        const auto camera_id = sanitize_path_part(config.camera_id);
        const auto output_dir = config.save_dir / camera_id / std::to_string(channel_index);
        std::filesystem::create_directories(output_dir);

        std::filesystem::path candidate;
        do {
            std::ostringstream file_name;
            file_name << camera_id << '_' << meta.capture_wall_time_ms << '_'
                      << ++segment_sequence << ".mp4";
            candidate = output_dir / file_name.str();
        } while (std::filesystem::exists(candidate) ||
                 std::filesystem::exists(
                     candidate.parent_path() / (candidate.stem().string() + ".partial.mp4")));
        return candidate;
    }

    bool vp_indexed_file_des_node::open_segment(
        const vp_objects::vp_frame_meta& meta,
        const cv::Mat& frame) {
        close_segment();

        std::uintmax_t available_bytes = 0;
        try {
            std::filesystem::create_directories(config.save_dir);
            available_bytes = std::filesystem::space(config.save_dir).available;
        }
        catch (const std::exception& error) {
            update_status(vp_indexed_record_state::OPEN_FAILED, error.what());
            return false;
        }
        if (available_bytes < config.minimum_free_bytes) {
            update_status(
                vp_indexed_record_state::DISK_LOW,
                "available disk space is below the recording threshold",
                available_bytes);
            return false;
        }

        try {
            current_video_path = make_segment_path(meta);
            current_work_path = current_video_path.parent_path() /
                (current_video_path.stem().string() + ".partial.mp4");
            const auto output_path = current_work_path.generic_string();
            if (output_path.find('"') != std::string::npos) {
                update_status(vp_indexed_record_state::OPEN_FAILED, "video path contains an unsupported quote");
                current_video_path.clear();
                current_work_path.clear();
                return false;
            }

            std::ostringstream pipeline;
            pipeline << "appsrc ! videoconvert ! " << config.gst_encoder_name
                     << " bitrate=" << config.bitrate
                     << " ! h264parse ! mp4mux ! filesink location=\"" << output_path << "\"";
            if (!video_writer.open(
                    pipeline.str(),
                    cv::CAP_GSTREAMER,
                    0,
                    meta.fps,
                    cv::Size(frame.cols, frame.rows))) {
                update_status(vp_indexed_record_state::OPEN_FAILED, "failed to open GStreamer video writer");
                current_video_path.clear();
                current_work_path.clear();
                return false;
            }
        }
        catch (const cv::Exception& error) {
            update_status(vp_indexed_record_state::OPEN_FAILED, error.what());
            current_video_path.clear();
            current_work_path.clear();
            return false;
        }
        catch (const std::exception& error) {
            update_status(vp_indexed_record_state::OPEN_FAILED, error.what());
            current_video_path.clear();
            current_work_path.clear();
            return false;
        }

        current_video_frame_index = -1;
        segment_started_monotonic_us = meta.capture_monotonic_us;
        last_checkpoint_monotonic_us = -1;
        last_space_check_monotonic_us = meta.capture_monotonic_us;
        current_source_session = meta.source_session;
        current_width = frame.cols;
        current_height = frame.rows;
        current_fps = meta.fps;
        current_segment_checkpoints.clear();
        update_status(vp_indexed_record_state::RECORDING, "recording", available_bytes);
        return true;
    }

    bool vp_indexed_file_des_node::verify_and_publish_segment() {
        if (current_work_path.empty() || current_video_path.empty() ||
            current_video_frame_index < 0) {
            update_status(vp_indexed_record_state::WRITE_FAILED, "recorded segment file is missing or empty");
            return false;
        }

        std::error_code exists_error;
        if (!std::filesystem::exists(current_work_path, exists_error) || exists_error) {
            update_status(vp_indexed_record_state::WRITE_FAILED, "recorded segment file is missing or inaccessible");
            return false;
        }

        std::error_code size_error;
        const auto file_size = std::filesystem::file_size(current_work_path, size_error);
        if (size_error || file_size == 0) {
            update_status(vp_indexed_record_state::WRITE_FAILED, "recorded segment file is empty");
            return false;
        }

        cv::VideoCapture verification(current_work_path.string());
        cv::Mat first_frame;
        if (!verification.isOpened() || !verification.read(first_frame) || first_frame.empty()) {
            update_status(vp_indexed_record_state::WRITE_FAILED, "recorded segment is not readable after finalization");
            return false;
        }
        const auto expected_frame_count = current_video_frame_index + 1;
        const auto reported_frame_count = verification.get(cv::CAP_PROP_FRAME_COUNT);
        if (std::isfinite(reported_frame_count) && reported_frame_count > 0 &&
            static_cast<std::int64_t>(std::llround(reported_frame_count)) < expected_frame_count) {
            update_status(vp_indexed_record_state::WRITE_FAILED, "recorded segment contains fewer frames than expected");
            return false;
        }
        verification.release();

        std::error_code rename_error;
        std::filesystem::rename(current_work_path, current_video_path, rename_error);
        if (rename_error) {
            update_status(vp_indexed_record_state::WRITE_FAILED, rename_error.message());
            return false;
        }
        return true;
    }

    bool vp_indexed_file_des_node::close_segment() {
        const auto had_segment = video_writer.isOpened() || !current_work_path.empty();
        if (video_writer.isOpened()) {
            video_writer.release();
        }
        auto finalized = !had_segment;
        if (had_segment) {
            bool published = false;
            try {
                published = verify_and_publish_segment();
            }
            catch (const cv::Exception& error) {
                update_status(vp_indexed_record_state::WRITE_FAILED, error.what());
            }
            catch (const std::exception& error) {
                update_status(vp_indexed_record_state::WRITE_FAILED, error.what());
            }
            if (published) {
                finalized = true;
                deferred_index_entries.insert(
                    deferred_index_entries.end(),
                    current_segment_checkpoints.begin(),
                    current_segment_checkpoints.end());
            }
        }
        if (!deferred_index_entries.empty()) {
            try {
                frame_index_store->append_batch(deferred_index_entries);
                deferred_index_entries.clear();
            }
            catch (const std::exception& error) {
                update_status(vp_indexed_record_state::INDEX_FAILED, error.what());
                finalized = false;
            }
        }
        current_video_frame_index = -1;
        segment_started_monotonic_us = -1;
        last_checkpoint_monotonic_us = -1;
        last_space_check_monotonic_us = -1;
        current_source_session = -1;
        current_width = 0;
        current_height = 0;
        current_fps = 0;
        current_segment_checkpoints.clear();
        current_video_path.clear();
        current_work_path.clear();
        return finalized;
    }

    void vp_indexed_file_des_node::collect_checkpoint(
        const vp_objects::vp_frame_meta& meta) {
        if (!vp_utils::is_video_frame_checkpoint_due(
                current_video_path.string(),
                current_source_session,
                last_checkpoint_monotonic_us,
                current_video_path.string(),
                meta.source_session,
                meta.capture_monotonic_us,
                config.index_interval_ms)) {
            return;
        }

        vp_utils::vp_video_frame_index_entry entry;
        entry.camera_id = config.camera_id;
        entry.channel_index = meta.channel_index;
        entry.source_session = meta.source_session;
        entry.source_pts_us = meta.source_pts_us;
        entry.capture_monotonic_us = meta.capture_monotonic_us;
        entry.capture_wall_time_ms = meta.capture_wall_time_ms;
        entry.frame_index = meta.frame_index;
        entry.video_path = current_video_path.string();
        entry.video_frame_index = current_video_frame_index;
        entry.entry_id = vp_utils::make_video_frame_index_entry_id(entry);
        current_segment_checkpoints.push_back(std::move(entry));
        last_checkpoint_monotonic_us = meta.capture_monotonic_us;
    }

    bool vp_indexed_file_des_node::check_available_space(
        std::uintmax_t& available_bytes) const {
        try {
            available_bytes = std::filesystem::space(config.save_dir).available;
            return true;
        }
        catch (...) {
            available_bytes = 0;
            return false;
        }
    }

    void vp_indexed_file_des_node::update_status(
        vp_indexed_record_state state,
        std::string message,
        std::uintmax_t available_bytes) {
        vp_indexed_record_status_hooker hooker;
        vp_indexed_record_status status;
        {
            std::lock_guard<std::mutex> guard(status_lock);
            current_status.state = state;
            current_status.video_path = current_video_path.string();
            current_status.available_bytes = available_bytes;
            current_status.message = std::move(message);
            status = current_status;
            hooker = status_hooker;
        }
        if (hooker) {
            try {
                hooker(status);
            }
            catch (...) {
                // A UI/status observer must not terminate the recording thread.
            }
        }
    }

    std::shared_ptr<vp_objects::vp_meta> vp_indexed_file_des_node::handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        if (!meta || meta->channel_index != channel_index) {
            return nullptr;
        }

        const auto& source_frame = config.osd && !meta->osd_frame.empty()
            ? meta->osd_frame
            : meta->frame;
        if (source_frame.empty() || meta->fps <= 0 || meta->capture_monotonic_us < 0 ||
            meta->capture_wall_time_ms < 0 || meta->source_session < 0) {
            update_status(vp_indexed_record_state::INVALID_FRAME, "frame timing or video properties are invalid");
            return vp_des_node::handle_frame_meta(meta);
        }

        cv::Mat frame;
        try {
            if (config.resolution_w_h.width > 0 && config.resolution_w_h.height > 0) {
                cv::resize(
                    source_frame,
                    frame,
                    cv::Size(config.resolution_w_h.width, config.resolution_w_h.height));
            }
            else {
                frame = source_frame;
            }
        }
        catch (const cv::Exception& error) {
            update_status(vp_indexed_record_state::INVALID_FRAME, error.what());
            return vp_des_node::handle_frame_meta(meta);
        }

        if (should_rotate(*meta, frame) && !open_segment(*meta, frame)) {
            return vp_des_node::handle_frame_meta(meta);
        }

        if (meta->capture_monotonic_us - last_space_check_monotonic_us >= 1000000) {
            std::uintmax_t available_bytes = 0;
            last_space_check_monotonic_us = meta->capture_monotonic_us;
            if (!check_available_space(available_bytes)) {
                update_status(vp_indexed_record_state::WRITE_FAILED, "failed to query available disk space");
                close_segment();
                return vp_des_node::handle_frame_meta(meta);
            }
            if (available_bytes < config.minimum_free_bytes) {
                update_status(
                    vp_indexed_record_state::DISK_LOW,
                    "available disk space is below the recording threshold",
                    available_bytes);
                close_segment();
                return vp_des_node::handle_frame_meta(meta);
            }
        }

        try {
            video_writer.write(frame);
            ++current_video_frame_index;
        }
        catch (const cv::Exception& error) {
            update_status(vp_indexed_record_state::WRITE_FAILED, error.what());
            close_segment();
            return vp_des_node::handle_frame_meta(meta);
        }

        collect_checkpoint(*meta);
        return vp_des_node::handle_frame_meta(meta);
    }

    std::shared_ptr<vp_objects::vp_meta> vp_indexed_file_des_node::handle_control_meta(
        std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return vp_des_node::handle_control_meta(meta);
    }

    void vp_indexed_file_des_node::set_status_hooker(vp_indexed_record_status_hooker hooker) {
        std::lock_guard<std::mutex> guard(status_lock);
        status_hooker = std::move(hooker);
    }

    vp_indexed_record_status vp_indexed_file_des_node::get_record_status() const {
        std::lock_guard<std::mutex> guard(status_lock);
        return current_status;
    }

}
