#include "vp_video_frame_index_writer.h"

#include <stdexcept>
#include <utility>

namespace vp_utils {

    bool is_video_frame_checkpoint_due(
        const std::string& previous_video_path,
        int previous_source_session,
        std::int64_t previous_monotonic_time_us,
        const std::string& video_path,
        int source_session,
        std::int64_t capture_monotonic_us,
        std::int64_t minimum_interval_ms) {
        return previous_video_path != video_path ||
            previous_source_session != source_session ||
            previous_monotonic_time_us < 0 ||
            capture_monotonic_us - previous_monotonic_time_us >= minimum_interval_ms * 1000;
    }

    vp_video_frame_index_writer::vp_video_frame_index_writer(
        std::string camera_id,
        vp_video_frame_index_store& store,
        std::int64_t minimum_interval_ms):
        camera_id(std::move(camera_id)),
        store(store),
        minimum_interval_ms(minimum_interval_ms) {
        if (this->camera_id.empty()) {
            throw std::invalid_argument("video frame index writer camera_id is required");
        }
        if (this->minimum_interval_ms <= 0) {
            throw std::invalid_argument("video frame index writer minimum_interval_ms must be positive");
        }
    }

    bool vp_video_frame_index_writer::append_frame(
        const vp_objects::vp_frame_meta& frame_meta,
        const std::string& video_path,
        std::int64_t video_frame_index) {
        auto& state = channel_states[frame_meta.channel_index];
        if (!is_video_frame_checkpoint_due(
                state.video_path,
                state.source_session,
                state.last_monotonic_time_us,
                video_path,
                frame_meta.source_session,
                frame_meta.capture_monotonic_us,
                minimum_interval_ms)) {
            return false;
        }

        vp_video_frame_index_entry entry;
        entry.camera_id = camera_id;
        entry.channel_index = frame_meta.channel_index;
        entry.source_session = frame_meta.source_session;
        entry.source_pts_us = frame_meta.source_pts_us;
        entry.capture_monotonic_us = frame_meta.capture_monotonic_us;
        entry.capture_wall_time_ms = frame_meta.capture_wall_time_ms;
        entry.frame_index = frame_meta.frame_index;
        entry.video_path = video_path;
        entry.video_frame_index = video_frame_index;
        entry.entry_id = make_video_frame_index_entry_id(entry);

        const auto appended = store.append(entry);
        if (appended) {
            state.video_path = video_path;
            state.source_session = frame_meta.source_session;
            state.last_monotonic_time_us = frame_meta.capture_monotonic_us;
        }
        return appended;
    }

}
