#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "../../objects/vp_frame_meta.h"
#include "vp_video_frame_index.h"

namespace vp_utils {

    bool is_video_frame_checkpoint_due(
        const std::string& previous_video_path,
        int previous_source_session,
        std::int64_t previous_monotonic_time_us,
        const std::string& video_path,
        int source_session,
        std::int64_t capture_monotonic_us,
        std::int64_t minimum_interval_ms);

    class vp_video_frame_index_writer {
    private:
        struct channel_state {
            std::string video_path;
            int source_session = -1;
            std::int64_t last_monotonic_time_us = -1;
        };

        std::string camera_id;
        vp_video_frame_index_store& store;
        std::int64_t minimum_interval_ms;
        std::map<int, channel_state> channel_states;

    public:
        vp_video_frame_index_writer(
            std::string camera_id,
            vp_video_frame_index_store& store,
            std::int64_t minimum_interval_ms = 250);

        // Returns true when a persistent checkpoint was appended.
        bool append_frame(
            const vp_objects::vp_frame_meta& frame_meta,
            const std::string& video_path,
            std::int64_t video_frame_index);
    };

}
