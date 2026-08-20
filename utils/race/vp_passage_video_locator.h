#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../../objects/vp_passage_event.h"
#include "vp_video_frame_index.h"

namespace vp_utils {

    struct vp_video_time_calibration {
        std::string camera_id;
        int channel_index = -1;

        // A known matching point on the CycleRace and video timelines.
        std::int64_t race_reference_time_ms = -1;
        std::int64_t video_reference_wall_time_ms = -1;

        // Positive means the video clock advances faster than CycleRace.
        double drift_ppm = 0.0;
        std::int64_t expected_error_ms = 0;

        void validate() const;
        std::int64_t map_race_time_to_video_wall_time(std::int64_t race_time_ms) const;
    };

    struct vp_passage_video_location {
        std::string event_id;
        std::string camera_id;
        int channel_index = -1;
        std::int64_t target_wall_time_ms = -1;
        std::int64_t window_start_wall_time_ms = -1;
        std::int64_t window_end_wall_time_ms = -1;
        std::int64_t expected_error_ms = 0;
        std::optional<vp_video_frame_index_entry> nearest_checkpoint;
        std::vector<vp_video_frame_index_entry> window_checkpoints;
    };

    class vp_passage_video_locator {
    private:
        const vp_video_frame_index_store& frame_index;

    public:
        explicit vp_passage_video_locator(const vp_video_frame_index_store& frame_index);

        vp_passage_video_location locate(
            const vp_objects::vp_passage_event& event,
            const vp_video_time_calibration& calibration,
            std::int64_t before_ms = 1000,
            std::int64_t after_ms = 1000) const;
    };

}
