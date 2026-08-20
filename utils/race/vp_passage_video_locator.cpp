#include "vp_passage_video_locator.h"

#include <cmath>
#include <stdexcept>

namespace vp_utils {

    void vp_video_time_calibration::validate() const {
        if (camera_id.empty()) {
            throw std::invalid_argument("video time calibration camera_id is required");
        }
        if (channel_index < 0) {
            throw std::invalid_argument("video time calibration channel_index must not be negative");
        }
        if (race_reference_time_ms < 0) {
            throw std::invalid_argument("video time calibration race_reference_time_ms is required");
        }
        if (video_reference_wall_time_ms < 0) {
            throw std::invalid_argument("video time calibration video_reference_wall_time_ms is required");
        }
        if (!std::isfinite(drift_ppm) || std::abs(drift_ppm) > 10000.0) {
            throw std::invalid_argument("video time calibration drift_ppm is out of range");
        }
        if (expected_error_ms < 0) {
            throw std::invalid_argument("video time calibration expected_error_ms must not be negative");
        }
    }

    std::int64_t vp_video_time_calibration::map_race_time_to_video_wall_time(
        std::int64_t race_time_ms) const {
        validate();
        if (race_time_ms < 0) {
            throw std::invalid_argument("race_time_ms must not be negative");
        }
        const auto race_delta_ms = static_cast<double>(race_time_ms - race_reference_time_ms);
        const auto scale = 1.0 + drift_ppm / 1000000.0;
        return video_reference_wall_time_ms + static_cast<std::int64_t>(std::llround(race_delta_ms * scale));
    }

    vp_passage_video_locator::vp_passage_video_locator(
        const vp_video_frame_index_store& frame_index):
        frame_index(frame_index) {
    }

    vp_passage_video_location vp_passage_video_locator::locate(
        const vp_objects::vp_passage_event& event,
        const vp_video_time_calibration& calibration,
        std::int64_t before_ms,
        std::int64_t after_ms) const {
        event.validate();
        calibration.validate();
        if (before_ms < 0 || after_ms < 0) {
            throw std::invalid_argument("video locate window must not be negative");
        }

        vp_passage_video_location location;
        location.event_id = event.event_id;
        location.camera_id = calibration.camera_id;
        location.channel_index = calibration.channel_index;
        location.target_wall_time_ms = calibration.map_race_time_to_video_wall_time(event.passage_time_ms);
        location.window_start_wall_time_ms = location.target_wall_time_ms - before_ms - calibration.expected_error_ms;
        location.window_end_wall_time_ms = location.target_wall_time_ms + after_ms + calibration.expected_error_ms;
        location.expected_error_ms = calibration.expected_error_ms;
        location.window_checkpoints = frame_index.range(
            calibration.camera_id,
            calibration.channel_index,
            location.window_start_wall_time_ms,
            location.window_end_wall_time_ms);
        if (!location.window_checkpoints.empty()) {
            location.nearest_checkpoint = frame_index.nearest(
                calibration.camera_id,
                calibration.channel_index,
                location.target_wall_time_ms);
        }
        return location;
    }

}
