#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "../utils/race/vp_passage_video_locator.h"
#include "../utils/race/vp_video_frame_index_writer.h"

namespace {

    vp_utils::vp_video_frame_index_entry make_entry(
        const std::string& entry_id,
        const std::string& camera_id,
        std::int64_t wall_time_ms,
        int frame_index) {
        vp_utils::vp_video_frame_index_entry entry;
        entry.entry_id = entry_id;
        entry.camera_id = camera_id;
        entry.channel_index = 0;
        entry.source_session = 0;
        entry.source_pts_us = frame_index * 250000LL;
        entry.capture_monotonic_us = wall_time_ms * 1000;
        entry.capture_wall_time_ms = wall_time_ms;
        entry.frame_index = frame_index;
        entry.video_path = camera_id + "-segment-0001.mp4";
        entry.video_frame_index = frame_index;
        return entry;
    }

}

int main() {
    const auto journal_path = std::filesystem::temp_directory_path() / "videopipe_frame_index.jsonl";
    std::error_code error;
    std::filesystem::remove(journal_path, error);

    {
        vp_utils::vp_video_frame_index_store store(journal_path);
        store.append(make_entry("camera-a-1", "camera-a", 1000750, 3));
        store.append(make_entry("camera-a-2", "camera-a", 1001001, 4));
        store.append(make_entry("camera-a-3", "camera-a", 1001250, 5));
        store.append(make_entry("camera-b-1", "camera-b", 1001001, 40));

        if (store.append(make_entry("camera-a-2", "camera-a", 1001001, 4))) {
            std::cerr << "duplicate frame index entry was accepted" << std::endl;
            return 1;
        }
    }

    {
        std::ofstream corrupt_tail(journal_path, std::ios::out | std::ios::app | std::ios::binary);
        corrupt_tail << "{\"schema_version\":1";
    }
    vp_utils::vp_video_frame_index_store recovered_store(journal_path);
    if (recovered_store.size() != 4 || !recovered_store.recovered_incomplete_tail()) {
        std::cerr << "frame index recovery failed" << std::endl;
        return 1;
    }

    const auto malformed_path = journal_path.string() + ".malformed";
    std::filesystem::copy_file(
        journal_path,
        malformed_path,
        std::filesystem::copy_options::overwrite_existing,
        error);
    {
        std::ofstream malformed_tail(
            malformed_path,
            std::ios::out | std::ios::app | std::ios::binary);
        malformed_tail << "{\"schema_version\":1,}";
    }
    const auto malformed_size = std::filesystem::file_size(malformed_path);
    bool malformed_rejected = false;
    try {
        vp_utils::vp_video_frame_index_store malformed_store(malformed_path);
    }
    catch (const std::exception&) {
        malformed_rejected = true;
    }
    if (!malformed_rejected || std::filesystem::file_size(malformed_path) != malformed_size) {
        std::cerr << "malformed frame index tail was incorrectly recovered" << std::endl;
        return 1;
    }

    auto channel_zero = make_entry("", "camera-a", 1002000, 0);
    channel_zero.video_path = "camera-a-segment-0002.mp4";
    channel_zero.entry_id = vp_utils::make_video_frame_index_entry_id(channel_zero);
    auto channel_one = channel_zero;
    channel_one.channel_index = 1;
    channel_one.entry_id = vp_utils::make_video_frame_index_entry_id(channel_one);
    auto restarted_segment = channel_zero;
    restarted_segment.video_path = "camera-a-segment-0003.mp4";
    restarted_segment.entry_id = vp_utils::make_video_frame_index_entry_id(restarted_segment);
    if (channel_zero.entry_id == channel_one.entry_id ||
        channel_zero.entry_id == restarted_segment.entry_id) {
        std::cerr << "frame index identity isolation failed" << std::endl;
        return 1;
    }
    if (vp_utils::is_video_frame_checkpoint_due(
            "segment-a.mp4", 0, 1000000, "segment-a.mp4", 0, 1100000, 250) ||
        !vp_utils::is_video_frame_checkpoint_due(
            "segment-a.mp4", 0, 1000000, "segment-a.mp4", 0, 1250000, 250) ||
        !vp_utils::is_video_frame_checkpoint_due(
            "segment-a.mp4", 0, 1000000, "segment-b.mp4", 0, 1100000, 250) ||
        !vp_utils::is_video_frame_checkpoint_due(
            "segment-a.mp4", 0, 1000000, "segment-a.mp4", 1, 1100000, 250)) {
        std::cerr << "frame index checkpoint policy failed" << std::endl;
        return 1;
    }

    vp_objects::vp_passage_event event;
    event.event_id = "race-final-0001";
    event.race_id = "race";
    event.stage_id = "final";
    event.group_id = "men-open";
    event.sequence = 1;
    event.chip_id = "chip-23";
    event.bib = "23";
    event.passage_time_ms = 5000;
    event.lap = 1;

    vp_utils::vp_video_time_calibration calibration;
    calibration.camera_id = "camera-a";
    calibration.channel_index = 0;
    calibration.race_reference_time_ms = 4000;
    calibration.video_reference_wall_time_ms = 1000000;
    calibration.drift_ppm = 1000.0;
    calibration.expected_error_ms = 100;

    vp_utils::vp_passage_video_locator locator(recovered_store);
    const auto location = locator.locate(event, calibration, 500, 500);
    if (location.target_wall_time_ms != 1001001 ||
        !location.nearest_checkpoint ||
        location.nearest_checkpoint->entry_id != "camera-a-2" ||
        location.nearest_checkpoint->camera_id != "camera-a" ||
        location.window_checkpoints.size() != 3) {
        std::cerr << "passage video location failed" << std::endl;
        return 1;
    }

    vp_objects::vp_passage_event outside_event = event;
    outside_event.event_id = "race-final-outside";
    outside_event.passage_time_ms = 9000000;
    const auto outside_location = locator.locate(outside_event, calibration, 500, 500);
    if (outside_location.nearest_checkpoint || !outside_location.window_checkpoints.empty()) {
        std::cerr << "out-of-range video location returned a checkpoint" << std::endl;
        return 1;
    }

    std::filesystem::remove(journal_path, error);
    std::filesystem::remove(malformed_path, error);
    std::cout << "cycle race video locator sample passed" << std::endl;
    return 0;
}
