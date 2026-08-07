#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../nodes/track/vp_bytetrack_node.h"

namespace {

using vp_nodes::vp_bytetrack_config;
using vp_nodes::vp_bytetrack_engine;
using vp_objects::vp_rect;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_invalid_argument(
    const std::function<void()>& operation,
    const char* message) {
    try {
        operation();
    }
    catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

std::vector<int> update(
    vp_bytetrack_engine& engine,
    int channel_index,
    std::vector<vp_rect> rects,
    std::vector<float> scores) {
    std::vector<int> track_ids;
    engine.update(channel_index, rects, scores, track_ids);
    require(track_ids.size() == rects.size(), "tracker output must align with detections");
    return track_ids;
}

void verify_configuration_validation() {
    require_invalid_argument(
        []() {
            vp_bytetrack_engine(vp_bytetrack_config {.6f, .6f, .3f, 2, 2});
        },
        "low threshold must be below the high threshold");
    require_invalid_argument(
        []() {
            vp_bytetrack_engine(vp_bytetrack_config {.6f, .1f, 0.0f, 2, 2});
        },
        "IoU threshold must be positive");
    require_invalid_argument(
        []() {
            vp_bytetrack_engine(vp_bytetrack_config {.6f, .1f, .3f, 0, 2});
        },
        "min_hits must be positive");
    require_invalid_argument(
        []() {
            vp_bytetrack_engine(vp_bytetrack_config {.6f, .1f, .3f, 2, -1});
        },
        "max_age must be non-negative");
}

void verify_two_frame_miss_and_low_confidence_recovery() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .2f, 2, 2});

    const auto first = update(engine, 0, {{100, 100, 20, 40}}, {.9f});
    require(first[0] == -1, "a track must remain tentative before min_hits");

    const auto confirmed = update(engine, 0, {{102, 100, 20, 40}}, {.9f});
    require(confirmed[0] > 0, "a continuous target must be confirmed at min_hits");
    const auto stable_id = confirmed[0];

    require(update(engine, 0, {}, {}).empty(), "an empty frame must return no IDs");
    require(update(engine, 0, {}, {}).empty(), "a second empty frame must return no IDs");

    const auto recovered = update(engine, 0, {{108, 100, 20, 40}}, {.3f});
    require(recovered[0] == stable_id, "low-confidence recovery after two misses must preserve the ID");

    const auto stats = engine.statistics();
    require(stats.created_tracks == 1, "recovery must not create a replacement track");
    require(stats.matched_detections == 2, "high and low stage matches must be counted");
    require(stats.unmatched_detections == 1, "the first detection must be counted as unmatched");
}

void verify_adjacent_targets_remain_distinct() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .2f, 2, 2});

    const auto tentative = update(
        engine,
        1,
        {{0, 0, 20, 40}, {18, 0, 20, 40}},
        {.9f, .9f});
    require(tentative[0] == -1 && tentative[1] == -1, "adjacent tracks must start tentative");

    const auto confirmed = update(
        engine,
        1,
        {{1, 0, 20, 40}, {19, 0, 20, 40}},
        {.9f, .9f});
    require(confirmed[0] > 0 && confirmed[1] > 0, "both adjacent targets must be confirmed");
    require(confirmed[0] != confirmed[1], "adjacent targets must not merge into one track");
}

void verify_iou_gate_preserves_maximum_legal_matching() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .3f, 1, 2});

    const auto first = update(
        engine,
        0,
        {{0, 0, 20, 40}, {0, 0, 70, 40}},
        {.9f, .9f});
    require(first == std::vector<int>({1, 2}), "initial tracks must receive deterministic IDs");

    const auto crossed = update(
        engine,
        0,
        {{0, 0, 30, 40}, {10, 0, 20, 40}},
        {.9f, .9f});
    require(
        crossed == std::vector<int>({2, 1}),
        "IoU gating must maximize legal matches before optimizing IoU");

    const auto stats = engine.statistics();
    require(stats.created_tracks == 2, "a feasible crossing assignment must not create ID 3");
    require(stats.matched_detections == 2, "both legal crossing matches must be retained");
}

void verify_partial_legal_edges_do_not_duplicate_matches() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .3f, 1, 2});

    const auto first = update(
        engine,
        0,
        {{0, 0, 20, 40}, {100, 0, 20, 40}},
        {.9f, .9f});
    require(first == std::vector<int>({1, 2}), "initial tracks must receive deterministic IDs");

    const auto partial = update(
        engine,
        0,
        {{101, 0, 20, 40}, {300, 0, 20, 40}},
        {.9f, .9f});
    require(partial[0] == 2, "the only legal old-track match must be preserved");
    require(partial[1] == 3, "the unmatched detection must create exactly one new track");
    require(partial[0] != partial[1], "one track must not be assigned to two detections");
}

void verify_tentative_confirmation_requires_consecutive_hits() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .3f, 2, 2});

    require(
        update(engine, 0, {{10, 10, 20, 40}}, {.9f})[0] == -1,
        "the first hit must remain tentative");
    update(engine, 0, {}, {});
    require(
        update(engine, 0, {{10, 10, 20, 40}}, {.9f})[0] == -1,
        "a miss must reset a tentative track's consecutive hit streak");
    require(
        update(engine, 0, {{10, 10, 20, 40}}, {.9f})[0] == 1,
        "two consecutive hits after the miss must confirm the original track ID");
}

void verify_deletion_fragment_and_monotonic_ids() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .2f, 1, 2});

    const auto first = update(engine, 0, {{10, 10, 20, 40}}, {.9f});
    require(first[0] > 0, "min_hits=1 must confirm the first detection");

    update(engine, 0, {}, {});
    update(engine, 0, {}, {});
    update(engine, 0, {}, {});

    auto stats = engine.statistics();
    require(stats.deleted_tracks == 1, "a track must be deleted only after max_age is exceeded");
    require(stats.track_fragments == 1, "retiring a confirmed track must count one ended fragment");

    const auto replacement = update(engine, 0, {{200, 10, 20, 40}}, {.9f});
    require(replacement[0] > first[0], "track IDs must be monotonic and never reused");
    stats = engine.statistics();
    require(stats.created_tracks == 2, "the replacement must be counted as a new track");
}

void verify_invalid_detections_are_filtered() {
    vp_bytetrack_engine engine(vp_bytetrack_config {.6f, .1f, .2f, 1, 2});
    const auto ids = update(
        engine,
        0,
        {{0, 0, 0, 10}, {0, 0, 10, 10}, {20, 0, 10, 10}},
        {.9f, std::numeric_limits<float>::quiet_NaN(), .05f});
    require(ids == std::vector<int>({-1, -1, -1}), "invalid and below-threshold detections must stay untracked");

    const auto stats = engine.statistics();
    require(stats.created_tracks == 0, "filtered detections must not create tracks");
    require(stats.unmatched_detections == 0, "filtered detections must not affect association counters");

    require_invalid_argument(
        [&]() {
            std::vector<int> mismatched_ids;
            engine.update(0, {{0, 0, 10, 10}}, {}, mismatched_ids);
        },
        "rect and score counts must match");
}

}  // namespace

int main() {
    try {
        verify_configuration_validation();
        verify_two_frame_miss_and_low_confidence_recovery();
        verify_adjacent_targets_remain_distinct();
        verify_tentative_confirmation_requires_consecutive_hits();
        verify_iou_gate_preserves_maximum_legal_matching();
        verify_partial_legal_edges_do_not_duplicate_matches();
        verify_deletion_fragment_and_monotonic_ids();
        verify_invalid_detections_are_filtered();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    std::cout << "ByteTrack tests passed" << std::endl;
    return 0;
}
