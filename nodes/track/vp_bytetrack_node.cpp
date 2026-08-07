#include "vp_bytetrack_node.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <utility>

#include "sort/Hungarian.h"

namespace vp_nodes {
    namespace {
        struct detection {
            vp_objects::vp_rect rect;
            float score = 0.0f;
            int original_index = -1;
        };

        struct association_result {
            std::vector<std::pair<int, int>> matches;
            std::vector<int> unmatched_tracks;
            std::vector<int> unmatched_detections;
        };

        bool is_valid_rect(const vp_objects::vp_rect& rect) {
            return rect.width > 0 && rect.height > 0;
        }

        double intersection_over_union(
            const vp_objects::vp_rect& left,
            const vp_objects::vp_rect& right) {
            const auto intersection_left = std::max(left.x, right.x);
            const auto intersection_top = std::max(left.y, right.y);
            const auto intersection_right = std::min(left.x + left.width, right.x + right.width);
            const auto intersection_bottom = std::min(left.y + left.height, right.y + right.height);
            const auto intersection_width = std::max(0, intersection_right - intersection_left);
            const auto intersection_height = std::max(0, intersection_bottom - intersection_top);
            const auto intersection_area =
                static_cast<double>(intersection_width) * intersection_height;
            const auto union_area =
                static_cast<double>(left.width) * left.height +
                static_cast<double>(right.width) * right.height -
                intersection_area;
            return union_area > 0.0 ? intersection_area / union_area : 0.0;
        }

        template <typename Track>
        vp_objects::vp_rect predicted_rect(const Track& track) {
            const auto elapsed_frames = track.missed_frames + 1;
            return {
                static_cast<int>(std::lround(track.rect.x + track.velocity_x * elapsed_frames)),
                static_cast<int>(std::lround(track.rect.y + track.velocity_y * elapsed_frames)),
                std::max(1, static_cast<int>(std::lround(
                    track.rect.width + track.velocity_width * elapsed_frames))),
                std::max(1, static_cast<int>(std::lround(
                    track.rect.height + track.velocity_height * elapsed_frames))),
            };
        }

        template <typename Track>
        association_result associate(
            const std::vector<Track>& tracks,
            const std::vector<int>& track_indices,
            const std::vector<detection>& detections,
            const std::vector<int>& detection_indices,
            float iou_threshold) {
            association_result result;
            if (track_indices.empty()) {
                result.unmatched_detections = detection_indices;
                return result;
            }
            if (detection_indices.empty()) {
                result.unmatched_tracks = track_indices;
                return result;
            }

            const auto assignment_size = std::min(
                track_indices.size(),
                detection_indices.size());
            const auto forbidden_cost = static_cast<double>(assignment_size + 1);
            std::vector<std::vector<double>> cost_matrix(
                track_indices.size(),
                std::vector<double>(detection_indices.size(), forbidden_cost));
            for (std::size_t row = 0; row < track_indices.size(); ++row) {
                const auto prediction = predicted_rect(tracks[track_indices[row]]);
                for (std::size_t column = 0; column < detection_indices.size(); ++column) {
                    const auto iou = intersection_over_union(
                        prediction,
                        detections[detection_indices[column]].rect);
                    if (iou >= iou_threshold) {
                        cost_matrix[row][column] = 1.0 - iou;
                    }
                }
            }

            HungarianAlgorithm hungarian;
            std::vector<int> assignment;
            hungarian.Solve(cost_matrix, assignment);

            std::set<int> matched_detection_indices;
            for (std::size_t row = 0; row < track_indices.size(); ++row) {
                const auto assigned_column = assignment[row];
                if (assigned_column < 0 || assigned_column >= static_cast<int>(detection_indices.size())) {
                    result.unmatched_tracks.push_back(track_indices[row]);
                    continue;
                }

                const auto detection_index = detection_indices[assigned_column];
                const auto iou = intersection_over_union(
                    predicted_rect(tracks[track_indices[row]]),
                    detections[detection_index].rect);
                if (cost_matrix[row][assigned_column] >= forbidden_cost ||
                    iou < iou_threshold ||
                    matched_detection_indices.count(detection_index) != 0) {
                    result.unmatched_tracks.push_back(track_indices[row]);
                    continue;
                }
                result.matches.emplace_back(track_indices[row], detection_index);
                matched_detection_indices.insert(detection_index);
            }

            for (const auto detection_index : detection_indices) {
                if (matched_detection_indices.count(detection_index) == 0) {
                    result.unmatched_detections.push_back(detection_index);
                }
            }
            return result;
        }

        void add_stats(vp_track_stats& total, const vp_track_stats& delta) {
            total.created_tracks += delta.created_tracks;
            total.deleted_tracks += delta.deleted_tracks;
            total.matched_detections += delta.matched_detections;
            total.unmatched_detections += delta.unmatched_detections;
            total.track_fragments += delta.track_fragments;
        }
    }

    vp_bytetrack_engine::vp_bytetrack_engine(vp_bytetrack_config config):
        config(config) {
        if (!std::isfinite(config.high_confidence_threshold) ||
            !std::isfinite(config.low_confidence_threshold) ||
            !std::isfinite(config.iou_threshold) ||
            config.high_confidence_threshold <= 0.0f ||
            config.high_confidence_threshold > 1.0f ||
            config.low_confidence_threshold < 0.0f ||
            config.low_confidence_threshold >= config.high_confidence_threshold ||
            config.iou_threshold <= 0.0f ||
            config.iou_threshold > 1.0f ||
            config.min_hits <= 0 ||
            config.max_age < 0) {
            throw std::invalid_argument("invalid ByteTrack configuration");
        }
    }

    vp_track_stats vp_bytetrack_engine::update(
        int channel_index,
        const std::vector<vp_objects::vp_rect>& target_rects,
        const std::vector<float>& target_scores,
        std::vector<int>& track_ids) {
        if (target_rects.size() != target_scores.size()) {
            throw std::invalid_argument("ByteTrack rect and score counts must match");
        }

        std::lock_guard<std::mutex> guard(state_mutex);
        track_ids.assign(target_rects.size(), -1);
        vp_track_stats frame_stats;

        std::vector<detection> detections;
        std::vector<int> high_detection_indices;
        std::vector<int> low_detection_indices;
        for (std::size_t index = 0; index < target_rects.size(); ++index) {
            const auto score = target_scores[index];
            if (!is_valid_rect(target_rects[index]) ||
                !std::isfinite(score) ||
                score < config.low_confidence_threshold ||
                score > 1.0f) {
                continue;
            }

            const auto detection_index = static_cast<int>(detections.size());
            detections.push_back({target_rects[index], score, static_cast<int>(index)});
            if (score >= config.high_confidence_threshold) {
                high_detection_indices.push_back(detection_index);
            }
            else {
                low_detection_indices.push_back(detection_index);
            }
        }

        auto& tracks = tracks_by_channel[channel_index];
        std::vector<int> all_track_indices;
        all_track_indices.reserve(tracks.size());
        for (std::size_t index = 0; index < tracks.size(); ++index) {
            all_track_indices.push_back(static_cast<int>(index));
        }

        const auto high_association = associate(
            tracks,
            all_track_indices,
            detections,
            high_detection_indices,
            config.iou_threshold);
        const auto low_association = associate(
            tracks,
            high_association.unmatched_tracks,
            detections,
            low_detection_indices,
            config.iou_threshold);

        std::set<int> matched_track_indices;
        const auto apply_matches = [&](const std::vector<std::pair<int, int>>& matches) {
            for (const auto& match : matches) {
                auto& track = tracks[match.first];
                const auto& matched_detection = detections[match.second];
                const auto elapsed_frames = track.missed_frames + 1;
                track.velocity_x =
                    static_cast<float>(matched_detection.rect.x - track.rect.x) / elapsed_frames;
                track.velocity_y =
                    static_cast<float>(matched_detection.rect.y - track.rect.y) / elapsed_frames;
                track.velocity_width =
                    static_cast<float>(matched_detection.rect.width - track.rect.width) / elapsed_frames;
                track.velocity_height =
                    static_cast<float>(matched_detection.rect.height - track.rect.height) / elapsed_frames;
                track.rect = matched_detection.rect;
                track.missed_frames = 0;
                track.hits++;
                track.confirmed = track.hits >= config.min_hits;
                matched_track_indices.insert(match.first);
                frame_stats.matched_detections++;
                if (track.confirmed) {
                    track_ids[matched_detection.original_index] = track.id;
                }
            }
        };
        apply_matches(high_association.matches);
        apply_matches(low_association.matches);

        for (const auto track_index : low_association.unmatched_tracks) {
            if (matched_track_indices.count(track_index) == 0) {
                auto& unmatched_track = tracks[track_index];
                unmatched_track.missed_frames++;
                if (!unmatched_track.confirmed) {
                    unmatched_track.hits = 0;
                }
            }
        }

        frame_stats.unmatched_detections += high_association.unmatched_detections.size();
        frame_stats.unmatched_detections += low_association.unmatched_detections.size();
        for (const auto detection_index : high_association.unmatched_detections) {
            const auto& unmatched_detection = detections[detection_index];
            track_state new_track;
            new_track.id = next_track_id++;
            new_track.rect = unmatched_detection.rect;
            new_track.hits = 1;
            new_track.confirmed = config.min_hits <= 1;
            tracks.push_back(new_track);
            frame_stats.created_tracks++;
            if (new_track.confirmed) {
                track_ids[unmatched_detection.original_index] = new_track.id;
            }
        }

        tracks.erase(
            std::remove_if(
                tracks.begin(),
                tracks.end(),
                [&](const track_state& track) {
                    if (track.missed_frames <= config.max_age) {
                        return false;
                    }
                    frame_stats.deleted_tracks++;
                    // A retired confirmed track is one completed track segment.
                    if (track.confirmed) {
                        frame_stats.track_fragments++;
                    }
                    return true;
                }),
            tracks.end());

        add_stats(cumulative_stats, frame_stats);
        return frame_stats;
    }

    vp_track_stats vp_bytetrack_engine::statistics() const {
        std::lock_guard<std::mutex> guard(state_mutex);
        return cumulative_stats;
    }

#ifndef VP_BYTETRACK_TEST_ONLY
    vp_bytetrack_node::vp_bytetrack_node(
        std::string node_name,
        vp_track_for track_for,
        float high_confidence_threshold,
        float low_confidence_threshold,
        float iou_threshold,
        int min_hits,
        int max_age):
        vp_track_node(node_name, track_for),
        engine({
            high_confidence_threshold,
            low_confidence_threshold,
            iou_threshold,
            min_hits,
            max_age,
        }) {
        initialized();
    }

    vp_bytetrack_node::~vp_bytetrack_node() {
        deinitialized();
    }

    void vp_bytetrack_node::track(
        int channel_index,
        const std::vector<vp_objects::vp_rect>& target_rects,
        const std::vector<std::vector<float>>& target_embeddings,
        std::vector<int>& track_ids) {
        (void)target_embeddings;
        const auto& contextual_scores = current_target_scores();
        const auto fallback_scores = contextual_scores.size() == target_rects.size()
            ? std::vector<float> {}
            : std::vector<float>(target_rects.size(), 1.0f);
        const auto& target_scores = fallback_scores.empty()
            ? contextual_scores
            : fallback_scores;
        const auto delta = engine.update(
            channel_index,
            target_rects,
            target_scores,
            track_ids);
        record_tracking_statistics(delta);
    }
#endif
}
