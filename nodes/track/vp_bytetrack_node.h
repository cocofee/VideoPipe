#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "vp_track_node.h"

namespace vp_nodes {
    struct vp_bytetrack_config {
        float high_confidence_threshold = 0.6f;
        float low_confidence_threshold = 0.1f;
        float iou_threshold = 0.3f;
        int min_hits = 3;
        int max_age = 30;
    };

    class vp_bytetrack_engine {
    private:
        struct track_state {
            int id = -1;
            vp_objects::vp_rect rect;
            float velocity_x = 0.0f;
            float velocity_y = 0.0f;
            float velocity_width = 0.0f;
            float velocity_height = 0.0f;
            int hits = 0;
            int missed_frames = 0;
            bool confirmed = false;
        };

        vp_bytetrack_config config;
        std::map<int, std::vector<track_state>> tracks_by_channel;
        int next_track_id = 1;
        vp_track_stats cumulative_stats;
        mutable std::mutex state_mutex;

    public:
        explicit vp_bytetrack_engine(vp_bytetrack_config config = {});

        vp_track_stats update(
            int channel_index,
            const std::vector<vp_objects::vp_rect>& target_rects,
            const std::vector<float>& target_scores,
            std::vector<int>& track_ids);

        vp_track_stats statistics() const;
    };

#ifndef VP_BYTETRACK_TEST_ONLY
    class vp_bytetrack_node : public vp_track_node {
    private:
        vp_bytetrack_engine engine;

    protected:
        virtual void track(
            int channel_index,
            const std::vector<vp_objects::vp_rect>& target_rects,
            const std::vector<std::vector<float>>& target_embeddings,
            std::vector<int>& track_ids) override;

    public:
        vp_bytetrack_node(
            std::string node_name,
            vp_track_for track_for = vp_track_for::NORMAL,
            float high_confidence_threshold = 0.6f,
            float low_confidence_threshold = 0.1f,
            float iou_threshold = 0.3f,
            int min_hits = 3,
            int max_age = 30);
        virtual ~vp_bytetrack_node();
    };
#endif
}
