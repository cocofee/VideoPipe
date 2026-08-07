
#include "vp_track_node.h"
//#include "../objects/shapes/vp_rect.h"

namespace vp_nodes {
    namespace {
        struct track_statistics_registry {
            std::mutex mutex;
            std::map<const vp_track_node*, vp_track_stats> stats_by_node;
        };

        track_statistics_registry& statistics_registry() {
            static track_statistics_registry registry;
            return registry;
        }

        thread_local const vp_track_node* score_context_node = nullptr;
        thread_local const std::vector<float>* score_context_scores = nullptr;

        class track_score_context {
        private:
            const vp_track_node* previous_node = nullptr;
            const std::vector<float>* previous_scores = nullptr;

        public:
            track_score_context(
                const vp_track_node* node,
                const std::vector<float>* scores):
                previous_node(score_context_node),
                previous_scores(score_context_scores) {
                score_context_node = node;
                score_context_scores = scores;
            }

            ~track_score_context() {
                score_context_node = previous_node;
                score_context_scores = previous_scores;
            }
        };

        void add_statistics(vp_track_stats& total, const vp_track_stats& delta) {
            total.created_tracks += delta.created_tracks;
            total.deleted_tracks += delta.deleted_tracks;
            total.matched_detections += delta.matched_detections;
            total.unmatched_detections += delta.unmatched_detections;
            total.track_fragments += delta.track_fragments;
        }
    }
        
    vp_track_node::vp_track_node(std::string node_name, 
                                vp_track_for track_for): 
                                vp_node(node_name), 
                                track_for(track_for) {
        auto& registry = statistics_registry();
        std::lock_guard<std::mutex> guard(registry.mutex);
        registry.stats_by_node.emplace(this, vp_track_stats {});
    }
    
    vp_track_node::~vp_track_node() {
        auto& registry = statistics_registry();
        std::lock_guard<std::mutex> guard(registry.mutex);
        registry.stats_by_node.erase(this);
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) {
        return meta;
    }

    std::shared_ptr<vp_objects::vp_meta> vp_track_node::handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        // channel_index can be different each call
        auto channel_index = meta->channel_index;

        // data used for tracking
        std::vector<vp_objects::vp_rect> rects;      // rects of targets
        std::vector<std::vector<float>> embeddings;  // embeddings of targets
        std::vector<float> scores;                   // detector confidence of targets
        std::vector<int> track_ids;                  // track ids of targets

        // step 1, collect data
        preprocess(meta, rects, embeddings);
        if (track_for == vp_track_for::NORMAL) {
            for (const auto& target : meta->targets) {
                scores.push_back(target->primary_score);
            }
        }
        if (track_for == vp_track_for::FACE) {
            for (const auto& face : meta->face_targets) {
                scores.push_back(face->score);
            }
        }

        // step 2, track by channel
        {
            track_score_context score_context(this, &scores);
            track(channel_index, rects, embeddings, track_ids);
        }

        // step 3, postprocess
        postprocess(meta, rects, embeddings, track_ids);

        return meta;
    }

    void vp_track_node::preprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                                std::vector<vp_objects::vp_rect>& target_rects, 
                                std::vector<std::vector<float>>& target_embeddings) {
        if (track_for == vp_track_for::NORMAL) {
            for(auto& i: frame_meta->targets) {
                target_rects.push_back(i->get_rect());      // rect fo target (via i variable)
                target_embeddings.push_back(i->embeddings); // embeddings of target (via i variable)
            }
        }

        if (track_for == vp_track_for::FACE) {
            for(auto& i: frame_meta->face_targets) {
                target_rects.push_back(i->get_rect());       // rect of face target (via i variable)
                target_embeddings.push_back(i->embeddings);  // embeddings of face target (via i variable)
            }
        }
        /* ... extend for more track for... */
    }

    const std::vector<float>& vp_track_node::current_target_scores() const noexcept {
        static const std::vector<float> empty_scores;
        if (score_context_node == this && score_context_scores != nullptr) {
            return *score_context_scores;
        }
        return empty_scores;
    }

    void vp_track_node::record_tracking_statistics(const vp_track_stats& delta) {
        auto& registry = statistics_registry();
        std::lock_guard<std::mutex> guard(registry.mutex);
        const auto entry = registry.stats_by_node.find(this);
        if (entry != registry.stats_by_node.end()) {
            add_statistics(entry->second, delta);
        }
    }

    vp_track_stats vp_track_node::tracking_statistics() const {
        auto& registry = statistics_registry();
        std::lock_guard<std::mutex> guard(registry.mutex);
        const auto entry = registry.stats_by_node.find(this);
        return entry == registry.stats_by_node.end() ? vp_track_stats {} : entry->second;
    }

    // write track_ids back to frame meta
    // we can also cache history rects for each target, and then push them back to tracks field (such as vp_frame_target::tracks)
    void vp_track_node::postprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                    const std::vector<vp_objects::vp_rect>& target_rects, 
                    const std::vector<std::vector<float>>& target_embeddings, 
                    const std::vector<int>& track_ids) {

        if (track_ids.empty()) {
            return;
        }
        // assert for length of vectors since they are generated by step1 & step2 separately
        // assert(target_rects.size() == target_embeddings.size());
        assert(target_rects.size() == track_ids.size());

        // support multi channels
        auto& tracks_by_id = all_tracks_by_id[frame_meta->channel_index];
        auto& last_tracked_frame_indexes = all_last_tracked_frame_indexes[frame_meta->channel_index];

        if (track_for == vp_track_for::NORMAL) {
            //assert(target_rects.size() == frame_meta->targets.size());
            for (int i = 0; i < frame_meta->targets.size(); i++) {
                auto& target = frame_meta->targets[i];
                auto& rect = target_rects[i];
                auto& track_id = track_ids[i];

                // -1 means no track result returned yet
                if (track_id != -1) {
                    tracks_by_id[track_id].push_back(rect);                           // cache
                    last_tracked_frame_indexes[track_id] = frame_meta->frame_index;   // update stamp

                    target->track_id = track_id;               // write track_id back to target
                    target->tracks = tracks_by_id[track_id];   // write tracks back to target
                }
            }
        }

        if (track_for == vp_track_for::FACE) {
            // assert(target_rects.size() == frame_meta->face_targets.size());
            for (int i = 0; i < frame_meta->face_targets.size(); i++) {
                auto& face = frame_meta->face_targets[i];
                auto& rect = target_rects[i];
                auto& track_id = track_ids[i];

                // -1 means no track result returned yet
                if (track_id != -1) {
                    tracks_by_id[track_id].push_back(rect);                           // cache
                    last_tracked_frame_indexes[track_id] = frame_meta->frame_index;   // update stamp

                    face->track_id = track_id;                // write track_id back to face target
                    face->tracks = tracks_by_id[track_id];    // write tracks back to face target
                }
            }
        }
        /* ... extend for more track for... */

        // remove cache tracks if has been long time since last updated (maybe it disappeared already).
        for (auto i = last_tracked_frame_indexes.begin(); i != last_tracked_frame_indexes.end();) {
            if (frame_meta->frame_index - (i->second) > max_allowed_disappear_frames 
                || frame_meta->frame_index < i->second) {
                VP_DEBUG(vp_utils::string_format("[%s] [tracking] long time no update, so erase cache of tracks for track_id:`%d`, size of tracks is:`%d`", node_name.c_str(), i->first, tracks_by_id[i->first].size()));
                tracks_by_id.erase(i->first);              // erase tracks first
                i = last_tracked_frame_indexes.erase(i);   // erase stamp then
            }
            else {
                i++;
            }
        }
    }
}
