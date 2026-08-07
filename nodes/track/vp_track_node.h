
#pragma once

#include <cstdint>
#include <map>
#include <assert.h>
#include "../vp_node.h"

namespace vp_nodes {
    // Counters are node-lifetime totals. ByteTrack counts detections at or
    // above its low threshold; a new high-confidence track starts unmatched.
    // track_fragments counts confirmed tracklets retired after max_age. True
    // identity fragmentation still requires comparison with external records.
    struct vp_track_stats {
        std::uint64_t created_tracks = 0;
        std::uint64_t deleted_tracks = 0;
        std::uint64_t matched_detections = 0;
        std::uint64_t unmatched_detections = 0;
        std::uint64_t track_fragments = 0;
    };

    // track node applied to which type of target (vp_frame_target, vp_frame_face_target or others)
    enum class vp_track_for {
        NORMAL = 1,    // vp_frame_target
        FACE = 2       // vp_frame_face_target
                       // others to extend
    };

    // base class for tracking, can not be initialized directly.
    // note that a track node can work on different channels at the same time
    class vp_track_node: public vp_node {
    private:
        // track for
        vp_track_for track_for = vp_track_for::NORMAL;
        
        // cache tracks at previous frames
        // std::map<int, std::vector<vp_objects::vp_rect>> tracks_by_id;
        std::map<int, std::map<int, std::vector<vp_objects::vp_rect>>> all_tracks_by_id;

        // stamp
        // std::map<int, int> last_tracked_frame_indexes;
        std::map<int, std::map<int, int>> all_last_tracked_frame_indexes;

        // remove cache tracks if it has been long time since last tracked.
        const int max_allowed_disappear_frames = 25;
    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(std::shared_ptr<vp_objects::vp_frame_meta> meta) override final;
        virtual std::shared_ptr<vp_objects::vp_meta> handle_control_meta(std::shared_ptr<vp_objects::vp_control_meta> meta) override final;

        // prepare data according to `track_for`
        void preprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                        std::vector<vp_objects::vp_rect>& target_rects, 
                        std::vector<std::vector<float>>& target_embeddings);
        
        // track api
        // it is a pure virtual function which should be implemented by derived class.
        // In:  rects & embeddings whose size() can be zero
        // Out: track ids
        virtual void track(int channel_index, const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        std::vector<int>& track_ids) = 0;

        // Scores are available only during the synchronous track(...) call.
        // Existing trackers do not need to use this compatibility accessor.
        const std::vector<float>& current_target_scores() const noexcept;

        void record_tracking_statistics(const vp_track_stats& delta);

        // write track_ids back to frame meta
        // we can also cache history rects for each target, and then push them back to tracks field (like vp_frame_target::tracks)
        void postprocess(std::shared_ptr<vp_objects::vp_frame_meta> frame_meta, 
                        const std::vector<vp_objects::vp_rect>& target_rects, 
                        const std::vector<std::vector<float>>& target_embeddings, 
                        const std::vector<int>& track_ids);
    public:
        vp_track_node(std::string node_name, vp_track_for track_for = vp_track_for::NORMAL);
        virtual ~vp_track_node();

        vp_track_stats tracking_statistics() const;
    };
}
