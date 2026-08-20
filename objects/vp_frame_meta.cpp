#include <chrono>
#include <iterator>

#include "vp_frame_meta.h"

namespace vp_objects {
        
    vp_frame_meta::vp_frame_meta(cv::Mat frame,
                                 int frame_index,
                                 int channel_index,
                                 int original_width,
                                 int original_height,
                                 int fps,
                                 std::int64_t source_pts_us,
                                 int source_session,
                                 std::int64_t capture_monotonic_us,
                                 std::int64_t capture_wall_time_ms):
        vp_meta(vp_meta_type::FRAME, channel_index), 
        frame_index(frame_index),
        fps(fps),
        source_pts_us(source_pts_us),
        source_session(source_session),
        capture_monotonic_us(
            capture_monotonic_us >= 0
                ? capture_monotonic_us
                : std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch()).count()),
        capture_wall_time_ms(
            capture_wall_time_ms >= 0
                ? capture_wall_time_ms
                : std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count()),
        original_width(original_width),
        original_height(original_height),
        frame(frame) {
            assert(!frame.empty());
    }
    
    // copy constructor of vp_frame_meta would NOT be called at most time.
    // only when it flow through vp_split_node with vp_split_node::split_with_deep_copy==true.
    // in fact, all kinds of meta would NOT be copyed in its lifecycle, we just pass them by poniter most time.
    vp_frame_meta::vp_frame_meta(const vp_frame_meta& meta): 
        vp_meta(meta),
        frame_index(meta.frame_index),
        fps(meta.fps),
        source_pts_us(meta.source_pts_us),
        source_session(meta.source_session),
        capture_monotonic_us(meta.capture_monotonic_us),
        capture_wall_time_ms(meta.capture_wall_time_ms),
        original_width(meta.original_width),
        original_height(meta.original_height),
        description(meta.description) {
            // deep copy frame data
            this->frame = meta.frame.clone();
            this->osd_frame = meta.osd_frame.clone();
            this->mask = meta.mask.clone();

            // deep copy targets
            for(auto& i: meta.targets) {
                this->targets.push_back(i->clone());
            }
            // deep copy pose targets
            for(auto& i: meta.pose_targets) {
                this->pose_targets.push_back(i->clone());
            }
            // deep copy face targets
            for(auto& i: meta.face_targets) {
                this->face_targets.push_back(i->clone());
            }
            // deep copy text targets
            for(auto& i: meta.text_targets) {
                this->text_targets.push_back(i->clone());
            }
            // deep copy ba results
            for(auto& i: meta.ba_results) {
                this->ba_results.push_back(i->clone());
            }
    }
    
    vp_frame_meta::~vp_frame_meta() {

    }

    std::shared_ptr<vp_meta> vp_frame_meta::clone() {
        // just call copy constructor and return new pointer
        return std::make_shared<vp_frame_meta>(*this);
    }

    std::vector<std::shared_ptr<vp_frame_target>> vp_frame_meta::get_targets_by_ids(const std::vector<int>& ids) {
        std::vector<std::shared_ptr<vp_objects::vp_frame_target>> results;
        for(auto& t: targets) {
            if (std::find(ids.begin(), ids.end(), t->track_id) != ids.end()) {
                results.push_back(t);
            }
        }
        return results;
    }
}
