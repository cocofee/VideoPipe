#pragma once

#if defined(VP_WITH_ASYNC_OCR) || defined(VP_ASYNC_OCR_CORE_ONLY)

#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "vp_ocr_job.h"

namespace vp_objects {
    enum class vp_ocr_consensus_status {
        PENDING,
        CONFIRMED
    };

    struct vp_ocr_consensus_result {
        vp_ocr_consensus_status status = vp_ocr_consensus_status::PENDING;
        std::string text;
        double confidence_weight = 0.0;
        std::size_t supporting_frames = 0;
    };

    class vp_ocr_consensus {
    private:
        struct frame_vote {
            std::string text;
            float confidence = 0.0f;
            bool conflicted = false;
        };

        struct track_votes {
            std::map<int, frame_vote> votes_by_frame;
            int latest_frame_index = -1;
        };

        mutable std::mutex lock_;
        std::map<std::pair<int, int>, track_votes> votes_by_track_;

        static vp_ocr_consensus_result evaluate(const track_votes& votes);

    public:
        static std::string normalize_numeric(const std::string& text);

        vp_ocr_consensus_result observe(
            int channel_index,
            int track_id,
            int frame_index,
            const vp_ocr_candidate& candidate);

        vp_ocr_consensus_result snapshot(int channel_index, int track_id) const;
        void clear(int channel_index, int track_id);
        void clear_channel(int channel_index);
        void reset();
        std::size_t tracked_state_count() const;
    };
}

#endif
