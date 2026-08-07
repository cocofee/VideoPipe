#include "vp_ocr_consensus.h"

#if defined(VP_WITH_ASYNC_OCR) || defined(VP_ASYNC_OCR_CORE_ONLY)

#include <cctype>
#include <cmath>
#include <limits>

namespace vp_objects {
    std::string vp_ocr_consensus::normalize_numeric(const std::string& text) {
        std::string normalized;
        normalized.reserve(text.size());
        for (const auto character : text) {
            const auto value = static_cast<unsigned char>(character);
            if (std::isspace(value)) {
                continue;
            }
            if (!std::isdigit(value)) {
                return "";
            }
            normalized.push_back(character);
        }
        return normalized;
    }

    vp_ocr_consensus_result vp_ocr_consensus::evaluate(const track_votes& votes) {
        vp_ocr_consensus_result result;
        double highest_weight = -1.0;
        double second_weight = -1.0;
        std::map<std::string, std::pair<double, std::size_t>> totals_by_text;

        for (const auto& frame_entry : votes.votes_by_frame) {
            const auto& vote = frame_entry.second;
            if (vote.conflicted) {
                continue;
            }
            auto& total = totals_by_text[vote.text];
            total.first += vote.confidence;
            ++total.second;
        }

        for (const auto& text_entry : totals_by_text) {
            const auto weight = text_entry.second.first;
            if (weight > highest_weight) {
                second_weight = highest_weight;
                highest_weight = weight;
                result.text = text_entry.first;
                result.confidence_weight = weight;
                result.supporting_frames = text_entry.second.second;
            }
            else if (weight > second_weight) {
                second_weight = weight;
            }
        }

        if (highest_weight < 0.0) {
            return result;
        }

        constexpr double tie_tolerance = 1e-6;
        const bool unique_winner = second_weight < 0.0 || highest_weight - second_weight > tie_tolerance;
        if (unique_winner && result.supporting_frames >= 2) {
            result.status = vp_ocr_consensus_status::CONFIRMED;
        }
        return result;
    }

    vp_ocr_consensus_result vp_ocr_consensus::observe(
        int channel_index,
        int track_id,
        int frame_index,
        const vp_ocr_candidate& candidate) {
        const auto normalized = normalize_numeric(candidate.text);
        if (normalized.empty() ||
            !std::isfinite(candidate.confidence) ||
            candidate.confidence <= 0.0f ||
            candidate.confidence > 1.0f ||
            channel_index < 0 ||
            track_id < 0 ||
            frame_index < 0) {
            return snapshot(channel_index, track_id);
        }

        std::lock_guard<std::mutex> guard(lock_);
        auto& votes = votes_by_track_[{channel_index, track_id}];
        if (votes.latest_frame_index >= 0 && frame_index < votes.latest_frame_index) {
            votes = track_votes {};
        }
        votes.latest_frame_index = std::max(votes.latest_frame_index, frame_index);

        constexpr float confidence_tolerance = 1e-6f;
        const auto found = votes.votes_by_frame.find(frame_index);
        if (found == votes.votes_by_frame.end()) {
            votes.votes_by_frame.emplace(
                frame_index,
                frame_vote {normalized, candidate.confidence, false});
        }
        else {
            auto& vote = found->second;
            if (candidate.confidence > vote.confidence + confidence_tolerance) {
                vote = frame_vote {normalized, candidate.confidence, false};
            }
            else if (std::abs(candidate.confidence - vote.confidence) <= confidence_tolerance &&
                (vote.conflicted || vote.text != normalized)) {
                vote.text.clear();
                vote.conflicted = true;
            }
        }
        return evaluate(votes);
    }

    vp_ocr_consensus_result vp_ocr_consensus::snapshot(int channel_index, int track_id) const {
        std::lock_guard<std::mutex> guard(lock_);
        const auto found = votes_by_track_.find({channel_index, track_id});
        if (found == votes_by_track_.end()) {
            return {};
        }
        return evaluate(found->second);
    }

    void vp_ocr_consensus::clear(int channel_index, int track_id) {
        std::lock_guard<std::mutex> guard(lock_);
        votes_by_track_.erase({channel_index, track_id});
    }

    void vp_ocr_consensus::clear_channel(int channel_index) {
        std::lock_guard<std::mutex> guard(lock_);
        for (auto entry = votes_by_track_.begin(); entry != votes_by_track_.end();) {
            if (entry->first.first == channel_index) {
                entry = votes_by_track_.erase(entry);
            }
            else {
                ++entry;
            }
        }
    }

    void vp_ocr_consensus::reset() {
        std::lock_guard<std::mutex> guard(lock_);
        votes_by_track_.clear();
    }

    std::size_t vp_ocr_consensus::tracked_state_count() const {
        std::lock_guard<std::mutex> guard(lock_);
        return votes_by_track_.size();
    }
}

#endif
