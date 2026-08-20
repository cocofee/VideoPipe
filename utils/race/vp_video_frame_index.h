#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../third_party/nlohmann/json.hpp"

namespace vp_utils {

    constexpr int VP_VIDEO_FRAME_INDEX_SCHEMA_VERSION = 1;

    struct vp_video_frame_index_entry {
        int schema_version = VP_VIDEO_FRAME_INDEX_SCHEMA_VERSION;
        std::string entry_id;
        std::string camera_id;
        int channel_index = -1;
        int source_session = -1;
        std::int64_t source_pts_us = -1;
        std::int64_t capture_monotonic_us = -1;
        std::int64_t capture_wall_time_ms = -1;
        int frame_index = -1;
        std::string video_path;
        std::int64_t video_frame_index = -1;

        void validate() const;
    };

    // Build a stable identity for one persisted frame checkpoint.
    std::string make_video_frame_index_entry_id(const vp_video_frame_index_entry& entry);

    void to_json(nlohmann::json& json, const vp_video_frame_index_entry& entry);
    void from_json(const nlohmann::json& json, vp_video_frame_index_entry& entry);

    class vp_video_frame_index_store {
    private:
        std::filesystem::path journal_path;
        mutable std::mutex store_lock;
        std::vector<vp_video_frame_index_entry> journal_entries;
        std::unordered_set<std::string> entry_ids;
        std::map<std::pair<std::string, int>, std::vector<std::size_t>> camera_channel_entry_positions;
        bool recovered_truncated_tail = false;

        void load_existing();
        void index_entry(std::size_t position);

    public:
        explicit vp_video_frame_index_store(std::filesystem::path journal_path);

        vp_video_frame_index_store(const vp_video_frame_index_store&) = delete;
        vp_video_frame_index_store& operator=(const vp_video_frame_index_store&) = delete;

        // Returns false when entry_id was already persisted.
        bool append(const vp_video_frame_index_entry& entry);
        std::size_t append_batch(const std::vector<vp_video_frame_index_entry>& entries);
        bool contains(const std::string& entry_id) const;
        std::size_t size() const;
        bool recovered_incomplete_tail() const;

        std::optional<vp_video_frame_index_entry> nearest(
            const std::string& camera_id,
            int channel_index,
            std::int64_t capture_wall_time_ms) const;

        std::vector<vp_video_frame_index_entry> range(
            const std::string& camera_id,
            int channel_index,
            std::int64_t start_wall_time_ms,
            std::int64_t end_wall_time_ms) const;

        const std::filesystem::path& path() const;
    };

}
