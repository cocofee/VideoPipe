#include "vp_video_frame_index.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace vp_utils {

    namespace {

        void append_identity_part(std::string& identity, const std::string& value) {
            identity += std::to_string(value.size());
            identity += ':';
            identity += value;
            identity += ':';
        }

        bool is_incomplete_json_tail(const std::string& line) {
            bool in_string = false;
            bool escaped = false;
            int nesting = 0;
            for (const auto character: line) {
                if (in_string) {
                    if (escaped) {
                        escaped = false;
                    }
                    else if (character == '\\') {
                        escaped = true;
                    }
                    else if (character == '"') {
                        in_string = false;
                    }
                    continue;
                }

                if (character == '"') {
                    in_string = true;
                }
                else if (character == '{' || character == '[') {
                    ++nesting;
                }
                else if (character == '}' || character == ']') {
                    if (nesting == 0) {
                        return false;
                    }
                    --nesting;
                }
            }
            return in_string || nesting > 0;
        }

    }

    std::string make_video_frame_index_entry_id(const vp_video_frame_index_entry& entry) {
        std::string identity;
        append_identity_part(identity, entry.camera_id);
        append_identity_part(identity, std::to_string(entry.channel_index));
        append_identity_part(identity, std::to_string(entry.source_session));
        append_identity_part(identity, std::to_string(entry.frame_index));
        append_identity_part(identity, entry.video_path);
        append_identity_part(identity, std::to_string(entry.video_frame_index));
        return identity;
    }

    void vp_video_frame_index_entry::validate() const {
        if (schema_version != VP_VIDEO_FRAME_INDEX_SCHEMA_VERSION) {
            throw std::invalid_argument("unsupported video frame index schema_version");
        }
        if (entry_id.empty()) {
            throw std::invalid_argument("video frame index entry_id is required");
        }
        if (camera_id.empty()) {
            throw std::invalid_argument("video frame index camera_id is required");
        }
        if (channel_index < 0) {
            throw std::invalid_argument("video frame index channel_index must not be negative");
        }
        if (source_session < 0) {
            throw std::invalid_argument("video frame index source_session must not be negative");
        }
        if (source_pts_us < -1) {
            throw std::invalid_argument("video frame index source_pts_us is invalid");
        }
        if (capture_monotonic_us < 0) {
            throw std::invalid_argument("video frame index capture_monotonic_us is required");
        }
        if (capture_wall_time_ms < 0) {
            throw std::invalid_argument("video frame index capture_wall_time_ms is required");
        }
        if (frame_index < 0) {
            throw std::invalid_argument("video frame index frame_index must not be negative");
        }
        if (video_path.empty()) {
            throw std::invalid_argument("video frame index video_path is required");
        }
        if (video_frame_index < 0) {
            throw std::invalid_argument("video frame index video_frame_index must not be negative");
        }
    }

    void to_json(nlohmann::json& json, const vp_video_frame_index_entry& entry) {
        entry.validate();
        json = nlohmann::json{
            {"schema_version", entry.schema_version},
            {"entry_id", entry.entry_id},
            {"camera_id", entry.camera_id},
            {"channel_index", entry.channel_index},
            {"source_session", entry.source_session},
            {"source_pts_us", entry.source_pts_us},
            {"capture_monotonic_us", entry.capture_monotonic_us},
            {"capture_wall_time_ms", entry.capture_wall_time_ms},
            {"frame_index", entry.frame_index},
            {"video_path", entry.video_path},
            {"video_frame_index", entry.video_frame_index},
        };
    }

    void from_json(const nlohmann::json& json, vp_video_frame_index_entry& entry) {
        entry.schema_version = json.at("schema_version").get<int>();
        entry.entry_id = json.at("entry_id").get<std::string>();
        entry.camera_id = json.at("camera_id").get<std::string>();
        entry.channel_index = json.at("channel_index").get<int>();
        entry.source_session = json.at("source_session").get<int>();
        entry.source_pts_us = json.at("source_pts_us").get<std::int64_t>();
        entry.capture_monotonic_us = json.at("capture_monotonic_us").get<std::int64_t>();
        entry.capture_wall_time_ms = json.at("capture_wall_time_ms").get<std::int64_t>();
        entry.frame_index = json.at("frame_index").get<int>();
        entry.video_path = json.at("video_path").get<std::string>();
        entry.video_frame_index = json.at("video_frame_index").get<std::int64_t>();
        entry.validate();
    }

    vp_video_frame_index_store::vp_video_frame_index_store(std::filesystem::path journal_path):
        journal_path(std::move(journal_path)) {
        if (this->journal_path.empty()) {
            throw std::invalid_argument("video frame index journal path is required");
        }
        const auto parent = this->journal_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        load_existing();
    }

    void vp_video_frame_index_store::index_entry(std::size_t position) {
        const auto& entry = journal_entries[position];
        auto& positions = camera_channel_entry_positions[{entry.camera_id, entry.channel_index}];
        const auto insertion = std::upper_bound(
            positions.begin(),
            positions.end(),
            journal_entries[position].capture_wall_time_ms,
            [this](std::int64_t wall_time, std::size_t existing_position) {
                return wall_time < journal_entries[existing_position].capture_wall_time_ms;
            });
        positions.insert(insertion, position);
    }

    void vp_video_frame_index_store::load_existing() {
        std::lock_guard<std::mutex> guard(store_lock);
        if (!std::filesystem::exists(journal_path)) {
            return;
        }

        std::ifstream input(journal_path, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("failed to open video frame index journal for reading: " + journal_path.string());
        }

        const std::string content{
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
        input.close();
        std::size_t offset = 0;
        std::size_t line_number = 0;
        while (offset < content.size()) {
            const auto line_end = content.find('\n', offset);
            const auto terminated = line_end != std::string::npos;
            auto line = content.substr(
                offset,
                terminated ? line_end - offset : content.size() - offset);
            ++line_number;
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                offset = terminated ? line_end + 1 : content.size();
                continue;
            }
            nlohmann::json json;
            try {
                json = nlohmann::json::parse(line);
            }
            catch (const nlohmann::json::parse_error& error) {
                if (!terminated && is_incomplete_json_tail(line)) {
                    std::filesystem::resize_file(journal_path, offset);
                    recovered_truncated_tail = true;
                    return;
                }
                throw std::runtime_error(
                    "invalid video frame index journal line " + std::to_string(line_number) + ": " + error.what());
            }

            try {
                auto entry = json.get<vp_video_frame_index_entry>();
                if (!entry_ids.insert(entry.entry_id).second) {
                    throw std::runtime_error("duplicate entry_id in video frame index journal: " + entry.entry_id);
                }
                journal_entries.push_back(std::move(entry));
                index_entry(journal_entries.size() - 1);
            }
            catch (const std::exception& error) {
                throw std::runtime_error(
                    "invalid video frame index journal line " + std::to_string(line_number) + ": " + error.what());
            }
            offset = terminated ? line_end + 1 : content.size();
        }
    }

    bool vp_video_frame_index_store::append(const vp_video_frame_index_entry& entry) {
        return append_batch({entry}) == 1;
    }

    std::size_t vp_video_frame_index_store::append_batch(
        const std::vector<vp_video_frame_index_entry>& entries) {
        for (const auto& entry: entries) {
            entry.validate();
        }

        std::lock_guard<std::mutex> guard(store_lock);
        std::vector<vp_video_frame_index_entry> pending;
        std::unordered_set<std::string> pending_ids;
        pending.reserve(entries.size());
        for (const auto& entry: entries) {
            if (entry_ids.find(entry.entry_id) != entry_ids.end()) {
                continue;
            }
            if (!pending_ids.insert(entry.entry_id).second) {
                throw std::invalid_argument(
                    "duplicate entry_id in video frame index batch: " + entry.entry_id);
            }
            pending.push_back(entry);
        }
        if (pending.empty()) {
            return 0;
        }

        std::ostringstream payload;
        for (const auto& entry: pending) {
            payload << nlohmann::json(entry).dump() << '\n';
        }
        const auto serialized = payload.str();
        const auto original_size = std::filesystem::exists(journal_path)
            ? std::filesystem::file_size(journal_path)
            : 0;
        std::ofstream output(journal_path, std::ios::out | std::ios::app | std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("failed to open video frame index journal for appending: " + journal_path.string());
        }
        output.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
        output.flush();
        if (!output.good()) {
            output.close();
            std::error_code rollback_error;
            std::filesystem::resize_file(journal_path, original_size, rollback_error);
            throw std::runtime_error("failed to append video frame index journal: " + journal_path.string());
        }

        for (const auto& entry: pending) {
            entry_ids.insert(entry.entry_id);
            journal_entries.push_back(entry);
            index_entry(journal_entries.size() - 1);
        }
        return pending.size();
    }

    bool vp_video_frame_index_store::contains(const std::string& entry_id) const {
        std::lock_guard<std::mutex> guard(store_lock);
        return entry_ids.find(entry_id) != entry_ids.end();
    }

    std::size_t vp_video_frame_index_store::size() const {
        std::lock_guard<std::mutex> guard(store_lock);
        return journal_entries.size();
    }

    bool vp_video_frame_index_store::recovered_incomplete_tail() const {
        std::lock_guard<std::mutex> guard(store_lock);
        return recovered_truncated_tail;
    }

    std::optional<vp_video_frame_index_entry> vp_video_frame_index_store::nearest(
        const std::string& camera_id,
        int channel_index,
        std::int64_t capture_wall_time_ms) const {
        std::lock_guard<std::mutex> guard(store_lock);
        const auto channel = camera_channel_entry_positions.find({camera_id, channel_index});
        if (channel == camera_channel_entry_positions.end() || channel->second.empty()) {
            return std::nullopt;
        }

        const auto& positions = channel->second;
        const auto upper = std::lower_bound(
            positions.begin(),
            positions.end(),
            capture_wall_time_ms,
            [this](std::size_t position, std::int64_t wall_time) {
                return journal_entries[position].capture_wall_time_ms < wall_time;
            });
        if (upper == positions.begin()) {
            return journal_entries[*upper];
        }
        if (upper == positions.end()) {
            return journal_entries[positions.back()];
        }

        const auto after_position = *upper;
        const auto before_position = *(upper - 1);
        const auto before_delta = capture_wall_time_ms - journal_entries[before_position].capture_wall_time_ms;
        const auto after_delta = journal_entries[after_position].capture_wall_time_ms - capture_wall_time_ms;
        return journal_entries[before_delta <= after_delta ? before_position : after_position];
    }

    std::vector<vp_video_frame_index_entry> vp_video_frame_index_store::range(
        const std::string& camera_id,
        int channel_index,
        std::int64_t start_wall_time_ms,
        std::int64_t end_wall_time_ms) const {
        if (start_wall_time_ms > end_wall_time_ms) {
            throw std::invalid_argument("video frame index range start must not exceed end");
        }

        std::lock_guard<std::mutex> guard(store_lock);
        const auto channel = camera_channel_entry_positions.find({camera_id, channel_index});
        if (channel == camera_channel_entry_positions.end()) {
            return {};
        }

        const auto& positions = channel->second;
        const auto first = std::lower_bound(
            positions.begin(), positions.end(), start_wall_time_ms,
            [this](std::size_t position, std::int64_t wall_time) {
                return journal_entries[position].capture_wall_time_ms < wall_time;
            });
        const auto last = std::upper_bound(
            positions.begin(), positions.end(), end_wall_time_ms,
            [this](std::int64_t wall_time, std::size_t position) {
                return wall_time < journal_entries[position].capture_wall_time_ms;
            });

        std::vector<vp_video_frame_index_entry> result;
        result.reserve(static_cast<std::size_t>(std::distance(first, last)));
        for (auto position = first; position != last; ++position) {
            result.push_back(journal_entries[*position]);
        }
        return result;
    }

    const std::filesystem::path& vp_video_frame_index_store::path() const {
        return journal_path;
    }

}
