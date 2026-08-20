#include "vp_passage_event_store.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace vp_utils {

    namespace {

        bool same_event(
            const vp_objects::vp_passage_event& left,
            const vp_objects::vp_passage_event& right) {
            return left.schema_version == right.schema_version &&
                left.message_type == right.message_type &&
                left.event_id == right.event_id &&
                left.race_id == right.race_id &&
                left.stage_id == right.stage_id &&
                left.group_id == right.group_id &&
                left.sequence == right.sequence &&
                left.chip_id == right.chip_id &&
                left.bib == right.bib &&
                left.passage_time_ms == right.passage_time_ms &&
                left.lap == right.lap &&
                left.source == right.source &&
                left.emitted_at_ms == right.emitted_at_ms &&
                left.revision == right.revision;
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

    vp_passage_event_store::vp_passage_event_store(std::filesystem::path journal_path):
        journal_path(std::move(journal_path)) {
        if (this->journal_path.empty()) {
            throw std::invalid_argument("passage event journal path is required");
        }

        const auto parent = this->journal_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        load_existing();
    }

    void vp_passage_event_store::load_existing() {
        std::lock_guard<std::mutex> guard(store_lock);
        if (!std::filesystem::exists(journal_path)) {
            return;
        }

        std::ifstream input(journal_path, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("failed to open passage event journal for reading: " + journal_path.string());
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
                    "invalid passage event journal line " + std::to_string(line_number) + ": " + error.what());
            }

            try {
                auto event = json.get<vp_objects::vp_passage_event>();
                const auto existing = event_positions.find(event.event_id);
                if (existing != event_positions.end()) {
                    const auto& current = journal_events[existing->second];
                    if (event.revision < current.revision) {
                        offset = terminated ? line_end + 1 : content.size();
                        continue;
                    }
                    if (event.revision == current.revision) {
                        if (!same_event(current, event)) {
                            throw std::runtime_error(
                                "conflicting passage event revision in journal: " + event.event_id);
                        }
                        offset = terminated ? line_end + 1 : content.size();
                        continue;
                    }
                }
                if (existing != event_positions.end()) {
                    journal_events[existing->second] = std::move(event);
                }
                else {
                    event_positions[event.event_id] = journal_events.size();
                    journal_events.push_back(std::move(event));
                }
            }
            catch (const std::exception& error) {
                throw std::runtime_error(
                    "invalid passage event journal line " + std::to_string(line_number) + ": " + error.what());
            }
            offset = terminated ? line_end + 1 : content.size();
        }
    }

    bool vp_passage_event_store::append(const vp_objects::vp_passage_event& event) {
        event.validate();
        std::lock_guard<std::mutex> guard(store_lock);
        const auto existing = event_positions.find(event.event_id);
        if (existing != event_positions.end()) {
            const auto& current = journal_events[existing->second];
            if (event.revision < current.revision) {
                return false;
            }
            if (event.revision == current.revision && !same_event(current, event)) {
                throw vp_passage_event_conflict_error(
                    "passage event revision was reused with different content: " + event.event_id);
            }
            if (event.revision == current.revision) {
                return false;
            }
        }

        nlohmann::json json = event;
        const auto original_size = std::filesystem::exists(journal_path)
            ? std::filesystem::file_size(journal_path)
            : 0;
        std::ofstream output(journal_path, std::ios::out | std::ios::app | std::ios::binary);
        if (!output.is_open()) {
            throw std::runtime_error("failed to open passage event journal for appending: " + journal_path.string());
        }
        output << json.dump() << '\n';
        output.flush();
        if (!output.good()) {
            output.close();
            std::error_code rollback_error;
            std::filesystem::resize_file(journal_path, original_size, rollback_error);
            throw std::runtime_error("failed to append passage event journal: " + journal_path.string());
        }

        if (existing != event_positions.end()) {
            journal_events[existing->second] = event;
        }
        else {
            event_positions[event.event_id] = journal_events.size();
            journal_events.push_back(event);
        }
        return true;
    }

    bool vp_passage_event_store::contains(const std::string& event_id) const {
        std::lock_guard<std::mutex> guard(store_lock);
        return event_positions.find(event_id) != event_positions.end();
    }

    std::size_t vp_passage_event_store::size() const {
        std::lock_guard<std::mutex> guard(store_lock);
        return journal_events.size();
    }

    std::vector<vp_objects::vp_passage_event> vp_passage_event_store::events() const {
        std::lock_guard<std::mutex> guard(store_lock);
        return journal_events;
    }

    bool vp_passage_event_store::recovered_incomplete_tail() const {
        std::lock_guard<std::mutex> guard(store_lock);
        return recovered_truncated_tail;
    }

    const std::filesystem::path& vp_passage_event_store::path() const {
        return journal_path;
    }

}
