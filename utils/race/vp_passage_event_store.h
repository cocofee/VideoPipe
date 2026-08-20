#pragma once

#include <cstddef>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../objects/vp_passage_event.h"

namespace vp_utils {

    class vp_passage_event_conflict_error: public std::runtime_error {
    public:
        explicit vp_passage_event_conflict_error(const std::string& message):
            std::runtime_error(message) {
        }
    };

    // Append-only local event journal used by VideoPipe before a remote transport is added.
    class vp_passage_event_store {
    private:
        std::filesystem::path journal_path;
        mutable std::mutex store_lock;
        std::vector<vp_objects::vp_passage_event> journal_events;
        std::unordered_map<std::string, std::size_t> event_positions;
        bool recovered_truncated_tail = false;

        void load_existing();

    public:
        explicit vp_passage_event_store(std::filesystem::path journal_path);

        vp_passage_event_store(const vp_passage_event_store&) = delete;
        vp_passage_event_store& operator=(const vp_passage_event_store&) = delete;

        // Returns false when this event revision was already persisted or is stale.
        bool append(const vp_objects::vp_passage_event& event);
        bool contains(const std::string& event_id) const;
        std::size_t size() const;
        std::vector<vp_objects::vp_passage_event> events() const;
        bool recovered_incomplete_tail() const;
        const std::filesystem::path& path() const;
    };

}
