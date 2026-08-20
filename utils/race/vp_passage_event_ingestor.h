#pragma once

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "vp_passage_event_store.h"

namespace vp_utils {

    class vp_passage_event_delivery_error: public std::runtime_error {
    public:
        explicit vp_passage_event_delivery_error(const std::string& message):
            std::runtime_error(message) {
        }
    };

    enum class vp_passage_event_ingest_result {
        accepted,
        duplicate,
    };

    // Transport-neutral receiver. HTTP/TCP/WebSocket adapters can feed this class later.
    class vp_passage_event_ingestor {
    private:
        vp_passage_event_store& event_store;
        std::mutex delivery_lock;
        std::function<void(const vp_objects::vp_passage_event&)> accepted_hooker;
        std::unordered_map<std::string, int> delivered_event_revisions;

    public:
        explicit vp_passage_event_ingestor(vp_passage_event_store& event_store);

        void set_accepted_hooker(std::function<void(const vp_objects::vp_passage_event&)> hooker);

        vp_passage_event_ingest_result ingest(const vp_objects::vp_passage_event& event);
        vp_passage_event_ingest_result ingest_json(const std::string& payload);
    };

}
