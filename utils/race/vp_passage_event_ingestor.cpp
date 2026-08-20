#include "vp_passage_event_ingestor.h"

#include <utility>

namespace vp_utils {

    vp_passage_event_ingestor::vp_passage_event_ingestor(vp_passage_event_store& event_store):
        event_store(event_store) {
    }

    void vp_passage_event_ingestor::set_accepted_hooker(
        std::function<void(const vp_objects::vp_passage_event&)> hooker) {
        std::lock_guard<std::mutex> guard(delivery_lock);
        accepted_hooker = std::move(hooker);
        delivered_event_revisions.clear();
    }

    vp_passage_event_ingest_result vp_passage_event_ingestor::ingest(
        const vp_objects::vp_passage_event& event) {
        const auto appended = event_store.append(event);
        std::lock_guard<std::mutex> guard(delivery_lock);
        const auto delivered = delivered_event_revisions.find(event.event_id);
        const auto already_delivered = delivered != delivered_event_revisions.end()
            && delivered->second >= event.revision;
        if (accepted_hooker && !already_delivered) {
            try {
                accepted_hooker(event);
                delivered_event_revisions[event.event_id] = event.revision;
            }
            catch (const std::exception& error) {
                throw vp_passage_event_delivery_error(error.what());
            }
            catch (...) {
                throw vp_passage_event_delivery_error("passage event callback failed");
            }
        }
        return appended
            ? vp_passage_event_ingest_result::accepted
            : vp_passage_event_ingest_result::duplicate;
    }

    vp_passage_event_ingest_result vp_passage_event_ingestor::ingest_json(const std::string& payload) {
        const auto json = nlohmann::json::parse(payload);
        const auto event = json.get<vp_objects::vp_passage_event>();
        return ingest(event);
    }

}
