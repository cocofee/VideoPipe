#include "vp_passage_event.h"

#include <stdexcept>

namespace vp_objects {

    void vp_passage_event::validate() const {
        if (schema_version != VP_PASSAGE_EVENT_SCHEMA_VERSION) {
            throw std::invalid_argument("unsupported passage event schema_version");
        }
        if (message_type != VP_PASSAGE_EVENT_MESSAGE_TYPE) {
            throw std::invalid_argument("invalid passage event message_type");
        }
        if (event_id.empty()) {
            throw std::invalid_argument("passage event event_id is required");
        }
        if (race_id.empty()) {
            throw std::invalid_argument("passage event race_id is required");
        }
        if (stage_id.empty()) {
            throw std::invalid_argument("passage event stage_id is required");
        }
        if (group_id.empty()) {
            throw std::invalid_argument("passage event group_id is required");
        }
        if (sequence <= 0) {
            throw std::invalid_argument("passage event sequence must be positive");
        }
        if (chip_id.empty() && bib.empty()) {
            throw std::invalid_argument("passage event requires chip_id or bib");
        }
        if (passage_time_ms < 0) {
            throw std::invalid_argument("passage event passage_time_ms must not be negative");
        }
        if (lap < 0) {
            throw std::invalid_argument("passage event lap must not be negative");
        }
        if (source.empty()) {
            throw std::invalid_argument("passage event source is required");
        }
        if (emitted_at_ms < 0) {
            throw std::invalid_argument("passage event emitted_at_ms must not be negative");
        }
    }

    void to_json(nlohmann::json& json, const vp_passage_event& event) {
        event.validate();
        json = nlohmann::json{
            {"schema_version", event.schema_version},
            {"message_type", event.message_type},
            {"event_id", event.event_id},
            {"race_id", event.race_id},
            {"stage_id", event.stage_id},
            {"group_id", event.group_id},
            {"sequence", event.sequence},
            {"chip_id", event.chip_id},
            {"bib", event.bib},
            {"passage_time_ms", event.passage_time_ms},
            {"lap", event.lap},
            {"source", event.source},
            {"emitted_at_ms", event.emitted_at_ms},
        };
    }

    void from_json(const nlohmann::json& json, vp_passage_event& event) {
        event.schema_version = json.at("schema_version").get<int>();
        event.message_type = json.at("message_type").get<std::string>();
        event.event_id = json.at("event_id").get<std::string>();
        event.race_id = json.at("race_id").get<std::string>();
        event.stage_id = json.at("stage_id").get<std::string>();
        event.group_id = json.at("group_id").get<std::string>();
        event.sequence = json.at("sequence").get<int>();
        event.chip_id = json.value("chip_id", std::string());
        event.bib = json.value("bib", std::string());
        event.passage_time_ms = json.at("passage_time_ms").get<std::int64_t>();
        event.lap = json.at("lap").get<int>();
        event.source = json.at("source").get<std::string>();
        event.emitted_at_ms = json.value("emitted_at_ms", static_cast<std::int64_t>(0));
        event.validate();
    }

}
