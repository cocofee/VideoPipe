#pragma once

#include <cstdint>
#include <string>

#include "../third_party/nlohmann/json.hpp"

namespace vp_objects {

    // Version 1 is the first stable CycleRace -> VideoPipe event contract.
    constexpr int VP_PASSAGE_EVENT_SCHEMA_VERSION = 1;
    constexpr const char* VP_PASSAGE_EVENT_MESSAGE_TYPE = "passage";

    struct vp_passage_event {
        int schema_version = VP_PASSAGE_EVENT_SCHEMA_VERSION;
        std::string message_type = VP_PASSAGE_EVENT_MESSAGE_TYPE;

        // event_id is the idempotency key. CycleRace must keep it stable when retrying.
        std::string event_id;
        std::string race_id;
        std::string stage_id;
        std::string group_id;

        int sequence = 0;
        std::string chip_id;
        std::string bib;
        std::int64_t passage_time_ms = 0;
        int lap = 0;
        std::string source = "cyclerace";

        // Optional sender timestamp. It is diagnostic only, not the race time authority.
        std::int64_t emitted_at_ms = 0;

        // Monotonic passage correction version. Legacy payloads default to 1.
        int revision = 1;

        void validate() const;
    };

    void to_json(nlohmann::json& json, const vp_passage_event& event);
    void from_json(const nlohmann::json& json, vp_passage_event& event);

}
