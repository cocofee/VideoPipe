#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>
#include <stdexcept>

#include "../utils/race/vp_passage_event_http_server.h"
#include "../third_party/cpp_httplib/httplib.h"

int main() {
    const auto journal_path = std::filesystem::temp_directory_path() / "videopipe_passage_http_events.jsonl";
    std::error_code error;
    std::filesystem::remove(journal_path, error);

    vp_utils::vp_passage_event_store store(journal_path);
    vp_utils::vp_passage_event_ingestor ingestor(store);
    bool fail_first_delivery = true;
    ingestor.set_accepted_hooker([&fail_first_delivery](const vp_objects::vp_passage_event&) {
        if (fail_first_delivery) {
            fail_first_delivery = false;
            throw std::runtime_error("simulated review callback outage");
        }
    });
    vp_utils::vp_passage_event_http_server server("127.0.0.1", 18765, ingestor);
    server.start();

    const std::string payload = R"({
        "schema_version": 1,
        "message_type": "passage",
        "event_id": "http-race-final-0001",
        "race_id": "http-race",
        "stage_id": "final",
        "group_id": "men-open",
        "sequence": 1,
        "chip_id": "chip-23",
        "bib": "23",
        "passage_time_ms": 123456,
        "lap": 1,
        "source": "cyclerace"
    })";

    httplib::Client client(server.listen_host(), server.listen_port());
    const auto first = client.Post("/api/v1/passage-events", payload, "application/json");
    const auto second = client.Post("/api/v1/passage-events", payload, "application/json");
    const auto third = client.Post("/api/v1/passage-events", payload, "application/json");

    const std::string conflict_payload = R"({
        "schema_version": 1,
        "message_type": "passage",
        "event_id": "http-race-final-0001",
        "race_id": "http-race",
        "stage_id": "final",
        "group_id": "men-open",
        "sequence": 1,
        "chip_id": "chip-23",
        "bib": "99",
        "passage_time_ms": 123456,
        "lap": 1,
        "source": "cyclerace"
    })";
    const auto conflict = client.Post(
        "/api/v1/passage-events", conflict_payload, "application/json");
    const auto invalid = client.Post(
        "/api/v1/passage-events", "{invalid", "application/json");

    const bool passed = first && second && third && conflict && invalid &&
        first->status == 503 && second->status == 200 && third->status == 200 &&
        conflict->status == 409 && invalid->status == 400 && store.size() == 1;
    server.stop();
    std::filesystem::remove(journal_path, error);

    if (!passed) {
        std::cerr << "cycle race HTTP sample failed" << std::endl;
        return 1;
    }
    std::cout << "cycle race HTTP sample passed" << std::endl;
    return 0;
}
