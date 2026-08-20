#include <filesystem>
#include <fstream>
#include <iostream>

#include "../utils/race/vp_passage_event_ingestor.h"

int main() {
    const auto journal_path = std::filesystem::temp_directory_path() / "videopipe_passage_events.jsonl";
    std::error_code error;
    std::filesystem::remove(journal_path, error);

    vp_utils::vp_passage_event_store store(journal_path);
    vp_utils::vp_passage_event_ingestor ingestor(store);
    ingestor.set_accepted_hooker([](const vp_objects::vp_passage_event& event) {
        std::cout << "accepted: " << event.event_id << " sequence=" << event.sequence << std::endl;
    });

    const std::string payload = R"({
        "schema_version": 1,
        "message_type": "passage",
        "event_id": "demo-race-final-0001",
        "race_id": "demo-race",
        "stage_id": "final",
        "group_id": "men-open",
        "sequence": 1,
        "chip_id": "chip-23",
        "bib": "23",
        "passage_time_ms": 123456,
        "lap": 1,
        "source": "cyclerace"
    })";

    const auto first = ingestor.ingest_json(payload);
    const auto second = ingestor.ingest_json(payload);
    if (first != vp_utils::vp_passage_event_ingest_result::accepted ||
        second != vp_utils::vp_passage_event_ingest_result::duplicate ||
        store.size() != 1) {
        std::cerr << "event idempotency check failed" << std::endl;
        return 1;
    }

    const std::string conflict_payload = R"({
        "schema_version": 1,
        "message_type": "passage",
        "event_id": "demo-race-final-0001",
        "race_id": "demo-race",
        "stage_id": "final",
        "group_id": "men-open",
        "sequence": 1,
        "chip_id": "chip-23",
        "bib": "99",
        "passage_time_ms": 123456,
        "lap": 1,
        "source": "cyclerace"
    })";
    bool conflict_rejected = false;
    try {
        ingestor.ingest_json(conflict_payload);
    }
    catch (const vp_utils::vp_passage_event_conflict_error&) {
        conflict_rejected = true;
    }
    if (!conflict_rejected) {
        std::cerr << "event id conflict was accepted" << std::endl;
        return 1;
    }

    {
        std::ofstream corrupt_tail(journal_path, std::ios::out | std::ios::app | std::ios::binary);
        corrupt_tail << "{\"schema_version\":1";
    }
    vp_utils::vp_passage_event_store recovered_store(journal_path);
    if (recovered_store.size() != 1 || !recovered_store.contains("demo-race-final-0001") ||
        !recovered_store.recovered_incomplete_tail()) {
        std::cerr << "event recovery check failed" << std::endl;
        return 1;
    }

    const auto malformed_path = journal_path.string() + ".malformed";
    std::filesystem::copy_file(
        journal_path,
        malformed_path,
        std::filesystem::copy_options::overwrite_existing,
        error);
    {
        std::ofstream malformed_tail(
            malformed_path,
            std::ios::out | std::ios::app | std::ios::binary);
        malformed_tail << "{\"schema_version\":1,}";
    }
    const auto malformed_size = std::filesystem::file_size(malformed_path);
    bool malformed_rejected = false;
    try {
        vp_utils::vp_passage_event_store malformed_store(malformed_path);
    }
    catch (const std::exception&) {
        malformed_rejected = true;
    }
    if (!malformed_rejected || std::filesystem::file_size(malformed_path) != malformed_size) {
        std::cerr << "malformed event tail was incorrectly recovered" << std::endl;
        return 1;
    }

    std::filesystem::remove(journal_path, error);
    std::filesystem::remove(malformed_path, error);
    std::cout << "cycle race event sample passed" << std::endl;
    return 0;
}
