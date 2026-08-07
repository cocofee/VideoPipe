#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../nodes/infers/vp_async_ocr_node.h"
#include "../objects/vp_frame_meta.h"
#include "../objects/vp_frame_target.h"
#include "../objects/vp_ocr_consensus.h"
#include "../objects/vp_ocr_job.h"

namespace {

using vp_nodes::vp_async_ocr_config;
using vp_nodes::vp_async_ocr_dispatcher;
using vp_nodes::vp_ocr_shutdown_mode;
using vp_objects::vp_frame_meta;
using vp_objects::vp_frame_target;
using vp_objects::vp_ocr_candidate;
using vp_objects::vp_ocr_consensus;
using vp_objects::vp_ocr_consensus_status;
using vp_objects::vp_ocr_job;
using vp_objects::vp_rect;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_invalid_argument(
    const std::function<void()>& operation,
    const char* message) {
    try {
        operation();
    }
    catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

vp_async_ocr_config test_config(std::size_t capacity = 4) {
    vp_async_ocr_config config;
    config.capacity = capacity;
    config.min_crop_width = 4;
    config.min_crop_height = 4;
    config.max_crop_width = 80;
    config.max_crop_height = 80;
    config.fallback_padding = 5;
    config.min_candidate_confidence = .1f;
    return config;
}

vp_ocr_job make_job(int frame_index, std::string evidence_path = "evidence.jpg") {
    cv::Mat crop(12, 10, CV_8UC3, cv::Scalar(frame_index, 0, 0));
    cv::Mat padded(16, 14, CV_8UC3, cv::Scalar(frame_index, 1, 0));
    return vp_ocr_job(
        0,
        frame_index,
        7,
        frame_index * 40,
        25.0,
        crop,
        vp_rect(2, 3, crop.cols, crop.rows),
        std::move(evidence_path),
        padded);
}

void verify_validation_and_crop_clone() {
    require_invalid_argument(
        []() {
            auto config = test_config(0);
            vp_async_ocr_dispatcher dispatcher(
                config,
                [](const cv::Mat&) { return vp_ocr_candidate {"1", .9f}; });
        },
        "capacity=0 must fail fast");
    require_invalid_argument(
        []() {
            auto config = test_config();
            config.min_crop_width = 0;
            vp_async_ocr_dispatcher dispatcher(
                config,
                [](const cv::Mat&) { return vp_ocr_candidate {"1", .9f}; });
        },
        "invalid crop dimensions must fail fast");
    require_invalid_argument(
        []() {
            auto config = test_config();
            config.min_candidate_confidence = 1.1f;
            vp_async_ocr_dispatcher dispatcher(
                config,
                [](const cv::Mat&) { return vp_ocr_candidate {"1", .9f}; });
        },
        "invalid confidence threshold must fail fast");
    require_invalid_argument(
        []() {
            vp_async_ocr_dispatcher dispatcher(
                test_config(),
                vp_async_ocr_dispatcher::recognizer_callback {});
        },
        "an empty recognizer must fail fast");
    require_invalid_argument(
        []() {
            vp_ocr_job(0, 1, 2, 40, 0.0, cv::Mat(4, 4, CV_8UC3), vp_rect(0, 0, 4, 4), "x");
        },
        "non-positive FPS must fail fast");
    require_invalid_argument(
        []() {
            vp_ocr_job(0, 1, 2, 40, 25.0, cv::Mat(), vp_rect(0, 0, 4, 4), "x");
        },
        "an empty OCR crop must fail fast");

    cv::Mat crop(4, 4, CV_8UC1, cv::Scalar(7));
    vp_ocr_job job(1, 3, 9, 120, 25.0, crop, vp_rect(0, 0, 4, 4), "proof/frame-3.jpg");
    crop.setTo(cv::Scalar(99));
    require(job.crop().at<unsigned char>(0, 0) == 7, "OCR jobs must clone the crop at construction");
    auto crop_alias = job.crop();
    crop_alias.setTo(cv::Scalar(55));
    require(job.crop().at<unsigned char>(0, 0) == 7, "OCR crop access must not expose the job's internal buffer");
    require(job.evidence_image_path() == "proof/frame-3.jpg", "evidence path must be stored verbatim");
}

void verify_bounded_queue_drop_and_drain_shutdown() {
    std::mutex recognized_mutex;
    std::vector<int> recognized_job_ids;
    vp_async_ocr_dispatcher dispatcher(
        test_config(2),
        [&](const cv::Mat& crop) {
            std::lock_guard<std::mutex> lock(recognized_mutex);
            recognized_job_ids.push_back(crop.at<cv::Vec3b>(0, 0)[0]);
            return vp_ocr_candidate {"17", .9f};
        },
        {},
        {},
        false);

    require(dispatcher.enqueue(make_job(1)), "first job must enqueue");
    require(dispatcher.enqueue(make_job(2)), "second job must enqueue");
    require(dispatcher.enqueue(make_job(3)), "third job must enqueue after dropping the oldest pending job");

    auto stats = dispatcher.statistics();
    require(stats.queue_size == 2, "queue size must never exceed capacity");
    require(stats.enqueued_jobs == 3, "all accepted jobs must be counted");
    require(stats.dropped_jobs == 1, "overflow must deterministically drop one oldest pending job");

    dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);
    dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);

    stats = dispatcher.statistics();
    require(stats.processed_jobs == 2, "drain shutdown must finish every pending job");
    require(stats.queue_size == 0, "drain shutdown must empty the queue");
    require(stats.worker_joined, "drain shutdown must join the worker");
    require(recognized_job_ids == std::vector<int>({2, 3}), "overflow must discard the oldest pending job before OCR");
}

void verify_cancel_before_start_clears_pending_without_callback() {
    std::atomic<int> recognized {0};
    vp_async_ocr_dispatcher dispatcher(
        test_config(3),
        [&](const cv::Mat&) {
            ++recognized;
            return vp_ocr_candidate {"18", .9f};
        },
        {},
        {},
        false);

    dispatcher.enqueue(make_job(1));
    dispatcher.enqueue(make_job(2));
    dispatcher.shutdown(vp_ocr_shutdown_mode::CANCEL);
    dispatcher.shutdown(vp_ocr_shutdown_mode::CANCEL);

    const auto stats = dispatcher.statistics();
    require(stats.queue_size == 0, "cancel before worker start must clear all pending jobs");
    require(stats.canceled_jobs == 2, "cancel before worker start must count every cleared job");
    require(stats.processed_jobs == 0 && recognized == 0, "cancel before worker start must not invoke OCR");
    require(stats.worker_joined, "cancel before worker start must complete shutdown safely");
}

void verify_cancel_shutdown_cancels_pending_and_joins() {
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool worker_entered = false;
    bool release_worker = false;

    vp_async_ocr_dispatcher dispatcher(
        test_config(3),
        [&](const cv::Mat&) {
            std::unique_lock<std::mutex> lock(gate_mutex);
            worker_entered = true;
            gate_cv.notify_all();
            gate_cv.wait(lock, [&]() { return release_worker; });
            return vp_ocr_candidate {"18", .9f};
        });

    dispatcher.enqueue(make_job(1));
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        require(
            gate_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return worker_entered; }),
            "worker did not start the active job");
    }
    dispatcher.enqueue(make_job(2));

    std::thread shutdown_thread([&]() {
        dispatcher.shutdown(vp_ocr_shutdown_mode::CANCEL);
    });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (dispatcher.statistics().canceled_jobs != 1 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    require(dispatcher.statistics().canceled_jobs == 1, "cancel shutdown must discard pending work explicitly");

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_worker = true;
    }
    gate_cv.notify_all();
    shutdown_thread.join();

    const auto stats = dispatcher.statistics();
    require(stats.processed_jobs == 1, "cancel shutdown must allow the already-running callback to finish safely");
    require(stats.queue_size == 0, "cancel shutdown must clear pending jobs");
    require(stats.worker_joined, "cancel shutdown must join the worker");
}

void verify_exception_isolated_and_evidence_preserved() {
    std::mutex result_mutex;
    std::vector<std::string> failed_paths;
    std::vector<std::string> result_paths;
    std::atomic<int> calls {0};

    vp_async_ocr_dispatcher dispatcher(
        test_config(),
        [&](const cv::Mat&) {
            if (++calls == 1) {
                throw std::runtime_error("recognizer failure");
            }
            return vp_ocr_candidate {"21", .8f};
        },
        [&](const vp_nodes::vp_async_ocr_result& result) {
            std::lock_guard<std::mutex> lock(result_mutex);
            result_paths.push_back(result.job.evidence_image_path());
        },
        [&](const vp_ocr_job& job, const std::string&) {
            std::lock_guard<std::mutex> lock(result_mutex);
            failed_paths.push_back(job.evidence_image_path());
        });

    auto frame = std::make_shared<vp_frame_meta>(cv::Mat(64, 64, CV_8UC3), 10, 2, 64, 64, 25);
    auto target = std::make_shared<vp_frame_target>(10, 10, 20, 20, 0, .9f, 10, 2, "cyclist");
    target->track_id = 5;
    frame->targets.push_back(target);

    require(dispatcher.submit(*frame, *target, "proof/failure.jpg"), "valid tracked target must enqueue");
    frame->frame_index = 11;
    target->frame_index = 11;
    require(dispatcher.submit(*frame, *target, "proof/success.jpg"), "worker must accept work after an OCR exception");
    require(dispatcher.wait_until_idle(std::chrono::seconds(2)), "worker did not become idle after exception recovery");
    dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);

    const auto stats = dispatcher.statistics();
    require(stats.processed_jobs == 2, "both failed and successful OCR attempts must be processed");
    require(stats.failed_jobs == 1, "OCR exception must increment failed count");
    require(stats.worker_joined, "worker must exit safely after an OCR exception");
    require(frame->targets.size() == 1 && frame->targets[0] == target, "OCR failure must not remove the source target");
    require(target->track_id == 5, "OCR failure must not rewrite source target metadata");
    require(failed_paths == std::vector<std::string>({"proof/failure.jpg"}), "failed OCR must preserve its evidence path");
    require(result_paths == std::vector<std::string>({"proof/success.jpg"}), "successful OCR must preserve its evidence path");
}

void verify_result_callback_can_request_drain_without_self_join() {
    std::atomic<int> callback_calls {0};
    std::atomic<bool> reentrant_enqueue_succeeded {false};
    std::atomic<bool> callback_shutdown_returned {false};
    vp_async_ocr_dispatcher* dispatcher_ptr = nullptr;

    vp_async_ocr_dispatcher dispatcher(
        test_config(3),
        [](const cv::Mat&) { return vp_ocr_candidate {"31", .9f}; },
        [&](const vp_nodes::vp_async_ocr_result&) {
            const auto call = ++callback_calls;
            const auto snapshot = dispatcher_ptr->statistics();
            require(snapshot.active_jobs == 1, "result callback statistics must be available without deadlock");
            if (call == 1) {
                reentrant_enqueue_succeeded = dispatcher_ptr->enqueue(make_job(2));
                dispatcher_ptr->shutdown(vp_ocr_shutdown_mode::DRAIN);
                callback_shutdown_returned = true;
            }
        },
        {},
        false);
    dispatcher_ptr = &dispatcher;

    dispatcher.enqueue(make_job(1));
    dispatcher.start();
    require(dispatcher.wait_until_idle(std::chrono::seconds(2)), "result callback drain did not become idle");
    dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);

    const auto stats = dispatcher.statistics();
    require(reentrant_enqueue_succeeded, "result callback must be able to enqueue before requesting drain");
    require(callback_shutdown_returned, "result callback shutdown must return without self-join");
    require(callback_calls == 2, "drain requested by result callback must process the reentrant pending job");
    require(stats.processed_jobs == 2, "result callback drain must preserve both jobs");
    require(stats.failed_jobs == 0, "self-shutdown control flow must not be counted as OCR failure");
    require(stats.worker_joined, "external shutdown must finally join a result-callback-stopped worker");
}

void verify_failure_callback_can_request_cancel_without_self_join() {
    std::atomic<bool> callback_shutdown_returned {false};
    std::atomic<bool> callback_stats_read {false};
    vp_async_ocr_dispatcher* dispatcher_ptr = nullptr;

    vp_async_ocr_dispatcher dispatcher(
        test_config(3),
        [](const cv::Mat&) -> vp_ocr_candidate {
            throw std::runtime_error("expected OCR failure");
        },
        {},
        [&](const vp_ocr_job&, const std::string&) {
            callback_stats_read = dispatcher_ptr->statistics().failed_jobs == 1;
            dispatcher_ptr->shutdown(vp_ocr_shutdown_mode::CANCEL);
            callback_shutdown_returned = true;
        },
        false);
    dispatcher_ptr = &dispatcher;

    dispatcher.enqueue(make_job(1));
    dispatcher.enqueue(make_job(2));
    dispatcher.start();
    require(dispatcher.wait_until_idle(std::chrono::seconds(2)), "failure callback cancel did not become idle");
    dispatcher.shutdown(vp_ocr_shutdown_mode::CANCEL);
    dispatcher.shutdown(vp_ocr_shutdown_mode::CANCEL);

    const auto stats = dispatcher.statistics();
    require(callback_stats_read, "failure callback must read statistics without internal-lock deadlock");
    require(callback_shutdown_returned, "failure callback shutdown must return without self-join");
    require(stats.failed_jobs == 1, "recognizer exception must be counted exactly once");
    require(stats.processed_jobs == 1, "cancel from failure callback must stop after the active failed job");
    require(stats.canceled_jobs == 1, "cancel from failure callback must clear the pending job");
    require(stats.worker_joined, "external shutdown must finally join a failure-callback-stopped worker");
}

void verify_dispatcher_can_be_destroyed_from_worker_callback() {
    struct state_lifetime_marker {
        std::promise<void>* released;
        explicit state_lifetime_marker(std::promise<void>* released): released(released) {
        }
        ~state_lifetime_marker() {
            released->set_value();
        }
    };

    std::promise<void> state_released;
    auto state_released_future = state_released.get_future();
    auto marker = std::make_shared<state_lifetime_marker>(&state_released);
    std::unique_ptr<vp_async_ocr_dispatcher> dispatcher;
    dispatcher = std::make_unique<vp_async_ocr_dispatcher>(
        test_config(),
        [marker](const cv::Mat&) { return vp_ocr_candidate {"41", .9f}; },
        [&](const vp_nodes::vp_async_ocr_result&) {
            dispatcher.reset();
        },
        vp_async_ocr_dispatcher::failure_callback {},
        false);
    marker.reset();

    dispatcher->enqueue(make_job(1));
    dispatcher->start();
    require(
        state_released_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
        "worker-owned state must outlive dispatcher destruction and release after worker exit");
    require(!dispatcher, "worker callback must be able to destroy the dispatcher without terminate");
}

void verify_result_callback_shutdown_during_external_join() {
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool callback_entered = false;
    bool release_callback = false;
    std::atomic<bool> callback_shutdown_returned {false};
    std::atomic<bool> external_shutdown_returned {false};
    vp_async_ocr_dispatcher* dispatcher_ptr = nullptr;

    vp_async_ocr_dispatcher dispatcher(
        test_config(),
        [](const cv::Mat&) { return vp_ocr_candidate {"51", .9f}; },
        [&](const vp_nodes::vp_async_ocr_result&) {
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                callback_entered = true;
                gate_cv.notify_all();
                gate_cv.wait(lock, [&]() { return release_callback; });
            }
            dispatcher_ptr->shutdown(vp_ocr_shutdown_mode::DRAIN);
            callback_shutdown_returned = true;
        },
        {},
        false);
    dispatcher_ptr = &dispatcher;
    dispatcher.enqueue(make_job(1));
    dispatcher.start();

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        require(
            gate_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return callback_entered; }),
            "result callback did not enter the external-join gate");
    }

    std::thread external_shutdown([&]() {
        dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);
        external_shutdown_returned = true;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (dispatcher.statistics().accepting && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(!dispatcher.statistics().accepting, "external drain did not request worker stop");
    require(!external_shutdown_returned, "external drain must still be waiting for the gated worker");

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_callback = true;
    }
    gate_cv.notify_all();
    external_shutdown.join();

    const auto stats = dispatcher.statistics();
    require(callback_shutdown_returned, "result callback shutdown must not wait for the external join owner");
    require(external_shutdown_returned, "external drain owner must return after worker exit");
    require(stats.failed_jobs == 0, "join coordination must not be reported as OCR failure");
    require(stats.worker_joined, "external drain owner must mark the worker joined");
}

void verify_failure_callback_shutdown_during_external_join() {
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool callback_entered = false;
    bool release_callback = false;
    std::atomic<bool> callback_shutdown_returned {false};
    std::atomic<bool> external_shutdown_returned {false};
    vp_async_ocr_dispatcher* dispatcher_ptr = nullptr;

    vp_async_ocr_dispatcher dispatcher(
        test_config(),
        [](const cv::Mat&) -> vp_ocr_candidate {
            throw std::runtime_error("expected gated failure");
        },
        {},
        [&](const vp_ocr_job&, const std::string&) {
            {
                std::unique_lock<std::mutex> lock(gate_mutex);
                callback_entered = true;
                gate_cv.notify_all();
                gate_cv.wait(lock, [&]() { return release_callback; });
            }
            dispatcher_ptr->shutdown(vp_ocr_shutdown_mode::CANCEL);
            callback_shutdown_returned = true;
        },
        false);
    dispatcher_ptr = &dispatcher;
    dispatcher.enqueue(make_job(1));
    dispatcher.start();

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        require(
            gate_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return callback_entered; }),
            "failure callback did not enter the external-join gate");
    }

    std::thread external_shutdown([&]() {
        dispatcher.shutdown(vp_ocr_shutdown_mode::CANCEL);
        external_shutdown_returned = true;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (dispatcher.statistics().accepting && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(!dispatcher.statistics().accepting, "external cancel did not request worker stop");
    require(!external_shutdown_returned, "external cancel must still be waiting for the gated worker");

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release_callback = true;
    }
    gate_cv.notify_all();
    external_shutdown.join();

    const auto stats = dispatcher.statistics();
    require(callback_shutdown_returned, "failure callback shutdown must not wait for the external join owner");
    require(external_shutdown_returned, "external cancel owner must return after worker exit");
    require(stats.failed_jobs == 1, "gated recognizer failure must be counted exactly once");
    require(stats.worker_joined, "external cancel owner must mark the worker joined");
}

void verify_filtering_clamp_and_single_fallback() {
    std::mutex result_mutex;
    std::vector<cv::Size> recognized_sizes;
    std::vector<vp_nodes::vp_async_ocr_result> results;

    vp_async_ocr_dispatcher dispatcher(
        test_config(),
        [&](const cv::Mat& crop) {
            std::lock_guard<std::mutex> lock(result_mutex);
            recognized_sizes.push_back(crop.size());
            if (recognized_sizes.size() == 1) {
                return vp_ocr_candidate {"bib-x", .9f};
            }
            return vp_ocr_candidate {"  314  ", .85f};
        },
        [&](const vp_nodes::vp_async_ocr_result& result) {
            std::lock_guard<std::mutex> lock(result_mutex);
            results.push_back(result);
        });

    vp_frame_meta frame(cv::Mat(100, 100, CV_8UC3), 4, 3, 100, 100, 25);
    vp_frame_target target(-5, 10, 30, 20, 0, .9f, 4, 3, "cyclist");
    target.track_id = 44;
    require(dispatcher.submit(frame, target, "proof/clamped.jpg"), "clamped tracked crop must enqueue");
    require(dispatcher.wait_until_idle(std::chrono::seconds(2)), "fallback OCR did not finish");
    dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);

    require(recognized_sizes.size() == 2, "invalid tight OCR must trigger exactly one padded fallback");
    require(recognized_sizes[0] == cv::Size(25, 20), "tight crop must use the clamped source box");
    require(recognized_sizes[1] == cv::Size(30, 30), "fallback crop must add context and remain clamped to the frame");
    require(results.size() == 1, "only the valid fallback candidate must be published");
    require(results[0].used_padded_fallback, "result must identify padded fallback usage");
    require(results[0].candidate.text == "314", "published numeric candidate must be normalized");
    require(results[0].job.source_box().x == 0 && results[0].job.source_box().width == 25, "job source box must be clamped");
    require(results[0].job.evidence_image_path() == "proof/clamped.jpg", "fallback must preserve evidence path");

    std::atomic<int> rejected_calls {0};
    vp_async_ocr_dispatcher rejected_dispatcher(
        test_config(),
        [&](const cv::Mat&) {
            ++rejected_calls;
            return vp_ocr_candidate {"1", .9f};
        });
    vp_frame_target no_track(10, 10, 20, 20, 0, .9f, 4, 3);
    vp_frame_target too_small(10, 10, 2, 2, 0, .9f, 4, 3);
    too_small.track_id = 1;
    vp_frame_target too_large(0, 0, 90, 90, 0, .9f, 4, 3);
    too_large.track_id = 2;
    require(!rejected_dispatcher.submit(frame, no_track, "no-track.jpg"), "untracked target must not enqueue");
    require(!rejected_dispatcher.submit(frame, too_small, "small.jpg"), "undersized crop must not enqueue");
    require(!rejected_dispatcher.submit(frame, too_large, "large.jpg"), "oversized crop must not enqueue");
    rejected_dispatcher.shutdown(vp_ocr_shutdown_mode::DRAIN);
    require(rejected_calls == 0, "rejected targets must never call OCR");
    require(rejected_dispatcher.statistics().enqueued_jobs == 0, "rejected targets must not affect queue counts");
}

void verify_consensus_requires_distinct_consistent_frames() {
    vp_ocr_consensus consensus;

    require(vp_ocr_consensus::normalize_numeric(" 12 34 ") == "1234", "whitespace must be removed from numeric OCR");
    require(vp_ocr_consensus::normalize_numeric("12A4").empty(), "mixed non-numeric OCR must be rejected");
    require(vp_ocr_consensus::normalize_numeric("   ").empty(), "empty normalized OCR must be rejected");

    auto result = consensus.observe(0, 9, 1, {"88", .8f});
    require(result.status == vp_ocr_consensus_status::PENDING, "one frame must never finalize a bib");
    result = consensus.observe(0, 9, 1, {"88", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "duplicate OCR from the same frame must remain pending");
    result = consensus.observe(0, 9, 2, {"89", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "equal conflicting votes must remain pending");
    result = consensus.observe(0, 9, 3, {"88", .8f});
    require(result.status == vp_ocr_consensus_status::CONFIRMED, "two distinct consistent frames with a unique weighted lead must confirm");
    require(result.text == "88", "confirmed consensus must return the winning numeric text");
    require(result.supporting_frames == 2, "confirmation must report distinct supporting frames");

    const auto other_track = consensus.observe(0, 10, 1, {"88", .9f});
    require(other_track.status == vp_ocr_consensus_status::PENDING, "consensus must be isolated by track ID");
    const auto other_channel = consensus.observe(1, 9, 2, {"88", .9f});
    require(other_channel.status == vp_ocr_consensus_status::PENDING, "consensus must be isolated by channel");

    vp_ocr_consensus frame_unique_consensus;
    result = frame_unique_consensus.observe(0, 20, 1, {"88", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "the first vote must remain pending");
    result = frame_unique_consensus.observe(0, 20, 1, {"89", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "equal conflicting text in one frame must invalidate that frame");
    result = frame_unique_consensus.observe(0, 20, 2, {"88", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "one clean frame plus one conflicted frame must remain pending");
    result = frame_unique_consensus.observe(0, 20, 3, {"88", .9f});
    require(result.status == vp_ocr_consensus_status::CONFIRMED, "two clean frames after an excluded conflict must confirm");
    require(result.supporting_frames == 2, "a conflicted frame must not support either text");
}

void verify_consensus_lifecycle_and_track_id_reuse() {
    vp_ocr_consensus consensus;
    consensus.observe(-1, 7, 1, {"123", .9f});
    consensus.observe(0, -1, 1, {"123", .9f});
    consensus.observe(0, 7, -1, {"123", .9f});
    consensus.observe(0, 7, 1, {"invalid", .9f});
    require(consensus.tracked_state_count() == 0, "invalid observations must not allocate consensus state");

    auto result = consensus.observe(0, 7, 100, {"123", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "first lifecycle vote must remain pending");
    result = consensus.observe(0, 7, 101, {"123", .9f});
    require(result.status == vp_ocr_consensus_status::CONFIRMED, "two clean lifecycle votes must confirm");

    result = consensus.observe(0, 7, 0, {"999", .9f});
    require(result.status == vp_ocr_consensus_status::PENDING, "frame rewind must reset reused track consensus");
    require(result.text == "999" && result.supporting_frames == 1, "reused track must retain only the new lifecycle vote");

    consensus.observe(0, 8, 5, {"8", .8f});
    consensus.observe(1, 7, 5, {"7", .8f});
    require(consensus.tracked_state_count() == 3, "valid track states must be countable for lifecycle management");
    consensus.clear(0, 7);
    require(consensus.tracked_state_count() == 2, "clear must remove only the selected channel and track");
    require(consensus.snapshot(0, 7).text.empty(), "cleared track must have no retained winner");
    consensus.clear_channel(0);
    require(consensus.tracked_state_count() == 1, "clear_channel must preserve other channels");
    consensus.reset();
    require(consensus.tracked_state_count() == 0, "reset must clear every consensus state");

    constexpr int concurrent_tracks = 8;
    std::vector<std::thread> workers;
    for (int track_id = 0; track_id < concurrent_tracks; ++track_id) {
        workers.emplace_back([&, track_id]() {
            consensus.observe(2, track_id, 1, {std::to_string(track_id), .7f});
            consensus.observe(2, track_id, 2, {std::to_string(track_id), .8f});
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    require(consensus.tracked_state_count() == concurrent_tracks, "concurrent observations must preserve every track state");
    for (int track_id = 0; track_id < concurrent_tracks; ++track_id) {
        require(
            consensus.snapshot(2, track_id).status == vp_ocr_consensus_status::CONFIRMED,
            "concurrent per-track votes must remain internally consistent");
    }
}

}  // namespace

int main() {
    try {
        verify_validation_and_crop_clone();
        verify_bounded_queue_drop_and_drain_shutdown();
        verify_cancel_before_start_clears_pending_without_callback();
        verify_cancel_shutdown_cancels_pending_and_joins();
        verify_exception_isolated_and_evidence_preserved();
        verify_result_callback_can_request_drain_without_self_join();
        verify_failure_callback_can_request_cancel_without_self_join();
        verify_dispatcher_can_be_destroyed_from_worker_callback();
        verify_result_callback_shutdown_during_external_join();
        verify_failure_callback_shutdown_during_external_join();
        verify_filtering_clamp_and_single_fallback();
        verify_consensus_requires_distinct_consistent_frames();
        verify_consensus_lifecycle_and_track_id_reuse();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    std::cout << "Async OCR dispatch tests passed" << std::endl;
    return 0;
}
