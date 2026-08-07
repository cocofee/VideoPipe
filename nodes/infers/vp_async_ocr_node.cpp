#include "vp_async_ocr_node.h"

#if defined(VP_WITH_ASYNC_OCR) || defined(VP_ASYNC_OCR_CORE_ONLY)

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace vp_nodes {
    struct vp_async_ocr_dispatcher::worker_state {
        const vp_async_ocr_config config;
        const recognizer_callback recognizer;
        const result_callback result_callback_fn;
        const failure_callback failure_callback_fn;

        mutable std::mutex lock;
        std::condition_variable queue_condition;
        std::condition_variable idle_condition;
        std::deque<vp_objects::vp_ocr_job> queue;
        vp_async_ocr_statistics statistics;
        bool stop_requested = false;
        bool cancel_pending = false;

        std::mutex join_lock;
        std::condition_variable join_condition;
        bool join_in_progress = false;
        bool join_complete = false;
        std::thread::id worker_thread_id;

        worker_state(
            vp_async_ocr_config config,
            recognizer_callback recognizer,
            result_callback result_callback,
            failure_callback failure_callback):
            config(std::move(config)),
            recognizer(std::move(recognizer)),
            result_callback_fn(std::move(result_callback)),
            failure_callback_fn(std::move(failure_callback)) {
            statistics.accepting = true;
        }
    };

    void vp_async_ocr_dispatcher::validate_config(const vp_async_ocr_config& config) {
        if (config.capacity == 0) {
            throw std::invalid_argument("async OCR queue capacity must be positive");
        }
        if (config.min_crop_width <= 0 || config.min_crop_height <= 0 ||
            config.max_crop_width < config.min_crop_width ||
            config.max_crop_height < config.min_crop_height) {
            throw std::invalid_argument("async OCR crop dimensions are invalid");
        }
        if (config.fallback_padding < 0) {
            throw std::invalid_argument("async OCR fallback padding must be non-negative");
        }
        if (!std::isfinite(config.min_candidate_confidence) ||
            config.min_candidate_confidence < 0.0f ||
            config.min_candidate_confidence > 1.0f) {
            throw std::invalid_argument("async OCR confidence threshold must be between 0 and 1");
        }
    }

    vp_async_ocr_dispatcher::vp_async_ocr_dispatcher(
        vp_async_ocr_config config,
        recognizer_callback recognizer,
        result_callback result_callback,
        failure_callback failure_callback,
        bool start_immediately):
        state_(std::make_shared<worker_state>(
            std::move(config),
            std::move(recognizer),
            std::move(result_callback),
            std::move(failure_callback))) {
        validate_config(state_->config);
        if (!state_->recognizer) {
            throw std::invalid_argument("async OCR recognizer callback must not be empty");
        }
        if (start_immediately) {
            start();
        }
    }

    vp_async_ocr_dispatcher::~vp_async_ocr_dispatcher() {
        shutdown(vp_ocr_shutdown_mode::CANCEL);
        const auto state = state_;
        std::lock_guard<std::mutex> join_guard(state->join_lock);
        if (worker_.joinable() && is_worker_thread(state)) {
            worker_.detach();
        }
    }

    void vp_async_ocr_dispatcher::start() {
        const auto state = state_;
        std::lock_guard<std::mutex> join_guard(state->join_lock);
        std::lock_guard<std::mutex> guard(state->lock);
        if (state->statistics.worker_started) {
            return;
        }
        if (state->stop_requested) {
            throw std::logic_error("async OCR worker cannot start after shutdown");
        }
        state->statistics.worker_started = true;
        state->statistics.worker_joined = false;
        worker_ = std::thread(&vp_async_ocr_dispatcher::worker_run, state);
    }

    bool vp_async_ocr_dispatcher::enqueue(vp_objects::vp_ocr_job job) {
        const auto state = state_;
        {
            std::lock_guard<std::mutex> guard(state->lock);
            if (!state->statistics.accepting || state->stop_requested) {
                return false;
            }
            if (state->queue.size() == state->config.capacity) {
                state->queue.pop_front();
                ++state->statistics.dropped_jobs;
            }
            state->queue.push_back(std::move(job));
            ++state->statistics.enqueued_jobs;
            state->statistics.queue_size = state->queue.size();
        }
        state->queue_condition.notify_one();
        return true;
    }

    bool vp_async_ocr_dispatcher::submit(
        const vp_objects::vp_frame_meta& frame_meta,
        const vp_objects::vp_frame_target& target,
        const std::string& evidence_image_path) {
        const auto state = state_;
        if (target.track_id < 0) {
            return false;
        }
        if (frame_meta.frame.empty()) {
            throw std::invalid_argument("async OCR source frame must not be empty");
        }
        if (frame_meta.frame_index < 0 || frame_meta.channel_index < 0 || frame_meta.fps <= 0) {
            throw std::invalid_argument("async OCR frame metadata must include valid channel, frame index, and FPS");
        }

        const auto frame_width = static_cast<long long>(frame_meta.frame.cols);
        const auto frame_height = static_cast<long long>(frame_meta.frame.rows);
        const auto left = std::clamp(static_cast<long long>(target.x), 0LL, frame_width);
        const auto top = std::clamp(static_cast<long long>(target.y), 0LL, frame_height);
        const auto right = std::clamp(
            static_cast<long long>(target.x) + static_cast<long long>(target.width),
            0LL,
            frame_width);
        const auto bottom = std::clamp(
            static_cast<long long>(target.y) + static_cast<long long>(target.height),
            0LL,
            frame_height);
        const auto width = right - left;
        const auto height = bottom - top;

        if (width < state->config.min_crop_width || height < state->config.min_crop_height ||
            width > state->config.max_crop_width || height > state->config.max_crop_height) {
            return false;
        }

        const vp_objects::vp_rect source_box(
            static_cast<int>(left),
            static_cast<int>(top),
            static_cast<int>(width),
            static_cast<int>(height));
        const cv::Rect tight_roi(source_box.x, source_box.y, source_box.width, source_box.height);
        const auto tight_crop = frame_meta.frame(tight_roi);

        cv::Mat padded_crop;
        if (state->config.fallback_padding > 0) {
            const auto padded_left = std::max(0LL, left - state->config.fallback_padding);
            const auto padded_top = std::max(0LL, top - state->config.fallback_padding);
            const auto padded_right = std::min(frame_width, right + state->config.fallback_padding);
            const auto padded_bottom = std::min(frame_height, bottom + state->config.fallback_padding);
            if (padded_left != left || padded_top != top || padded_right != right || padded_bottom != bottom) {
                padded_crop = frame_meta.frame(cv::Rect(
                    static_cast<int>(padded_left),
                    static_cast<int>(padded_top),
                    static_cast<int>(padded_right - padded_left),
                    static_cast<int>(padded_bottom - padded_top)));
            }
        }

        const auto timestamp_ms = static_cast<std::int64_t>(
            std::llround(static_cast<double>(frame_meta.frame_index) * 1000.0 / frame_meta.fps));
        return enqueue(vp_objects::vp_ocr_job(
            frame_meta.channel_index,
            frame_meta.frame_index,
            target.track_id,
            timestamp_ms,
            frame_meta.fps,
            tight_crop,
            source_box,
            evidence_image_path,
            padded_crop));
    }

    bool vp_async_ocr_dispatcher::candidate_is_valid(
        const std::shared_ptr<worker_state>& state,
        const vp_objects::vp_ocr_candidate& candidate) {
        return !vp_objects::vp_ocr_consensus::normalize_numeric(candidate.text).empty() &&
            std::isfinite(candidate.confidence) &&
            candidate.confidence >= state->config.min_candidate_confidence &&
            candidate.confidence > 0.0f &&
            candidate.confidence <= 1.0f;
    }

    bool vp_async_ocr_dispatcher::is_worker_thread(const std::shared_ptr<worker_state>& state) {
        std::lock_guard<std::mutex> guard(state->lock);
        return state->worker_thread_id == std::this_thread::get_id();
    }

    void vp_async_ocr_dispatcher::request_shutdown(
        const std::shared_ptr<worker_state>& state,
        vp_ocr_shutdown_mode mode) {
        {
            std::lock_guard<std::mutex> guard(state->lock);
            state->statistics.accepting = false;
            state->stop_requested = true;
            state->cancel_pending = state->cancel_pending || mode == vp_ocr_shutdown_mode::CANCEL;
            if (state->cancel_pending) {
                state->statistics.canceled_jobs += state->queue.size();
                state->queue.clear();
                state->statistics.queue_size = 0;
            }
        }
        state->queue_condition.notify_all();
        state->idle_condition.notify_all();
    }

    void vp_async_ocr_dispatcher::ensure_worker_for_drain(
        const std::shared_ptr<worker_state>& state) {
        std::lock_guard<std::mutex> join_guard(state->join_lock);
        std::lock_guard<std::mutex> guard(state->lock);
        if (!state->statistics.worker_started &&
            !state->stop_requested &&
            !state->queue.empty()) {
            state->statistics.worker_started = true;
            state->statistics.worker_joined = false;
            worker_ = std::thread(&vp_async_ocr_dispatcher::worker_run, state);
        }
    }

    void vp_async_ocr_dispatcher::join_worker(const std::shared_ptr<worker_state>& state) {
        std::thread worker_to_join;
        {
            std::unique_lock<std::mutex> join_guard(state->join_lock);
            if (state->join_complete) {
                return;
            }
            if (state->join_in_progress) {
                state->join_condition.wait(join_guard, [&]() { return state->join_complete; });
                return;
            }

            if (worker_.joinable()) {
                state->join_in_progress = true;
                worker_to_join = std::move(worker_);
            }
            else {
                {
                    std::lock_guard<std::mutex> guard(state->lock);
                    state->statistics.worker_joined = true;
                }
                state->join_complete = true;
                join_guard.unlock();
                state->join_condition.notify_all();
                return;
            }
        }

        worker_to_join.join();

        {
            std::lock_guard<std::mutex> join_guard(state->join_lock);
            {
                std::lock_guard<std::mutex> guard(state->lock);
                state->statistics.worker_joined = true;
            }
            state->join_in_progress = false;
            state->join_complete = true;
        }
        state->join_condition.notify_all();
    }

    void vp_async_ocr_dispatcher::report_failure(
        const std::shared_ptr<worker_state>& state,
        const vp_objects::vp_ocr_job& job,
        const std::string& message) {
        {
            std::lock_guard<std::mutex> guard(state->lock);
            ++state->statistics.failed_jobs;
            state->statistics.last_error = message;
        }
        if (state->failure_callback_fn) {
            try {
                state->failure_callback_fn(job, message);
            }
            catch (...) {
            }
        }
    }

    void vp_async_ocr_dispatcher::process_job(
        const std::shared_ptr<worker_state>& state,
        const vp_objects::vp_ocr_job& job) {
        try {
            const auto tight_crop = job.crop();
            auto candidate = state->recognizer(tight_crop);
            bool used_padded_fallback = false;
            if (!candidate_is_valid(state, candidate)) {
                const auto padded_crop = job.padded_crop();
                if (!padded_crop.empty()) {
                    candidate = state->recognizer(padded_crop);
                    used_padded_fallback = true;
                }
            }

            if (!candidate_is_valid(state, candidate)) {
                std::lock_guard<std::mutex> guard(state->lock);
                ++state->statistics.rejected_results;
                return;
            }

            candidate.text = vp_objects::vp_ocr_consensus::normalize_numeric(candidate.text);
            if (state->result_callback_fn) {
                state->result_callback_fn(vp_async_ocr_result {job, candidate, used_padded_fallback});
            }
        }
        catch (const std::exception& error) {
            report_failure(state, job, error.what());
        }
        catch (...) {
            report_failure(state, job, "unknown OCR exception");
        }
    }

    void vp_async_ocr_dispatcher::worker_run(std::shared_ptr<worker_state> state) {
        {
            std::lock_guard<std::mutex> guard(state->lock);
            state->worker_thread_id = std::this_thread::get_id();
        }
        while (true) {
            std::optional<vp_objects::vp_ocr_job> job;
            {
                std::unique_lock<std::mutex> lock(state->lock);
                state->queue_condition.wait(lock, [&]() {
                    return state->stop_requested || !state->queue.empty();
                });
                if (state->stop_requested && (state->cancel_pending || state->queue.empty())) {
                    break;
                }
                job.emplace(state->queue.front());
                state->queue.pop_front();
                state->statistics.queue_size = state->queue.size();
                ++state->statistics.active_jobs;
            }

            process_job(state, *job);
            {
                std::lock_guard<std::mutex> guard(state->lock);
                --state->statistics.active_jobs;
                ++state->statistics.processed_jobs;
                if (state->queue.empty() && state->statistics.active_jobs == 0) {
                    state->idle_condition.notify_all();
                }
            }
        }

        std::lock_guard<std::mutex> guard(state->lock);
        if (state->queue.empty() && state->statistics.active_jobs == 0) {
            state->idle_condition.notify_all();
        }
    }

    bool vp_async_ocr_dispatcher::wait_until_idle(std::chrono::milliseconds timeout) {
        const auto state = state_;
        std::unique_lock<std::mutex> lock(state->lock);
        return state->idle_condition.wait_for(lock, timeout, [&]() {
            return state->queue.empty() && state->statistics.active_jobs == 0;
        });
    }

    void vp_async_ocr_dispatcher::shutdown(vp_ocr_shutdown_mode mode) {
        const auto state = state_;
        const auto called_from_worker = is_worker_thread(state);
        if (!called_from_worker && mode == vp_ocr_shutdown_mode::DRAIN) {
            ensure_worker_for_drain(state);
        }
        request_shutdown(state, mode);
        if (called_from_worker) {
            return;
        }
        join_worker(state);
    }

    vp_async_ocr_statistics vp_async_ocr_dispatcher::statistics() const {
        const auto state = state_;
        std::lock_guard<std::mutex> guard(state->lock);
        auto snapshot = state->statistics;
        snapshot.queue_size = state->queue.size();
        return snapshot;
    }

#ifndef VP_ASYNC_OCR_CORE_ONLY
    vp_async_ocr_node::vp_async_ocr_node(
        std::string node_name,
        vp_async_ocr_config config,
        vp_async_ocr_dispatcher::recognizer_callback recognizer,
        vp_async_ocr_dispatcher::result_callback result_callback,
        vp_async_ocr_dispatcher::failure_callback failure_callback,
        evidence_path_callback evidence_path_callback):
        vp_node(std::move(node_name)),
        dispatcher_(
            std::move(config),
            std::move(recognizer),
            [this, result_callback = std::move(result_callback)](const vp_async_ocr_result& result) {
                consensus_.observe(
                    result.job.channel_index(),
                    result.job.track_id(),
                    result.job.frame_index(),
                    result.candidate);
                if (result_callback) {
                    result_callback(result);
                }
            },
            std::move(failure_callback)),
        evidence_path_callback_(std::move(evidence_path_callback)) {
        initialized();
    }

    vp_async_ocr_node::~vp_async_ocr_node() {
        deinitialized();
        dispatcher_.shutdown(vp_ocr_shutdown_mode::DRAIN);
        consensus_.reset();
    }

    std::shared_ptr<vp_objects::vp_meta> vp_async_ocr_node::handle_frame_meta(
        std::shared_ptr<vp_objects::vp_frame_meta> meta) {
        for (const auto& target : meta->targets) {
            if (!target) {
                continue;
            }
            const auto evidence_path = evidence_path_callback_
                ? evidence_path_callback_(*meta, *target)
                : std::string {};
            dispatcher_.submit(*meta, *target, evidence_path);
        }
        return vp_node::handle_frame_meta(meta);
    }

    vp_async_ocr_statistics vp_async_ocr_node::statistics() const {
        return dispatcher_.statistics();
    }

    vp_objects::vp_ocr_consensus_result vp_async_ocr_node::consensus_snapshot(
        int channel_index,
        int track_id) const {
        return consensus_.snapshot(channel_index, track_id);
    }

    void vp_async_ocr_node::clear_consensus(int channel_index, int track_id) {
        consensus_.clear(channel_index, track_id);
    }

    void vp_async_ocr_node::clear_consensus_channel(int channel_index) {
        consensus_.clear_channel(channel_index);
    }

    void vp_async_ocr_node::reset_consensus() {
        consensus_.reset();
    }

    void vp_async_ocr_node::shutdown(vp_ocr_shutdown_mode mode) {
        dispatcher_.shutdown(mode);
    }
#endif
}

#endif
