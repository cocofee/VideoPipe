#pragma once

#if defined(VP_WITH_ASYNC_OCR) || defined(VP_ASYNC_OCR_CORE_ONLY)

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "../../objects/vp_frame_meta.h"
#include "../../objects/vp_frame_target.h"
#include "../../objects/vp_ocr_consensus.h"
#include "../../objects/vp_ocr_job.h"

#ifndef VP_ASYNC_OCR_CORE_ONLY
#include "../vp_node.h"
#endif

namespace vp_nodes {
    struct vp_async_ocr_config {
        std::size_t capacity = 32;
        int min_crop_width = 8;
        int min_crop_height = 8;
        int max_crop_width = 2048;
        int max_crop_height = 2048;
        int fallback_padding = 12;
        float min_candidate_confidence = 0.0f;
    };

    enum class vp_ocr_shutdown_mode {
        DRAIN,
        CANCEL
    };

    struct vp_async_ocr_statistics {
        std::size_t queue_size = 0;
        std::size_t active_jobs = 0;
        std::uint64_t enqueued_jobs = 0;
        std::uint64_t processed_jobs = 0;
        std::uint64_t dropped_jobs = 0;
        std::uint64_t canceled_jobs = 0;
        std::uint64_t failed_jobs = 0;
        std::uint64_t rejected_results = 0;
        bool accepting = false;
        bool worker_started = false;
        bool worker_joined = false;
        std::string last_error;
    };

    struct vp_async_ocr_result {
        vp_objects::vp_ocr_job job;
        vp_objects::vp_ocr_candidate candidate;
        bool used_padded_fallback = false;
    };

    class vp_async_ocr_dispatcher {
    public:
        using recognizer_callback = std::function<vp_objects::vp_ocr_candidate(const cv::Mat&)>;
        using result_callback = std::function<void(const vp_async_ocr_result&)>;
        using failure_callback = std::function<void(const vp_objects::vp_ocr_job&, const std::string&)>;

    private:
        struct worker_state;
        std::shared_ptr<worker_state> state_;
        std::thread worker_;

        static void validate_config(const vp_async_ocr_config& config);
        static bool candidate_is_valid(
            const std::shared_ptr<worker_state>& state,
            const vp_objects::vp_ocr_candidate& candidate);
        static bool is_worker_thread(const std::shared_ptr<worker_state>& state);
        static void request_shutdown(
            const std::shared_ptr<worker_state>& state,
            vp_ocr_shutdown_mode mode);
        void ensure_worker_for_drain(const std::shared_ptr<worker_state>& state);
        void join_worker(const std::shared_ptr<worker_state>& state);
        static void worker_run(std::shared_ptr<worker_state> state);
        static void process_job(
            const std::shared_ptr<worker_state>& state,
            const vp_objects::vp_ocr_job& job);
        static void report_failure(
            const std::shared_ptr<worker_state>& state,
            const vp_objects::vp_ocr_job& job,
            const std::string& message);

    public:
        vp_async_ocr_dispatcher(
            vp_async_ocr_config config,
            recognizer_callback recognizer,
            result_callback result_callback = {},
            failure_callback failure_callback = {},
            bool start_immediately = true);
        ~vp_async_ocr_dispatcher();

        vp_async_ocr_dispatcher(const vp_async_ocr_dispatcher&) = delete;
        vp_async_ocr_dispatcher& operator=(const vp_async_ocr_dispatcher&) = delete;

        void start();
        bool enqueue(vp_objects::vp_ocr_job job);
        bool submit(
            const vp_objects::vp_frame_meta& frame_meta,
            const vp_objects::vp_frame_target& target,
            const std::string& evidence_image_path);
        bool wait_until_idle(std::chrono::milliseconds timeout);
        void shutdown(vp_ocr_shutdown_mode mode);
        vp_async_ocr_statistics statistics() const;
    };

#ifndef VP_ASYNC_OCR_CORE_ONLY
    class vp_async_ocr_node: public vp_node {
    public:
        using evidence_path_callback = std::function<std::string(
            const vp_objects::vp_frame_meta&,
            const vp_objects::vp_frame_target&)>;

    private:
        vp_objects::vp_ocr_consensus consensus_;
        vp_async_ocr_dispatcher dispatcher_;
        const evidence_path_callback evidence_path_callback_;

    protected:
        virtual std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(
            std::shared_ptr<vp_objects::vp_frame_meta> meta) override;

    public:
        vp_async_ocr_node(
            std::string node_name,
            vp_async_ocr_config config,
            vp_async_ocr_dispatcher::recognizer_callback recognizer,
            vp_async_ocr_dispatcher::result_callback result_callback = {},
            vp_async_ocr_dispatcher::failure_callback failure_callback = {},
            evidence_path_callback evidence_path_callback = {});
        ~vp_async_ocr_node();

        vp_async_ocr_statistics statistics() const;
        vp_objects::vp_ocr_consensus_result consensus_snapshot(int channel_index, int track_id) const;
        void clear_consensus(int channel_index, int track_id);
        void clear_consensus_channel(int channel_index);
        void reset_consensus();
        void shutdown(vp_ocr_shutdown_mode mode = vp_ocr_shutdown_mode::DRAIN);
    };
#endif
}

#endif
