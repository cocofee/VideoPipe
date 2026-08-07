#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "nodes/vp_node.h"
#include "nodes/vp_src_node.h"

namespace {
    class output_collector {
    public:
        void add(std::shared_ptr<vp_objects::vp_meta> meta) {
            std::lock_guard<std::mutex> guard(lock);
            if (meta->meta_type == vp_objects::vp_meta_type::FRAME) {
                values.push_back(std::dynamic_pointer_cast<vp_objects::vp_frame_meta>(meta)->frame_index);
            }
            else {
                values.push_back(-1);
            }
            changed.notify_all();
        }

        bool wait_for_count(std::size_t count) {
            std::unique_lock<std::mutex> guard(lock);
            return changed.wait_for(guard, std::chrono::seconds(1), [this, count] {
                return values.size() >= count;
            });
        }

        std::vector<int> snapshot() {
            std::lock_guard<std::mutex> guard(lock);
            return values;
        }

    private:
        std::mutex lock;
        std::condition_variable changed;
        std::vector<int> values;
    };

    class batch_collector {
    public:
        void add(std::size_t size) {
            std::lock_guard<std::mutex> guard(lock);
            batch_sizes.push_back(size);
        }

        std::vector<std::size_t> snapshot() {
            std::lock_guard<std::mutex> guard(lock);
            return batch_sizes;
        }

    private:
        std::mutex lock;
        std::vector<std::size_t> batch_sizes;
    };

    class recording_batch_node : public vp_nodes::vp_node {
    public:
        recording_batch_node(
            int batch_size,
            output_collector& outputs,
            batch_collector& batches):
            vp_node("recording_batch_node"),
            outputs(outputs),
            batches(batches) {
            frame_meta_handle_batch = batch_size;
            initialized();
        }

        ~recording_batch_node() {
            deinitialized();
        }

    protected:
        void handle_frame_meta(
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& metas) override {
            batches.add(metas.size());
        }

        void push_meta(std::shared_ptr<vp_objects::vp_meta> meta) override {
            outputs.add(meta);
        }

    private:
        output_collector& outputs;
        batch_collector& batches;
    };

    class manual_src_node : public vp_nodes::vp_src_node {
    public:
        explicit manual_src_node(output_collector& outputs):
            vp_src_node("manual_src_node", 0),
            outputs(outputs) {
        }

        void queue_frame(std::shared_ptr<vp_objects::vp_frame_meta> meta) {
            pendding_meta(meta);
        }

        void drain() {
            pendding_meta(nullptr);
            dispatch_run();
        }

    protected:
        void handle_run() override {
        }

        void push_meta(std::shared_ptr<vp_objects::vp_meta> meta) override {
            outputs.add(meta);
        }

    private:
        output_collector& outputs;
    };

    void require(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    std::shared_ptr<vp_objects::vp_frame_meta> make_frame(int frame_index) {
        return std::make_shared<vp_objects::vp_frame_meta>(
            cv::Mat::zeros(1, 1, CV_8UC3), frame_index, 0, 1, 1, 25);
    }

    void test_control_meta_flushes_partial_batch_first() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(3, outputs, batches);

        node.meta_flow(make_frame(1));
        node.meta_flow(make_frame(2));
        node.meta_flow(std::make_shared<vp_objects::vp_control_meta>(
            vp_objects::vp_control_type::SPEAK, 0, "control"));

        require(outputs.wait_for_count(3), "control meta did not flush the partial batch");
        require(outputs.snapshot() == std::vector<int>({1, 2, -1}), "control meta overtook cached frames");
        require(batches.snapshot() == std::vector<std::size_t>({2}), "partial batch size was not preserved");
    }

    void test_complete_batch_is_dispatched() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(3, outputs, batches);

        node.meta_flow(make_frame(1));
        node.meta_flow(make_frame(2));
        node.meta_flow(make_frame(3));

        require(outputs.wait_for_count(3), "complete batch was not dispatched");
        require(outputs.snapshot() == std::vector<int>({1, 2, 3}), "complete batch order changed");
        require(batches.snapshot() == std::vector<std::size_t>({3}), "complete batch size changed");
    }

    void test_shutdown_flushes_partial_batch() {
        output_collector outputs;
        batch_collector batches;
        {
            recording_batch_node node(3, outputs, batches);
            node.meta_flow(make_frame(1));
        }

        require(outputs.snapshot() == std::vector<int>({1}), "shutdown dropped the partial batch");
        require(batches.snapshot() == std::vector<std::size_t>({1}), "shutdown used the wrong partial batch size");
    }

    void test_source_controls_follow_queued_frames() {
        output_collector outputs;
        manual_src_node node(outputs);

        node.queue_frame(make_frame(1));
        node.speak();
        node.record_image_manually();
        node.record_video_manually(false, 5);
        node.drain();

        require(
            outputs.snapshot() == std::vector<int>({1, -1, -1, -1}),
            "source control meta overtook a queued frame");
    }

    void test_handled_hook_can_enqueue_source_control() {
        output_collector outputs;
        manual_src_node node(outputs);
        bool control_queued = false;
        node.set_meta_handled_hooker(
            [&node, &control_queued](
                std::string,
                int,
                std::shared_ptr<vp_objects::vp_meta> meta) {
                if (!control_queued && meta->meta_type == vp_objects::vp_meta_type::FRAME) {
                    control_queued = true;
                    node.record_image_manually();
                }
            });

        node.queue_frame(make_frame(1));
        node.drain();

        require(control_queued, "handled hook did not enqueue the source control");
        require(
            outputs.snapshot() == std::vector<int>({1, -1}),
            "source control enqueued by handled hook changed output order");
    }

    void test_throwing_handled_hook_does_not_stall_output() {
        output_collector outputs;
        manual_src_node node(outputs);
        node.set_meta_handled_hooker(
            [](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                throw std::runtime_error("handled hook failure");
            });

        bool hook_error_thrown = false;
        try {
            node.queue_frame(make_frame(1));
        }
        catch (const std::runtime_error&) {
            hook_error_thrown = true;
        }
        require(hook_error_thrown, "handled hook error was not propagated");

        node.set_meta_handled_hooker({});
        node.speak();
        node.drain();

        require(
            outputs.snapshot() == std::vector<int>({1, -1}),
            "handled hook error stalled the output queue");
    }

    template <typename Test>
    int run_test(const std::string& name, Test test) {
        try {
            test();
            return 0;
        }
        catch (const std::exception& error) {
            std::cerr << name << ": " << error.what() << std::endl;
            return 1;
        }
    }
}

int main() {
    auto& logger = vp_utils::vp_logger::get_logger();
    logger.log_level = vp_utils::vp_log_level::ERROR;
    logger.log_to_console = false;
    logger.log_to_file = false;
    logger.log_to_kafka = false;
    logger.init();

    int failures = 0;
    failures += run_test("control flush", test_control_meta_flushes_partial_batch_first);
    failures += run_test("complete batch", test_complete_batch_is_dispatched);
    failures += run_test("shutdown flush", test_shutdown_flushes_partial_batch);
    failures += run_test("source control order", test_source_controls_follow_queued_frames);
    failures += run_test("handled hook reentry", test_handled_hook_can_enqueue_source_control);
    failures += run_test("handled hook exception", test_throwing_handled_hook_does_not_stall_output);
    return failures == 0 ? 0 : 1;
}
