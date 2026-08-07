#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "nodes/vp_node.h"
#include "nodes/vp_split_node.h"
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

    class manual_publisher_node : public vp_nodes::vp_node {
    public:
        explicit manual_publisher_node(std::string name): vp_node(std::move(name)) {
        }

        void publish(std::shared_ptr<vp_objects::vp_meta> meta) {
            push_meta(std::move(meta));
        }
    };

    class passive_node : public vp_nodes::vp_node {
    public:
        explicit passive_node(std::string name): vp_node(std::move(name)) {
        }
    };

    class manual_split_node : public vp_nodes::vp_split_node {
    public:
        explicit manual_split_node(std::string name, bool split_with_channel_index = false):
            vp_split_node(std::move(name), split_with_channel_index) {
        }

        void publish(std::shared_ptr<vp_objects::vp_meta> meta) {
            push_meta(std::move(meta));
        }
    };

    class lifecycle_node : public vp_nodes::vp_node {
    public:
        lifecycle_node(std::string name, std::atomic<int>& destroyed):
            vp_node(std::move(name)), destroyed(destroyed) {
            initialized();
        }

        ~lifecycle_node() {
            deinitialized();
            destroyed.fetch_add(1);
        }

    private:
        std::atomic<int>& destroyed;
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

    void test_arriving_hook_can_reenter_meta_flow() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(1, outputs, batches);
        bool reentered = false;
        node.set_meta_arriving_hooker(
            [&node, &reentered](
                std::string,
                int,
                std::shared_ptr<vp_objects::vp_meta>) {
                if (!reentered) {
                    reentered = true;
                    node.meta_flow(make_frame(2));
                }
            });

        node.meta_flow(make_frame(1));

        require(reentered, "arriving hook did not reenter meta_flow");
        require(outputs.wait_for_count(2), "reentrant arriving hook stalled the input queue");
        require(
            outputs.snapshot() == std::vector<int>({1, 2}),
            "reentrant arriving hook changed input order");
    }

    void test_arriving_hook_can_replace_itself() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(1, outputs, batches);
        node.set_meta_arriving_hooker(
            [&node](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                node.set_meta_arriving_hooker({});
            });

        node.meta_flow(make_frame(1));

        require(outputs.wait_for_count(1), "self-replacing arriving hook stalled the input queue");
        require(outputs.snapshot() == std::vector<int>({1}), "arriving hook changed frame output");
    }

    void test_throwing_arriving_hook_does_not_stall_input() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(1, outputs, batches);
        node.set_meta_arriving_hooker(
            [](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                throw std::runtime_error("arriving hook failure");
            });

        bool hook_error_thrown = false;
        try {
            node.meta_flow(make_frame(1));
        }
        catch (const std::runtime_error&) {
            hook_error_thrown = true;
        }
        require(hook_error_thrown, "arriving hook error was not propagated");

        node.set_meta_arriving_hooker({});
        node.meta_flow(make_frame(2));

        require(outputs.wait_for_count(2), "arriving hook error stalled the input queue");
        require(
            outputs.snapshot() == std::vector<int>({1, 2}),
            "arriving hook error changed input order");
    }

    void test_subscriber_can_detach_while_receiving_meta() {
        auto upstream = std::make_shared<manual_publisher_node>("upstream");
        auto downstream = std::make_shared<passive_node>("downstream");
        downstream->attach_to({upstream});
        std::weak_ptr<passive_node> downstream_weak = downstream;
        downstream->set_meta_arriving_hooker(
            [downstream_weak](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                if (auto downstream = downstream_weak.lock()) {
                    downstream->detach();
                }
            });

        upstream->publish(make_frame(1));

        require(upstream->next_nodes().empty(), "subscriber did not detach during publication");
    }

    void test_split_subscriber_can_detach_while_receiving_meta() {
        auto upstream = std::make_shared<manual_split_node>("split");
        auto downstream = std::make_shared<passive_node>("downstream");
        downstream->attach_to({upstream});
        std::weak_ptr<passive_node> downstream_weak = downstream;
        downstream->set_meta_arriving_hooker(
            [downstream_weak](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                if (auto downstream = downstream_weak.lock()) {
                    downstream->detach();
                }
            });

        upstream->publish(make_frame(1));

        require(upstream->next_nodes().empty(), "split subscriber did not detach during publication");
    }

    void test_channel_split_subscriber_can_detach_while_receiving_meta() {
        auto upstream = std::make_shared<manual_split_node>("split", true);
        auto downstream = std::make_shared<passive_node>("downstream");
        downstream->attach_to({upstream});
        std::weak_ptr<passive_node> downstream_weak = downstream;
        downstream->set_meta_arriving_hooker(
            [downstream_weak](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                if (auto downstream = downstream_weak.lock()) {
                    downstream->detach();
                }
            });

        upstream->publish(make_frame(1));

        require(upstream->next_nodes().empty(), "channel split subscriber did not detach during publication");
    }

    void test_throwing_handling_hook_does_not_stop_node() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(1, outputs, batches);
        node.set_meta_handling_hooker(
            [](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                throw std::runtime_error("handling hook failure");
            });

        node.meta_flow(make_frame(1));

        require(outputs.wait_for_count(1), "handling hook error stopped the node");
        require(outputs.snapshot() == std::vector<int>({1}), "handling hook error dropped the frame");
    }

    void test_throwing_leaving_hook_does_not_stop_node() {
        output_collector outputs;
        batch_collector batches;
        recording_batch_node node(1, outputs, batches);
        node.set_meta_leaving_hooker(
            [](std::string, int, std::shared_ptr<vp_objects::vp_meta>) {
                throw std::runtime_error("leaving hook failure");
            });

        node.meta_flow(make_frame(1));

        require(outputs.wait_for_count(1), "leaving hook error stopped the node");
        require(outputs.snapshot() == std::vector<int>({1}), "leaving hook error dropped the frame");
    }

    void test_attached_nodes_are_released_without_manual_detach() {
        std::atomic<int> destroyed {0};
        std::weak_ptr<lifecycle_node> upstream_weak;
        std::weak_ptr<lifecycle_node> downstream_weak;
        {
            auto upstream = std::make_shared<lifecycle_node>("upstream", destroyed);
            auto downstream = std::make_shared<lifecycle_node>("downstream", destroyed);
            downstream->attach_to({upstream});
            upstream_weak = upstream;
            downstream_weak = downstream;
        }

        require(upstream_weak.expired(), "downstream retained its upstream node");
        require(downstream_weak.expired(), "upstream did not release its downstream node");
        require(destroyed.load() == 2, "attached node destructors did not run");
    }

    void test_multi_upstream_nodes_are_released_without_manual_detach() {
        std::atomic<int> destroyed {0};
        std::weak_ptr<lifecycle_node> upstream_a_weak;
        std::weak_ptr<lifecycle_node> upstream_b_weak;
        std::weak_ptr<lifecycle_node> downstream_weak;
        {
            auto upstream_a = std::make_shared<lifecycle_node>("upstream_a", destroyed);
            auto upstream_b = std::make_shared<lifecycle_node>("upstream_b", destroyed);
            auto downstream = std::make_shared<lifecycle_node>("downstream", destroyed);
            downstream->attach_to({upstream_a, upstream_b});
            upstream_a_weak = upstream_a;
            upstream_b_weak = upstream_b;
            downstream_weak = downstream;
        }

        require(upstream_a_weak.expired(), "downstream retained its first upstream node");
        require(upstream_b_weak.expired(), "downstream retained its second upstream node");
        require(downstream_weak.expired(), "upstream nodes did not release their shared downstream node");
        require(destroyed.load() == 3, "multi-upstream node destructors did not run");
    }

    void test_detach_ignores_an_expired_upstream_node() {
        std::atomic<int> destroyed {0};
        auto downstream = std::make_shared<lifecycle_node>("downstream", destroyed);
        std::weak_ptr<lifecycle_node> upstream_weak;
        {
            auto upstream = std::make_shared<lifecycle_node>("upstream", destroyed);
            downstream->attach_to({upstream});
            upstream_weak = upstream;
        }

        require(upstream_weak.expired(), "downstream retained an otherwise unused upstream node");
        downstream->detach();
        downstream.reset();
        require(destroyed.load() == 2, "expired upstream prevented node destruction");
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

int main(int argc, char** argv) {
    auto& logger = vp_utils::vp_logger::get_logger();
    logger.log_level = vp_utils::vp_log_level::ERROR;
    logger.log_to_console = false;
    logger.log_to_file = false;
    logger.log_to_kafka = false;
    logger.init();

    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"control_flush", test_control_meta_flushes_partial_batch_first},
        {"complete_batch", test_complete_batch_is_dispatched},
        {"shutdown_flush", test_shutdown_flushes_partial_batch},
        {"source_control_order", test_source_controls_follow_queued_frames},
        {"handled_hook_reentry", test_handled_hook_can_enqueue_source_control},
        {"handled_hook_exception", test_throwing_handled_hook_does_not_stall_output},
        {"arriving_hook_reentry", test_arriving_hook_can_reenter_meta_flow},
        {"arriving_hook_replacement", test_arriving_hook_can_replace_itself},
        {"arriving_hook_exception", test_throwing_arriving_hook_does_not_stall_input},
        {"subscriber_detach", test_subscriber_can_detach_while_receiving_meta},
        {"split_subscriber_detach", test_split_subscriber_can_detach_while_receiving_meta},
        {"channel_split_subscriber_detach", test_channel_split_subscriber_can_detach_while_receiving_meta},
        {"handling_hook_exception", test_throwing_handling_hook_does_not_stop_node},
        {"leaving_hook_exception", test_throwing_leaving_hook_does_not_stop_node},
        {"automatic_node_release", test_attached_nodes_are_released_without_manual_detach},
        {"multi_upstream_release", test_multi_upstream_nodes_are_released_without_manual_detach},
        {"expired_upstream_detach", test_detach_ignores_an_expired_upstream_node},
    };

    if (argc == 2) {
        auto test = std::find_if(tests.begin(), tests.end(), [argv](const auto& candidate) {
            return candidate.first == argv[1];
        });
        if (test == tests.end()) {
            std::cerr << "unknown test: " << argv[1] << std::endl;
            return 2;
        }
        return run_test(test->first, test->second);
    }

    int failures = 0;
    for (const auto& test: tests) {
        failures += run_test(test.first, test.second);
    }
    return failures == 0 ? 0 : 1;
}
