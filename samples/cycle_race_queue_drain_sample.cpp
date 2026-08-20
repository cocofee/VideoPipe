#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "../nodes/vp_app_src_node.h"
#include "../nodes/vp_des_node.h"

namespace {

    class test_app_src_node: public vp_nodes::vp_app_src_node {
    public:
        using vp_nodes::vp_app_src_node::vp_app_src_node;

        void shutdown() {
            deinitialized();
        }
    };

    class counting_des_node: public vp_nodes::vp_des_node {
    private:
        std::atomic<int> handled_frames = 0;

    protected:
        std::shared_ptr<vp_objects::vp_meta> handle_frame_meta(
            std::shared_ptr<vp_objects::vp_frame_meta> meta) override {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            ++handled_frames;
            return vp_nodes::vp_des_node::handle_frame_meta(std::move(meta));
        }

    public:
        counting_des_node(std::string node_name, int channel_index):
            vp_nodes::vp_des_node(std::move(node_name), channel_index) {
            initialized();
        }

        ~counting_des_node() {
            deinitialized();
        }

        void shutdown() {
            deinitialized();
        }

        int count() const {
            return handled_frames.load();
        }
    };

    class batching_mid_node: public vp_nodes::vp_node {
    private:
        std::atomic<int> handled_frames = 0;

    protected:
        void handle_frame_meta(
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& metas) override {
            handled_frames += static_cast<int>(metas.size());
        }

    public:
        explicit batching_mid_node(std::string node_name):
            vp_nodes::vp_node(std::move(node_name)) {
            frame_meta_handle_batch = 4;
            initialized();
        }

        ~batching_mid_node() {
            deinitialized();
        }

        void shutdown() {
            deinitialized();
        }

        int count() const {
            return handled_frames.load();
        }
    };

}

int main() {
    const auto log_dir = std::filesystem::temp_directory_path() /
        "cycle_race_queue_drain_logs";
    VP_SET_LOG_DIR(log_dir.string());
    VP_SET_LOG_TO_FILE(false);
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::ERROR);
    try {
        VP_LOGGER_INIT();
    }
    catch (const char* message) {
        std::cerr << "queue sample: logger init failed: " << message << std::endl;
        return 1;
    }
    catch (const std::exception& error) {
        std::cerr << "queue sample: logger init failed: " << error.what() << std::endl;
        return 1;
    }

    constexpr int frame_count = 64;
    auto source = std::make_shared<test_app_src_node>("queue_test_source", 0);
    auto destination = std::make_shared<counting_des_node>("queue_test_destination", 0);
    destination->attach_to({source});
    source->start();

    std::vector<cv::Mat> frames;
    frames.reserve(frame_count);
    for (auto i = 0; i < frame_count; ++i) {
        frames.emplace_back(8, 8, CV_8UC3, cv::Scalar(i, i, i));
    }
    if (!source->push_frames(std::move(frames))) {
        std::cerr << "failed to enqueue queue-drain frames" << std::endl;
        return 1;
    }

    source->shutdown();
    destination->shutdown();
    if (destination->count() != frame_count) {
        std::cerr << "queue drain lost frames: expected=" << frame_count
                  << " actual=" << destination->count() << std::endl;
        return 1;
    }

    destination->detach();

    auto batch_source = std::make_shared<test_app_src_node>("batch_queue_test_source", 1);
    auto batch_node = std::make_shared<batching_mid_node>("batch_queue_test_middle");
    auto batch_destination = std::make_shared<counting_des_node>("batch_queue_test_destination", 1);
    batch_node->attach_to({batch_source});
    batch_destination->attach_to({batch_node});
    batch_source->start();

    std::vector<cv::Mat> partial_batch_frames;
    partial_batch_frames.reserve(10);
    for (auto i = 0; i < 10; ++i) {
        partial_batch_frames.emplace_back(8, 8, CV_8UC3, cv::Scalar(i, i, i));
    }
    if (!batch_source->push_frames(std::move(partial_batch_frames))) {
        std::cerr << "failed to enqueue partial-batch frames" << std::endl;
        return 1;
    }
    batch_source->shutdown();
    batch_node->shutdown();
    batch_destination->shutdown();
    if (batch_node->count() != 10 || batch_destination->count() != 10) {
        std::cerr << "partial batch drain lost frames: handled=" << batch_node->count()
                  << " delivered=" << batch_destination->count() << std::endl;
        return 1;
    }
    batch_destination->detach();

    std::cout << "cycle race queue drain sample passed" << std::endl;
    return 0;
}
