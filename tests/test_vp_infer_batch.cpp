#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>

#include "nodes/vp_infer_node.h"

namespace {
    class identity_infer_node : public vp_nodes::vp_infer_node {
    public:
        explicit identity_infer_node(int batch_size):
            vp_infer_node(
                "identity_infer_node",
                vp_nodes::vp_infer_type::SECONDARY,
                "",
                "",
                "",
                1,
                1,
                batch_size,
                1.0,
                cv::Scalar(),
                cv::Scalar(1),
                false,
                false) {
            cv::dnn::LayerParams params;
            const auto identity_layer = net.addLayer("identity", "Identity", params);
            net.connect(0, 0, identity_layer, 0);
        }

        std::vector<cv::Mat> run(const cv::Mat& input) {
            std::vector<cv::Mat> outputs;
            infer(input, outputs);
            return outputs;
        }

    protected:
        void prepare(
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>&,
            std::vector<cv::Mat>&) override {
        }

        void postprocess(
            const std::vector<cv::Mat>&,
            const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>&) override {
        }
    };

    void require(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    cv::Mat make_input(int batch_count) {
        int sizes[] = {batch_count, 1, 1, 1};
        cv::Mat input(4, sizes, CV_32F);
        for (int i = 0; i < batch_count; ++i) {
            input.ptr<float>(i)[0] = static_cast<float>(i + 1);
        }
        return input;
    }

    void require_identity_output(const cv::Mat& output, int batch_count) {
        require(output.dims == 4, "output dimensions changed");
        require(output.size[0] == batch_count, "output batch size changed");
        require(output.total() == static_cast<std::size_t>(batch_count), "output element count changed");
        for (int i = 0; i < batch_count; ++i) {
            require(
                std::fabs(output.ptr<float>(i)[0] - static_cast<float>(i + 1)) < 0.0001f,
                "output batch values were merged at the wrong offset");
        }
    }

    void test_partial_final_batch() {
        identity_infer_node node(4);
        const auto outputs = node.run(make_input(5));
        require(outputs.size() == 1, "identity network returned an unexpected output count");
        require_identity_output(outputs[0], 5);
    }

    void test_exact_batches() {
        identity_infer_node node(4);
        const auto outputs = node.run(make_input(8));
        require(outputs.size() == 1, "identity network returned an unexpected output count");
        require_identity_output(outputs[0], 8);
    }
}

int main() {
    auto& logger = vp_utils::vp_logger::get_logger();
    logger.log_level = vp_utils::vp_log_level::ERROR;
    logger.log_to_console = false;
    logger.log_to_file = false;
    logger.log_to_kafka = false;
    logger.init();

    try {
        test_partial_final_batch();
        test_exact_batches();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    return 0;
}
