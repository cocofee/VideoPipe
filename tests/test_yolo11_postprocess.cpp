#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../nodes/infers/vp_yolo11_detector_node.h"
#include "../objects/vp_frame_target.h"

namespace {

using vp_nodes::vp_yolo11_detection;
using vp_nodes::vp_yolo11_letterbox;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, const char* message) {
    if (std::fabs(actual - expected) > 0.0001f) {
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

const vp_yolo11_detection& detection_for_class(
    const std::vector<vp_yolo11_detection>& detections,
    int class_id) {
    for (const auto& detection : detections) {
        if (detection.class_id == class_id) {
            return detection;
        }
    }
    throw std::runtime_error("expected class was not decoded");
}

cv::Mat make_channels_first_output() {
    const int sizes[] = {1, 6, 3};
    cv::Mat output(3, sizes, CV_32F, cv::Scalar(0));
    const float candidates[][6] = {
        {320.0f, 320.0f, 100.0f, 100.0f, 0.90f, 0.10f},
        {322.0f, 322.0f, 100.0f, 100.0f, 0.70f, 0.20f},
        {320.0f, 320.0f, 100.0f, 100.0f, 0.05f, 0.85f},
    };

    for (int candidate = 0; candidate < 3; ++candidate) {
        for (int channel = 0; channel < 6; ++channel) {
            int index[] = {0, channel, candidate};
            output.at<float>(index) = candidates[candidate][channel];
        }
    }
    return output;
}

cv::Mat make_candidates_first_output() {
    const int sizes[] = {1, 3, 6};
    cv::Mat output(3, sizes, CV_32F, cv::Scalar(0));
    const float candidates[][6] = {
        {320.0f, 320.0f, 100.0f, 100.0f, 0.90f, 0.10f},
        {322.0f, 322.0f, 100.0f, 100.0f, 0.70f, 0.20f},
        {320.0f, 320.0f, 100.0f, 100.0f, 0.05f, 0.85f},
    };

    for (int candidate = 0; candidate < 3; ++candidate) {
        for (int channel = 0; channel < 6; ++channel) {
            int index[] = {0, candidate, channel};
            output.at<float>(index) = candidates[candidate][channel];
        }
    }
    return output;
}

void verify_layout(const cv::Mat& output) {
    const vp_yolo11_letterbox letterbox {
        0.5f,
        0,
        140,
        1280,
        720,
    };

    const auto detections = vp_nodes::decode_yolo11_output(
        output,
        letterbox,
        0.25f,
        0.45f,
        2);

    require(detections.size() == 2, "class-aware NMS must suppress only the lower score in the same class");

    const auto& class_zero = detection_for_class(detections, 0);
    require(class_zero.box == cv::Rect(540, 260, 200, 200), "letterbox coordinates were not restored");
    require_close(class_zero.confidence, 0.90f, "highest class-zero confidence was not preserved");

    const auto& class_one = detection_for_class(detections, 1);
    require(class_one.box == cv::Rect(540, 260, 200, 200), "class-one box was not decoded");
    require_close(class_one.confidence, 0.85f, "class-one confidence was not decoded");
}

void verify_clamp_and_zero_area_rejection() {
    const int sizes[] = {1, 2, 5};
    cv::Mat output(3, sizes, CV_32F, cv::Scalar(0));
    const float candidates[][5] = {
        {10.0f, 320.0f, 40.0f, 100.0f, 0.90f},
        {320.0f, 50.0f, 20.0f, 20.0f, 0.80f},
    };
    for (int candidate = 0; candidate < 2; ++candidate) {
        for (int channel = 0; channel < 5; ++channel) {
            int index[] = {0, candidate, channel};
            output.at<float>(index) = candidates[candidate][channel];
        }
    }

    const vp_yolo11_letterbox letterbox {
        0.5f,
        0,
        140,
        1280,
        720,
    };
    const auto detections = vp_nodes::decode_yolo11_output(
        output,
        letterbox,
        0.25f,
        0.45f,
        1);

    require(detections.size() == 1, "boxes outside the original frame must be rejected after clamping");
    require(detections[0].box == cv::Rect(0, 260, 60, 200), "partially visible boxes must be clamped");
}

cv::Mat make_standard_output(bool channels_first) {
    const int sizes_channels_first[] = {1, 84, 8400};
    const int sizes_candidates_first[] = {1, 8400, 84};
    cv::Mat output(
        3,
        channels_first ? sizes_channels_first : sizes_candidates_first,
        CV_32F,
        cv::Scalar(0));

    const int candidate = 17;
    const float values[] = {320.0f, 320.0f, 100.0f, 100.0f};
    for (int channel = 0; channel < 4; ++channel) {
        int index_channels_first[] = {0, channel, candidate};
        int index_candidates_first[] = {0, candidate, channel};
        output.at<float>(channels_first ? index_channels_first : index_candidates_first) = values[channel];
    }
    int score_channels_first[] = {0, 7, candidate};
    int score_candidates_first[] = {0, candidate, 7};
    output.at<float>(channels_first ? score_channels_first : score_candidates_first) = 0.90f;
    return output;
}

void verify_standard_layout_inference() {
    const vp_yolo11_letterbox letterbox {
        1.0f,
        0,
        0,
        640,
        640,
    };

    for (const auto channels_first : {true, false}) {
        const auto detections = vp_nodes::decode_yolo11_output(
            make_standard_output(channels_first),
            letterbox,
            0.25f,
            0.45f);
        require(detections.size() == 1, "standard YOLO11 layout was not inferred");
        require(detections[0].class_id == 3, "standard YOLO11 class was not decoded");
        require(detections[0].box == cv::Rect(270, 270, 100, 100), "standard YOLO11 box was not decoded");
    }
}

void verify_ambiguous_layout_rejection() {
    const int ambiguous_sizes[] = {1, 6, 5};
    cv::Mat ambiguous(3, ambiguous_sizes, CV_32F, cv::Scalar(0));
    const vp_yolo11_letterbox letterbox {
        1.0f,
        0,
        0,
        640,
        640,
    };
    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                ambiguous,
                letterbox,
                0.25f,
                0.45f);
        },
        "ambiguous YOLO11 layout without a class hint must be rejected");

    const int mismatched_sizes[] = {1, 7, 5};
    cv::Mat mismatched(3, mismatched_sizes, CV_32F, cv::Scalar(0));
    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                mismatched,
                letterbox,
                0.25f,
                0.45f,
                2);
        },
        "a class hint that matches neither dimension must be rejected");

    const int double_match_sizes[] = {1, 6, 6};
    cv::Mat double_match(3, double_match_sizes, CV_32F, cv::Scalar(0));
    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                double_match,
                letterbox,
                0.25f,
                0.45f,
                2);
        },
        "a class hint that matches both dimensions must be rejected");
}

void verify_non_finite_configuration_rejection() {
    const auto output = make_channels_first_output();
    const vp_yolo11_letterbox valid_letterbox {
        0.5f,
        0,
        140,
        1280,
        720,
    };

    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                output,
                valid_letterbox,
                std::numeric_limits<float>::quiet_NaN(),
                0.45f,
                2);
        },
        "NaN confidence threshold must be rejected");
    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                output,
                valid_letterbox,
                0.25f,
                std::numeric_limits<float>::infinity(),
                2);
        },
        "infinite NMS threshold must be rejected");

    auto invalid_letterbox = valid_letterbox;
    invalid_letterbox.scale = std::numeric_limits<float>::quiet_NaN();
    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                output,
                invalid_letterbox,
                0.25f,
                0.45f,
                2);
        },
        "NaN letterbox scale must be rejected");

    invalid_letterbox.scale = std::numeric_limits<float>::infinity();
    require_invalid_argument(
        [&]() {
            vp_nodes::decode_yolo11_output(
                output,
                invalid_letterbox,
                0.25f,
                0.45f,
                2);
        },
        "infinite letterbox scale must be rejected");
}

void verify_model_load_fail_fast_contract() {
    const std::string model_path = "C:/external-models/yolo11s.onnx";
    try {
        vp_nodes::require_yolo11_model_loaded(false, model_path);
    }
    catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find(model_path) != std::string::npos, "model-load error must include the model path");
        require(message.find("failed") != std::string::npos, "model-load error must state the failure reason");
        vp_nodes::require_yolo11_model_loaded(true, model_path);
        return;
    }
    throw std::runtime_error("an empty YOLO11 network must fail before worker initialization");
}

void verify_class_id_offset_target_mapping() {
    const vp_yolo11_detection detection {
        cv::Rect(10, 20, 30, 40),
        3,
        0.90f,
    };
    const auto target = vp_nodes::make_yolo11_frame_target(
        detection,
        100,
        7,
        2,
        "cyclist");

    require(target->primary_class_id == 103, "target class ID must include class_id_offset");
    require(target->primary_label == "cyclist", "target label must use the raw class label");
    require_close(target->primary_score, 0.90f, "target confidence was not preserved");
    require(target->frame_index == 7, "target frame index was not preserved");
    require(target->channel_index == 2, "target channel index was not preserved");
    require(target->track_id == -1, "detector must not assign or overwrite track IDs");
}

}  // namespace

int main() {
    try {
        verify_layout(make_channels_first_output());
        verify_layout(make_candidates_first_output());
        verify_clamp_and_zero_area_rejection();
        verify_standard_layout_inference();
        verify_ambiguous_layout_rejection();
        verify_non_finite_configuration_rejection();
        verify_model_load_fail_fast_contract();
        verify_class_id_offset_target_mapping();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    std::cout << "YOLO11 postprocess tests passed" << std::endl;
    return 0;
}
