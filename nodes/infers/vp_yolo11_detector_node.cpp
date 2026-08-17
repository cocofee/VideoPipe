#include "vp_yolo11_detector_node.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../../objects/vp_frame_target.h"

namespace vp_nodes {
    namespace {
        struct yolo11_detection {
            cv::Rect box;
            int class_id;
            float confidence;
        };

        cv::Rect restore_box(float cx, float cy, float width, float height,
                             const vp_yolo11_letterbox& letterbox) {
            if (letterbox.scale <= 0.0f) {
                return {};
            }

            const auto left = std::clamp((cx - width / 2.0f - letterbox.pad_x) / letterbox.scale,
                                         0.0f, static_cast<float>(letterbox.original_width));
            const auto top = std::clamp((cy - height / 2.0f - letterbox.pad_y) / letterbox.scale,
                                        0.0f, static_cast<float>(letterbox.original_height));
            const auto right = std::clamp((cx + width / 2.0f - letterbox.pad_x) / letterbox.scale,
                                          0.0f, static_cast<float>(letterbox.original_width));
            const auto bottom = std::clamp((cy + height / 2.0f - letterbox.pad_y) / letterbox.scale,
                                           0.0f, static_cast<float>(letterbox.original_height));
            const auto x = static_cast<int>(std::floor(left));
            const auto y = static_cast<int>(std::floor(top));
            const auto x2 = static_cast<int>(std::ceil(right));
            const auto y2 = static_cast<int>(std::ceil(bottom));
            if (x2 <= x || y2 <= y) {
                return {};
            }
            return cv::Rect(x, y, x2 - x, y2 - y);
        }
    }

    void vp_yolo11_detector_node::validate_thresholds(float confidence, float nms) {
        if (!std::isfinite(confidence) || !std::isfinite(nms) ||
            confidence < 0.0f || confidence > 1.0f ||
            nms < 0.0f || nms > 1.0f) {
            throw std::invalid_argument("YOLO11 thresholds must be between zero and one");
        }
    }

    vp_yolo11_detector_node::vp_yolo11_detector_node(
        std::string node_name,
        std::string model_path,
        std::string labels_path,
        int input_width,
        int input_height,
        float confidence_threshold,
        float nms_threshold):
        vp_primary_infer_node(node_name,
                              model_path,
                              "",
                              labels_path,
                              input_width,
                              input_height,
                              1,
                              0,
                              1.0f / 255.0f,
                              cv::Scalar(0),
                              cv::Scalar(1),
                              true,
                              false),
        confidence_threshold(confidence_threshold),
        nms_threshold(nms_threshold) {
        if (input_width <= 0 || input_height <= 0) {
            throw std::invalid_argument("YOLO11 input dimensions must be positive");
        }
        validate_thresholds(confidence_threshold, nms_threshold);
        if (net.empty()) {
            throw std::runtime_error("failed to load YOLO11 model: " + model_path);
        }
        this->initialized();
    }

    vp_yolo11_detector_node::~vp_yolo11_detector_node() {
        deinitialized();
    }

    void vp_yolo11_detector_node::preprocess(
        const std::vector<cv::Mat>& mats_to_infer,
        cv::Mat& blob_to_infer) {
        current_letterboxes.clear();
        std::vector<cv::Mat> letterboxed_frames;
        letterboxed_frames.reserve(mats_to_infer.size());
        current_letterboxes.reserve(mats_to_infer.size());

        for (const auto& frame : mats_to_infer) {
            if (frame.empty()) {
                throw std::invalid_argument("YOLO11 cannot preprocess an empty frame");
            }

            const auto scale = std::min(
                static_cast<float>(input_width) / static_cast<float>(frame.cols),
                static_cast<float>(input_height) / static_cast<float>(frame.rows));
            const auto resized_width = std::max(1, static_cast<int>(std::round(frame.cols * scale)));
            const auto resized_height = std::max(1, static_cast<int>(std::round(frame.rows * scale)));
            const auto pad_x = (input_width - resized_width) / 2;
            const auto pad_y = (input_height - resized_height) / 2;

            cv::Mat resized;
            cv::resize(frame, resized, cv::Size(resized_width, resized_height));
            cv::Mat letterboxed;
            cv::copyMakeBorder(resized,
                               letterboxed,
                               pad_y,
                               input_height - resized_height - pad_y,
                               pad_x,
                               input_width - resized_width - pad_x,
                               cv::BORDER_CONSTANT,
                               cv::Scalar(114, 114, 114));
            letterboxed_frames.push_back(letterboxed);
            current_letterboxes.push_back({scale, pad_x, pad_y, frame.cols, frame.rows});
        }

        cv::dnn::blobFromImages(letterboxed_frames,
                                blob_to_infer,
                                1.0f / 255.0f,
                                cv::Size(input_width, input_height),
                                cv::Scalar(),
                                true,
                                false,
                                CV_32F);
    }

    void vp_yolo11_detector_node::postprocess(
        const std::vector<cv::Mat>& raw_outputs,
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        if (raw_outputs.size() != 1 || frame_meta_with_batch.size() != 1 ||
            current_letterboxes.size() != 1) {
            throw std::invalid_argument("YOLO11 detector expects one frame and one output tensor");
        }

        const auto& output = raw_outputs[0];
        if (output.empty() || output.type() != CV_32F || output.dims != 3 || output.size[0] != 1) {
            throw std::invalid_argument("YOLO11 output must be a [1, *, *] float tensor");
        }

        const auto channels_first = output.size[1] < output.size[2];
        const auto channels = channels_first ? output.size[1] : output.size[2];
        const auto candidates = channels_first ? output.size[2] : output.size[1];
        if (channels < 5 || candidates <= 0) {
            throw std::invalid_argument("YOLO11 output has an unsupported shape");
        }

        const auto continuous = output.isContinuous() ? output : output.clone();
        const auto* data = continuous.ptr<float>(0);
        const auto value_at = [&](int candidate, int channel) {
            return channels_first
                ? data[channel * candidates + candidate]
                : data[candidate * channels + channel];
        };

        std::vector<yolo11_detection> decoded;
        for (int candidate = 0; candidate < candidates; ++candidate) {
            int class_id = -1;
            float confidence = -1.0f;
            for (int channel = 4; channel < channels; ++channel) {
                const auto score = value_at(candidate, channel);
                if (std::isfinite(score) && score > confidence) {
                    class_id = channel - 4;
                    confidence = score;
                }
            }
            if (class_id < 0 || confidence < confidence_threshold) {
                continue;
            }

            const auto box = restore_box(value_at(candidate, 0),
                                         value_at(candidate, 1),
                                         value_at(candidate, 2),
                                         value_at(candidate, 3),
                                         current_letterboxes[0]);
            if (box.area() > 0) {
                decoded.push_back({box, class_id, confidence});
            }
        }

        std::vector<yolo11_detection> kept;
        for (int class_id = 0; class_id < channels - 4; ++class_id) {
            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<int> indices;
            for (int index = 0; index < static_cast<int>(decoded.size()); ++index) {
                if (decoded[index].class_id == class_id) {
                    boxes.push_back(decoded[index].box);
                    scores.push_back(decoded[index].confidence);
                    indices.push_back(index);
                }
            }
            std::vector<int> selected;
            cv::dnn::NMSBoxes(boxes, scores, confidence_threshold, nms_threshold, selected);
            for (const auto selected_index : selected) {
                kept.push_back(decoded[indices[selected_index]]);
            }
        }

        auto& frame_meta = frame_meta_with_batch[0];
        for (const auto& detection : kept) {
            const auto label = detection.class_id < static_cast<int>(labels.size())
                ? labels[detection.class_id]
                : "";
            frame_meta->targets.push_back(std::make_shared<vp_objects::vp_frame_target>(
                detection.box.x,
                detection.box.y,
                detection.box.width,
                detection.box.height,
                detection.class_id,
                detection.confidence,
                frame_meta->frame_index,
                frame_meta->channel_index,
                label));
        }
    }
}
