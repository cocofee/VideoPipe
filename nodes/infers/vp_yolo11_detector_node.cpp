#include "vp_yolo11_detector_node.h"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

#include "../../objects/vp_frame_target.h"

#if defined(VP_WITH_YOLO11) && !defined(VP_YOLO11_POSTPROCESS_ONLY)
#include <opencv2/imgproc.hpp>
#endif

namespace vp_nodes {
    namespace {
        enum class vp_yolo11_layout {
            CHANNELS_FIRST,
            CANDIDATES_FIRST,
        };

        void validate_thresholds(
            float confidence_threshold,
            float nms_threshold) {
            if (!std::isfinite(confidence_threshold) ||
                !std::isfinite(nms_threshold) ||
                confidence_threshold < 0.0f ||
                confidence_threshold > 1.0f ||
                nms_threshold < 0.0f ||
                nms_threshold > 1.0f) {
                throw std::invalid_argument("YOLO11 thresholds must be finite and between zero and one");
            }
        }

        vp_yolo11_layout resolve_layout(
            int first_dimension,
            int second_dimension,
            int class_count_hint) {
            if (class_count_hint < 0 ||
                class_count_hint > std::numeric_limits<int>::max() - 4) {
                throw std::invalid_argument("invalid YOLO11 class count hint");
            }
            if (class_count_hint > 0) {
                const auto expected_channels = class_count_hint + 4;
                const auto first_matches = first_dimension == expected_channels;
                const auto second_matches = second_dimension == expected_channels;
                if (first_matches == second_matches) {
                    throw std::invalid_argument(
                        "YOLO11 output layout does not uniquely match the class count hint");
                }
                return first_matches
                    ? vp_yolo11_layout::CHANNELS_FIRST
                    : vp_yolo11_layout::CANDIDATES_FIRST;
            }

            if (first_dimension < 5 && second_dimension >= 5) {
                return vp_yolo11_layout::CANDIDATES_FIRST;
            }
            if (second_dimension < 5 && first_dimension >= 5) {
                return vp_yolo11_layout::CHANNELS_FIRST;
            }
            const auto smaller_dimension = std::min(first_dimension, second_dimension);
            const auto larger_dimension = std::max(first_dimension, second_dimension);
            if (smaller_dimension < 5 ||
                static_cast<long long>(larger_dimension) <
                    static_cast<long long>(smaller_dimension) * 4) {
                throw std::invalid_argument("ambiguous YOLO11 output layout");
            }
            return first_dimension < second_dimension
                ? vp_yolo11_layout::CHANNELS_FIRST
                : vp_yolo11_layout::CANDIDATES_FIRST;
        }

        cv::Rect restore_box(
            float center_x,
            float center_y,
            float width,
            float height,
            const vp_yolo11_letterbox& letterbox) {
            if (letterbox.scale <= 0.0f ||
                letterbox.original_width <= 0 ||
                letterbox.original_height <= 0 ||
                width <= 0.0f ||
                height <= 0.0f) {
                return {};
            }

            auto left = (center_x - width / 2.0f - letterbox.pad_x) / letterbox.scale;
            auto top = (center_y - height / 2.0f - letterbox.pad_y) / letterbox.scale;
            auto right = (center_x + width / 2.0f - letterbox.pad_x) / letterbox.scale;
            auto bottom = (center_y + height / 2.0f - letterbox.pad_y) / letterbox.scale;

            left = std::clamp(left, 0.0f, static_cast<float>(letterbox.original_width));
            top = std::clamp(top, 0.0f, static_cast<float>(letterbox.original_height));
            right = std::clamp(right, 0.0f, static_cast<float>(letterbox.original_width));
            bottom = std::clamp(bottom, 0.0f, static_cast<float>(letterbox.original_height));

            const auto x = std::clamp(
                static_cast<int>(std::floor(left)),
                0,
                letterbox.original_width);
            const auto y = std::clamp(
                static_cast<int>(std::floor(top)),
                0,
                letterbox.original_height);
            const auto right_edge = std::clamp(
                static_cast<int>(std::ceil(right)),
                0,
                letterbox.original_width);
            const auto bottom_edge = std::clamp(
                static_cast<int>(std::ceil(bottom)),
                0,
                letterbox.original_height);

            if (right_edge <= x || bottom_edge <= y) {
                return {};
            }
            return cv::Rect(x, y, right_edge - x, bottom_edge - y);
        }
    }

    std::vector<vp_yolo11_detection> decode_yolo11_output(
        const cv::Mat& output,
        const vp_yolo11_letterbox& letterbox,
        float confidence_threshold,
        float nms_threshold,
        int class_count_hint) {
        if (output.empty() || output.type() != CV_32F || output.dims != 3 || output.size[0] != 1) {
            throw std::invalid_argument("YOLO11 output must be a [1, *, *] CV_32F tensor");
        }
        validate_thresholds(confidence_threshold, nms_threshold);
        if (!std::isfinite(letterbox.scale) ||
            letterbox.scale <= 0.0f ||
            letterbox.original_width <= 0 ||
            letterbox.original_height <= 0) {
            throw std::invalid_argument("YOLO11 letterbox metadata is invalid");
        }

        const auto layout = resolve_layout(output.size[1], output.size[2], class_count_hint);
        const auto channels = layout == vp_yolo11_layout::CHANNELS_FIRST
            ? output.size[1]
            : output.size[2];
        const auto candidates = layout == vp_yolo11_layout::CHANNELS_FIRST
            ? output.size[2]
            : output.size[1];
        if (channels < 5 || candidates <= 0) {
            throw std::invalid_argument("YOLO11 output has no class scores or candidates");
        }

        const auto continuous_output = output.isContinuous() ? output : output.clone();
        const auto* data = continuous_output.ptr<float>(0);
        const auto value_at = [&](int candidate, int channel) {
            if (layout == vp_yolo11_layout::CHANNELS_FIRST) {
                return data[channel * candidates + candidate];
            }
            return data[candidate * channels + channel];
        };

        std::vector<vp_yolo11_detection> decoded;
        for (int candidate = 0; candidate < candidates; ++candidate) {
            auto class_id = -1;
            auto confidence = -1.0f;
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

            const auto center_x = value_at(candidate, 0);
            const auto center_y = value_at(candidate, 1);
            const auto width = value_at(candidate, 2);
            const auto height = value_at(candidate, 3);
            if (!std::isfinite(center_x) || !std::isfinite(center_y) ||
                !std::isfinite(width) || !std::isfinite(height)) {
                continue;
            }

            const auto box = restore_box(center_x, center_y, width, height, letterbox);
            if (box.area() <= 0) {
                continue;
            }
            decoded.push_back({box, class_id, confidence});
        }

        std::map<int, std::vector<int>> indices_by_class;
        for (int index = 0; index < static_cast<int>(decoded.size()); ++index) {
            indices_by_class[decoded[index].class_id].push_back(index);
        }

        std::vector<vp_yolo11_detection> kept;
        for (const auto& class_indices : indices_by_class) {
            std::vector<cv::Rect> boxes;
            std::vector<float> confidences;
            for (const auto index : class_indices.second) {
                boxes.push_back(decoded[index].box);
                confidences.push_back(decoded[index].confidence);
            }

            std::vector<int> selected;
            cv::dnn::NMSBoxes(
                boxes,
                confidences,
                confidence_threshold,
                nms_threshold,
                selected);
            for (const auto selected_index : selected) {
                kept.push_back(decoded[class_indices.second[selected_index]]);
            }
        }

        std::sort(
            kept.begin(),
            kept.end(),
            [](const vp_yolo11_detection& left, const vp_yolo11_detection& right) {
                return left.confidence > right.confidence;
            });
        return kept;
    }

    void require_yolo11_model_loaded(
        bool model_loaded,
        const std::string& model_path) {
        if (!model_loaded) {
            throw std::runtime_error(
                "YOLO11 model load failed for '" + model_path +
                "': OpenCV DNN returned an empty network");
        }
    }

    std::shared_ptr<vp_objects::vp_frame_target> make_yolo11_frame_target(
        const vp_yolo11_detection& detection,
        int class_id_offset,
        int frame_index,
        int channel_index,
        const std::string& label) {
        return std::make_shared<vp_objects::vp_frame_target>(
            detection.box.x,
            detection.box.y,
            detection.box.width,
            detection.box.height,
            detection.class_id + class_id_offset,
            detection.confidence,
            frame_index,
            channel_index,
            label);
    }

#if defined(VP_WITH_YOLO11) && !defined(VP_YOLO11_POSTPROCESS_ONLY)
    vp_yolo11_detector_node::vp_yolo11_detector_node(
        std::string node_name,
        std::string model_path,
        std::string labels_path,
        int input_width,
        int input_height,
        float confidence_threshold,
        float nms_threshold):
        vp_primary_infer_node(
            node_name,
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
            true),
        confidence_threshold(confidence_threshold),
        nms_threshold(nms_threshold) {
        if (input_width <= 0 || input_height <= 0) {
            throw std::invalid_argument("YOLO11 input dimensions must be positive");
        }
        validate_thresholds(confidence_threshold, nms_threshold);
        require_yolo11_model_loaded(!net.empty(), model_path);
        this->initialized();
    }

    vp_yolo11_detector_node::~vp_yolo11_detector_node() {
        deinitialized();
    }

    void vp_yolo11_detector_node::preprocess(
        const std::vector<cv::Mat>& mats_to_infer,
        cv::Mat& blob_to_infer) {
        std::vector<cv::Mat> letterboxed_frames;
        current_letterboxes.clear();
        letterboxed_frames.reserve(mats_to_infer.size());
        current_letterboxes.reserve(mats_to_infer.size());

        for (const auto& frame : mats_to_infer) {
            if (frame.empty()) {
                throw std::invalid_argument("YOLO11 cannot preprocess an empty frame");
            }

            const auto resize_scale = std::min(
                static_cast<float>(input_width) / frame.cols,
                static_cast<float>(input_height) / frame.rows);
            const auto resized_width = std::max(
                1,
                static_cast<int>(std::round(frame.cols * resize_scale)));
            const auto resized_height = std::max(
                1,
                static_cast<int>(std::round(frame.rows * resize_scale)));
            const auto pad_x = (input_width - resized_width) / 2;
            const auto pad_y = (input_height - resized_height) / 2;

            cv::Mat resized;
            cv::resize(
                frame,
                resized,
                cv::Size(resized_width, resized_height),
                0.0,
                0.0,
                cv::INTER_LINEAR);

            cv::Mat letterboxed;
            cv::copyMakeBorder(
                resized,
                letterboxed,
                pad_y,
                input_height - resized_height - pad_y,
                pad_x,
                input_width - resized_width - pad_x,
                cv::BORDER_CONSTANT,
                cv::Scalar(114, 114, 114));
            letterboxed_frames.push_back(letterboxed);
            current_letterboxes.push_back({
                resize_scale,
                pad_x,
                pad_y,
                frame.cols,
                frame.rows,
            });
        }

        cv::dnn::blobFromImages(
            letterboxed_frames,
            blob_to_infer,
            1.0f / 255.0f,
            cv::Size(input_width, input_height),
            cv::Scalar(0),
            true,
            false,
            CV_32F);
    }

    void vp_yolo11_detector_node::postprocess(
        const std::vector<cv::Mat>& raw_outputs,
        const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
        if (raw_outputs.size() != 1 ||
            frame_meta_with_batch.size() != 1 ||
            current_letterboxes.size() != 1) {
            throw std::invalid_argument("YOLO11 detector expects one frame and one output tensor");
        }

        auto& frame_meta = frame_meta_with_batch[0];
        const auto detections = decode_yolo11_output(
            raw_outputs[0],
            current_letterboxes[0],
            confidence_threshold,
            nms_threshold,
            static_cast<int>(labels.size()));
        for (const auto& detection : detections) {
            const auto label = detection.class_id < static_cast<int>(labels.size())
                ? labels[detection.class_id]
                : "";
            frame_meta->targets.push_back(make_yolo11_frame_target(
                detection,
                class_id_offset,
                frame_meta->frame_index,
                frame_meta->channel_index,
                label));
        }
    }
#endif
}
