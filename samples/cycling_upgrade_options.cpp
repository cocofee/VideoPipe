#include "cycling_upgrade_options.h"

#include <optional>
#include <utility>

namespace vp_samples {
    namespace {
        bool starts_with(const std::string& value, const std::string& prefix) {
            return value.compare(0, prefix.size(), prefix) == 0;
        }

        std::string option_value(const std::string& argument, const std::string& prefix) {
            return argument.substr(prefix.size());
        }

        bool is_one_of(
            const std::string& value,
            const std::string& first,
            const std::string& second,
            const std::string& third = "") {
            return value == first || value == second || (!third.empty() && value == third);
        }
    }

    bool parse_cycling_upgrade_options(
        const std::vector<std::string>& arguments,
        cycling_upgrade_options& options,
        std::string& error) {
        options = {};
        error.clear();

        std::optional<std::string> detector_override;
        std::optional<std::string> tracker_override;
        std::optional<std::string> ocr_override;
        bool profile_provided = false;

        for (const auto& argument : arguments) {
            if (starts_with(argument, "--profile=")) {
                profile_provided = true;
                options.profile = option_value(argument, "--profile=");
            }
            else if (starts_with(argument, "--detector=")) {
                detector_override = option_value(argument, "--detector=");
            }
            else if (starts_with(argument, "--tracker=")) {
                tracker_override = option_value(argument, "--tracker=");
            }
            else if (starts_with(argument, "--ocr=")) {
                ocr_override = option_value(argument, "--ocr=");
            }
            else if (starts_with(argument, "--runtime-dir=")) {
                options.runtime_dir = option_value(argument, "--runtime-dir=");
            }
            else if (starts_with(argument, "--model=")) {
                options.model_path = option_value(argument, "--model=");
            }
            else if (starts_with(argument, "--labels=")) {
                options.labels_path = option_value(argument, "--labels=");
            }
            else if (argument == "--print-config") {
                options.print_config = true;
            }
            else if (starts_with(argument, "--")) {
                error = "unsupported option: " + argument;
                return false;
            }
            else if (options.video_path.empty()) {
                options.video_path = argument;
            }
            else {
                error = "only one video path is allowed";
                return false;
            }
        }

        if (profile_provided) {
            if (options.profile == "baseline") {
                options.detector = "yolov8";
                options.tracker = "sort";
                options.ocr = "sync";
            }
            else if (options.profile == "candidate") {
                options.detector = "yolo11";
                options.tracker = "bytetrack";
                options.ocr = "async";
            }
            else {
                error = "unsupported profile: " + options.profile;
                return false;
            }
        }

        if (detector_override) {
            options.detector = std::move(*detector_override);
        }
        if (tracker_override) {
            options.tracker = std::move(*tracker_override);
        }
        if (ocr_override) {
            options.ocr = std::move(*ocr_override);
        }

        if (!is_one_of(options.detector, "yolov8", "yolo11")) {
            error = "unsupported detector: " + options.detector;
            return false;
        }
        if (!is_one_of(options.tracker, "none", "sort", "bytetrack")) {
            error = "unsupported tracker: " + options.tracker;
            return false;
        }
        if (!is_one_of(options.ocr, "none", "sync", "async")) {
            error = "unsupported OCR mode: " + options.ocr;
            return false;
        }
        if (options.ocr == "async" && options.tracker == "none") {
            error = "asynchronous OCR requires sort or bytetrack";
            return false;
        }
        if (options.video_path.empty() && !options.print_config) {
            error = "video path is required";
            return false;
        }
        return true;
    }
}
