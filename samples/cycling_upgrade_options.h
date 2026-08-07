#pragma once

#include <string>
#include <vector>

namespace vp_samples {
    struct cycling_upgrade_options {
        std::string profile;
        std::string detector = "yolov8";
        std::string tracker = "none";
        std::string ocr = "none";
        std::string video_path;
        std::string runtime_dir;
        std::string model_path;
        std::string labels_path;
        bool print_config = false;
    };

    bool parse_cycling_upgrade_options(
        const std::vector<std::string>& arguments,
        cycling_upgrade_options& options,
        std::string& error);
}
