#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../samples/cycling_upgrade_options.h"

namespace {
    void require(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    vp_samples::cycling_upgrade_options parse(const std::vector<std::string>& arguments) {
        vp_samples::cycling_upgrade_options options;
        std::string error;
        require(
            vp_samples::parse_cycling_upgrade_options(arguments, options, error),
            "option parsing failed: " + error);
        return options;
    }

    void test_legacy_defaults_remain_detection_only() {
        const auto options = parse({"race.mp4"});
        require(options.profile.empty(), "legacy invocation must not select a profile");
        require(options.detector == "yolov8", "legacy detector must remain YOLOv8");
        require(options.tracker == "none", "legacy invocation must not add a tracker");
        require(options.ocr == "none", "legacy invocation must not add OCR");
    }

    void test_profile_defaults() {
        const auto baseline = parse({"race.mp4", "--profile=baseline"});
        require(baseline.detector == "yolov8", "baseline detector mismatch");
        require(baseline.tracker == "sort", "baseline tracker mismatch");
        require(baseline.ocr == "sync", "baseline OCR mismatch");

        const auto candidate = parse({"race.mp4", "--profile=candidate"});
        require(candidate.detector == "yolo11", "candidate detector mismatch");
        require(candidate.tracker == "bytetrack", "candidate tracker mismatch");
        require(candidate.ocr == "async", "candidate OCR mismatch");
    }

    void test_component_overrides_are_order_independent() {
        const auto options = parse({
            "--tracker=sort",
            "--ocr=none",
            "race.mp4",
            "--profile=candidate",
            "--detector=yolov8",
            "--runtime-dir=C:/cycling/baseline"});
        require(options.profile == "candidate", "candidate profile was not retained");
        require(options.detector == "yolov8", "detector override was not applied");
        require(options.tracker == "sort", "tracker override was not applied");
        require(options.ocr == "none", "OCR override was not applied");
        require(
            options.runtime_dir == "C:/cycling/baseline",
            "runtime directory override was not applied");
    }

    void test_print_config_does_not_require_video() {
        const auto options = parse({"--profile=baseline", "--print-config"});
        require(options.print_config, "print-config flag was not retained");
        require(options.video_path.empty(), "print-config should not invent a video path");
    }

    void test_invalid_values_are_rejected() {
        for (const auto& arguments : std::vector<std::vector<std::string>> {
                 {"race.mp4", "--profile=production"},
                 {"race.mp4", "--profile="},
                 {"race.mp4", "--detector=yolov12"},
                 {"race.mp4", "--tracker=deep-sort"},
                 {"race.mp4", "--ocr=remote"},
                 {"race.mp4", "--ocr=async"},
                 {"race.mp4", "--unknown=value"},
                 {"first.mp4", "second.mp4"},
                 {"--profile=baseline"},
             }) {
            vp_samples::cycling_upgrade_options options;
            std::string error;
            require(
                !vp_samples::parse_cycling_upgrade_options(arguments, options, error),
                "invalid arguments were accepted");
            require(!error.empty(), "invalid arguments must report an error");
        }
    }
}

int main() {
    test_legacy_defaults_remain_detection_only();
    test_profile_defaults();
    test_component_overrides_are_order_independent();
    test_print_config_does_not_require_video();
    test_invalid_values_are_rejected();
    std::cout << "Cycling upgrade option tests passed" << std::endl;
    return 0;
}
