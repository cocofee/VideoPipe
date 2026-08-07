#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../nodes/vp_file_src_node.h"
#include "../nodes/track/vp_bytetrack_node.h"
#include "../nodes/track/vp_sort_track_node.h"
#include "../nodes/osd/vp_osd_node_v3.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"
#include "cycling_upgrade_options.h"

#ifdef VP_WITH_TRT
#include "../nodes/infers/vp_trt_yolov8_detector.h"
#endif

#ifdef VP_WITH_YOLO11
#include "../nodes/infers/vp_yolo11_detector_node.h"
#endif

#ifdef VP_WITH_PADDLE
#include "../nodes/infers/vp_ppocr_text_detector_node.h"
#endif

#if defined(VP_WITH_PADDLE) && defined(VP_WITH_ASYNC_OCR)
#include "../nodes/infers/vp_async_ocr_node.h"
#endif

namespace {
    struct ocr_runtime_paths {
        std::string detector;
        std::string classifier;
        std::string recognizer;
        std::string dictionary;
    };

    std::string join_path(const std::string& directory, const std::string& file_name) {
        if (directory.empty()) {
            return file_name;
        }
        const auto last = directory.back();
        if (last == '/' || last == '\\') {
            return directory + file_name;
        }
        return directory + "/" + file_name;
    }

    std::string environment_value(const char* name) {
        const auto* value = std::getenv(name);
        return value == nullptr ? std::string {} : value;
    }

    ocr_runtime_paths resolve_ocr_paths(const std::string& runtime_directory) {
        const auto ocr_directory = join_path(runtime_directory, "ocr");
        return {
            join_path(ocr_directory, "ch_PP-OCRv3_det_infer"),
            join_path(ocr_directory, "ch_ppocr_mobile_v2.0_cls_infer"),
            join_path(ocr_directory, "ch_PP-OCRv3_rec_infer"),
            join_path(ocr_directory, "ppocr_keys_v1.txt"),
        };
    }

    void print_config(
        const vp_samples::cycling_upgrade_options& options,
        const ocr_runtime_paths& ocr_paths) {
        std::cout
            << "profile=" << (options.profile.empty() ? "legacy" : options.profile) << '\n'
            << "detector=" << options.detector << '\n'
            << "tracker=" << options.tracker << '\n'
            << "ocr=" << options.ocr << '\n'
            << "runtime_dir=" << options.runtime_dir << '\n'
            << "model=" << options.model_path << '\n'
            << "labels=" << options.labels_path << '\n';
        if (options.ocr != "none") {
            std::cout
                << "ocr_detector=" << ocr_paths.detector << '\n'
                << "ocr_classifier=" << ocr_paths.classifier << '\n'
                << "ocr_recognizer=" << ocr_paths.recognizer << '\n'
                << "ocr_dictionary=" << ocr_paths.dictionary << '\n';
        }
    }

    void print_usage() {
        std::cerr
            << "usage: cycling_upgrade_sample <video_path> "
            << "[--profile=baseline|candidate] "
            << "[--detector=yolov8|yolo11] "
            << "[--tracker=none|sort|bytetrack] "
            << "[--ocr=none|sync|async] "
            << "[--runtime-dir=<path>] [--model=<path>] [--labels=<path>] "
            << "[--print-config]"
            << std::endl;
    }
}

int main(int argc, char** argv) {
    std::vector<std::string> arguments;
    arguments.reserve(argc > 0 ? static_cast<std::size_t>(argc - 1) : 0);
    for (int index = 1; index < argc; ++index) {
        arguments.emplace_back(argv[index]);
    }

    vp_samples::cycling_upgrade_options options;
    std::string parse_error;
    if (!vp_samples::parse_cycling_upgrade_options(arguments, options, parse_error)) {
        std::cerr << parse_error << std::endl;
        print_usage();
        return 2;
    }

    if (options.runtime_dir.empty()) {
        options.runtime_dir = environment_value("VP_CYCLING_RUNTIME_DIR");
    }
    if (options.runtime_dir.empty()) {
        options.runtime_dir = environment_value("VP_CYCLING_MODEL_DIR");
    }
    if (options.model_path.empty()) {
        if (options.runtime_dir.empty() && !options.print_config) {
            std::cerr
                << "VP_CYCLING_RUNTIME_DIR, VP_CYCLING_MODEL_DIR, --runtime-dir, "
                << "or --model is required"
                << std::endl;
            return 2;
        }
        if (!options.runtime_dir.empty()) {
            options.model_path = join_path(
                options.runtime_dir,
                options.detector == "yolo11" ? "yolo11s.onnx" : "yolov8s.engine");
        }
    }
    if (options.labels_path.empty() && !options.runtime_dir.empty()) {
        options.labels_path = join_path(options.runtime_dir, "labels.txt");
    }
    if (options.ocr != "none" && options.runtime_dir.empty() && !options.print_config) {
        std::cerr << "OCR requires VP_CYCLING_RUNTIME_DIR or --runtime-dir" << std::endl;
        return 2;
    }

    const auto ocr_paths = resolve_ocr_paths(options.runtime_dir);
    if (options.print_config) {
        print_config(options, ocr_paths);
        return 0;
    }

    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    try {
        auto file_source = std::make_shared<vp_nodes::vp_file_src_node>(
            "cycling_file_source",
            0,
            options.video_path);
        std::shared_ptr<vp_nodes::vp_node> detector;

        if (options.detector == "yolov8") {
#ifdef VP_WITH_TRT
            detector = std::make_shared<vp_nodes::vp_trt_yolov8_detector>(
                "cycling_yolov8_detector",
                options.model_path,
                options.labels_path);
#else
            std::cerr << "YOLOv8 requires VP_WITH_TRT=ON" << std::endl;
            return 2;
#endif
        }
        else if (options.detector == "yolo11") {
#ifdef VP_WITH_YOLO11
            detector = std::make_shared<vp_nodes::vp_yolo11_detector_node>(
                "cycling_yolo11_detector",
                options.model_path,
                options.labels_path);
#else
            std::cerr << "YOLO11 requires VP_WITH_YOLO11=ON" << std::endl;
            return 2;
#endif
        }
        else {
            std::cerr << "unsupported detector: " << options.detector << std::endl;
            return 2;
        }

        std::shared_ptr<vp_nodes::vp_node> pipeline_tail = detector;
        if (options.tracker == "sort") {
            auto tracker = std::make_shared<vp_nodes::vp_sort_track_node>("cycling_sort_tracker");
            tracker->attach_to({pipeline_tail});
            pipeline_tail = tracker;
        }
        else if (options.tracker == "bytetrack") {
            auto tracker = std::make_shared<vp_nodes::vp_bytetrack_node>("cycling_bytetrack_tracker");
            tracker->attach_to({pipeline_tail});
            pipeline_tail = tracker;
        }

        if (options.ocr == "sync") {
#ifdef VP_WITH_PADDLE
            auto ocr = std::make_shared<vp_nodes::vp_ppocr_text_detector_node>(
                "cycling_sync_ocr",
                ocr_paths.detector,
                ocr_paths.classifier,
                ocr_paths.recognizer,
                ocr_paths.dictionary);
            ocr->attach_to({pipeline_tail});
            pipeline_tail = ocr;
#else
            std::cerr << "synchronous OCR requires VP_WITH_PADDLE=ON" << std::endl;
            return 2;
#endif
        }
        else if (options.ocr == "async") {
#if defined(VP_WITH_PADDLE) && defined(VP_WITH_ASYNC_OCR)
            auto recognizer = std::make_shared<vp_nodes::vp_ppocr_text_detector_node>(
                "cycling_async_ocr_runtime",
                ocr_paths.detector,
                ocr_paths.classifier,
                ocr_paths.recognizer,
                ocr_paths.dictionary);
            vp_nodes::vp_async_ocr_config config;
            auto ocr = std::make_shared<vp_nodes::vp_async_ocr_node>(
                "cycling_async_ocr",
                config,
                [recognizer](const cv::Mat& crop) {
                    return recognizer->recognize_crop(crop);
                });
            ocr->attach_to({pipeline_tail});
            pipeline_tail = ocr;
#else
            std::cerr
                << "asynchronous OCR requires VP_WITH_PADDLE=ON and "
                << "VP_WITH_ASYNC_OCR=ON"
                << std::endl;
            return 2;
#endif
        }

        auto osd = std::make_shared<vp_nodes::vp_osd_node_v3>("cycling_osd");
        auto screen = std::make_shared<vp_nodes::vp_screen_des_node>("cycling_screen", 0);

        detector->attach_to({file_source});
        osd->attach_to({pipeline_tail});
        screen->attach_to({osd});
        file_source->start();

        vp_utils::vp_analysis_board board({file_source});
        board.display();
    }
    catch (const std::exception& error) {
        std::cerr
            << "failed to initialize profile '"
            << (options.profile.empty() ? "legacy" : options.profile)
            << "' (detector=" << options.detector
            << ", tracker=" << options.tracker
            << ", ocr=" << options.ocr
            << ", model=" << options.model_path
            << "): " << error.what()
            << std::endl;
        return 1;
    }
    return 0;
}
