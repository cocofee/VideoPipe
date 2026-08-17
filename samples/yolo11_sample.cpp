#include <cstdlib>
#include <memory>
#include <string>

#include "../nodes/vp_camera_src_node.h"
#include "../nodes/infers/vp_yolo11_detector_node.h"
#include "../nodes/osd/vp_osd_node_v3.h"
#include "../nodes/vp_screen_des_node.h"
#include "../utils/analysis_board/vp_analysis_board.h"

int main(int argc, char** argv) {
    VP_SET_LOG_LEVEL(vp_utils::vp_log_level::INFO);
    VP_LOGGER_INIT();

    const auto model_path = argc > 1 ? argv[1] : "./vp_data/models/yolo11n.onnx";
    const auto labels_path = argc > 2 ? argv[2] : "./vp_data/models/coco_80classes.txt";
    const auto device_index = argc > 3 ? std::atoi(argv[3]) : 0;

    auto camera = std::make_shared<vp_nodes::vp_camera_src_node>(
        "camera_0", 0, device_index, 0.5f, 1920, 1080, 30, 1);
    auto detector = std::make_shared<vp_nodes::vp_yolo11_detector_node>(
        "yolo11_detector", model_path, labels_path);
    auto osd = std::make_shared<vp_nodes::vp_osd_node_v3>("osd_0");
    auto screen = std::make_shared<vp_nodes::vp_screen_des_node>("screen_0", 0);

    detector->attach_to({camera});
    osd->attach_to({detector});
    screen->attach_to({osd});

    camera->start();
    vp_utils::vp_analysis_board board({camera});
    board.display();
    return 0;
}
