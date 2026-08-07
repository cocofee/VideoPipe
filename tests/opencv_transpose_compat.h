#pragma once

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>

#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 6
namespace cv {
    void transposeND(InputArray src, const std::vector<int>& order, OutputArray dst);
}
#endif
