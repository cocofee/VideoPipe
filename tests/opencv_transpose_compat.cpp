#include "opencv_transpose_compat.h"

#if CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 6
namespace cv {
    void transposeND(InputArray, const std::vector<int>&, OutputArray) {
        CV_Error(Error::StsNotImplemented, "transposeND requires OpenCV 4.6 or newer");
    }
}
#endif
