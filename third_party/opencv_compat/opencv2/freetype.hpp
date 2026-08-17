#pragma once

#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace cv {
namespace freetype {

    class FreeType2: public cv::Algorithm {
    public:
        void loadFontData(const cv::String&, int) {
        }

        void putText(cv::InputOutputArray image,
                     const cv::String& text,
                     cv::Point org,
                     int font_height,
                     cv::Scalar color,
                     int thickness,
                     int line_type,
                     bool bottom_left_origin = false) {
            const auto font_scale = std::max(0.1, static_cast<double>(font_height) / 30.0);
            cv::putText(image,
                        text,
                        org,
                        cv::FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness,
                        line_type,
                        bottom_left_origin);
        }

        cv::Size getTextSize(const cv::String& text,
                             int font_height,
                             int thickness,
                             int* baseline) {
            const auto font_scale = std::max(0.1, static_cast<double>(font_height) / 30.0);
            return cv::getTextSize(text,
                                   cv::FONT_HERSHEY_SIMPLEX,
                                   font_scale,
                                   thickness,
                                   baseline);
        }
    };

    inline cv::Ptr<FreeType2> createFreeType2() {
        return cv::makePtr<FreeType2>();
    }

}
}
