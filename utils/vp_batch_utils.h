#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vp_utils {
    struct vp_batch_range {
        int offset;
        int size;
    };

    inline std::vector<vp_batch_range> split_batch_ranges(int total_size, int max_batch_size) {
        if (total_size < 0) {
            throw std::invalid_argument("total_size must not be negative");
        }
        if (max_batch_size <= 0) {
            throw std::invalid_argument("max_batch_size must be positive");
        }

        std::vector<vp_batch_range> ranges;
        for (int offset = 0; offset < total_size;) {
            const auto range_size = std::min(max_batch_size, total_size - offset);
            ranges.push_back({offset, range_size});
            offset += range_size;
        }
        return ranges;
    }
}
