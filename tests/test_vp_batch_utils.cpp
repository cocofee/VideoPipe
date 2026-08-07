#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/vp_batch_utils.h"

namespace {
    void require(bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void require_ranges(
        const std::vector<vp_utils::vp_batch_range>& actual,
        const std::vector<vp_utils::vp_batch_range>& expected) {
        require(actual.size() == expected.size(), "unexpected range count");
        for (std::size_t i = 0; i < expected.size(); ++i) {
            require(actual[i].offset == expected[i].offset, "unexpected range offset");
            require(actual[i].size == expected[i].size, "unexpected range size");
        }
    }

    void test_partial_final_batch() {
        require_ranges(
            vp_utils::split_batch_ranges(5, 4),
            {{0, 4}, {4, 1}});
    }

    void test_exact_batches() {
        require_ranges(
            vp_utils::split_batch_ranges(8, 4),
            {{0, 4}, {4, 4}});
    }

    void test_empty_input() {
        require(vp_utils::split_batch_ranges(0, 4).empty(), "empty input should have no ranges");
    }

    void test_invalid_arguments() {
        bool invalid_total_thrown = false;
        try {
            vp_utils::split_batch_ranges(-1, 4);
        }
        catch (const std::invalid_argument&) {
            invalid_total_thrown = true;
        }
        require(invalid_total_thrown, "negative total should be rejected");

        bool invalid_batch_thrown = false;
        try {
            vp_utils::split_batch_ranges(4, 0);
        }
        catch (const std::invalid_argument&) {
            invalid_batch_thrown = true;
        }
        require(invalid_batch_thrown, "non-positive batch size should be rejected");
    }
}

int main() {
    try {
        test_partial_final_batch();
        test_exact_batches();
        test_empty_input();
        test_invalid_arguments();
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    return 0;
}
