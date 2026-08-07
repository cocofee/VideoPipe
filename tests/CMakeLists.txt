add_executable(vp_batch_utils_test
    test_vp_batch_utils.cpp
)

target_include_directories(vp_batch_utils_test PRIVATE ${PROJECT_SOURCE_DIR})

add_test(NAME vp_batch_utils_test COMMAND vp_batch_utils_test)

find_package(Threads REQUIRED)

set(VP_TEST_SUPPORT_SOURCES
    ${PROJECT_SOURCE_DIR}/nodes/vp_meta_publisher.cpp
    ${PROJECT_SOURCE_DIR}/nodes/vp_meta_subscriber.cpp
    ${OBJECTS}
    ${PROJECT_SOURCE_DIR}/utils/logger/vp_logger.cpp
    ${PROJECT_SOURCE_DIR}/utils/logger/vp_log_file_writer.cpp
)

add_executable(vp_node_ordering_test
    test_vp_node_ordering.cpp
    ${PROJECT_SOURCE_DIR}/nodes/vp_node.cpp
    ${PROJECT_SOURCE_DIR}/nodes/vp_src_node.cpp
    ${VP_TEST_SUPPORT_SOURCES}
)

target_include_directories(vp_node_ordering_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(vp_node_ordering_test ${OpenCV_LIBS} Threads::Threads stdc++fs)

add_test(NAME vp_node_ordering_test COMMAND vp_node_ordering_test)
set_tests_properties(vp_node_ordering_test PROPERTIES TIMEOUT 5)

add_executable(vp_infer_batch_test
    test_vp_infer_batch.cpp
    opencv_transpose_compat.cpp
    ${PROJECT_SOURCE_DIR}/nodes/vp_node.cpp
    ${PROJECT_SOURCE_DIR}/nodes/vp_infer_node.cpp
    ${VP_TEST_SUPPORT_SOURCES}
)

target_include_directories(vp_infer_batch_test PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(vp_infer_batch_test ${OpenCV_LIBS} Threads::Threads stdc++fs)

if(OpenCV_VERSION VERSION_LESS "4.6")
    target_compile_options(vp_infer_batch_test PRIVATE
        -include ${CMAKE_CURRENT_SOURCE_DIR}/opencv_transpose_compat.h
    )
endif()

add_test(NAME vp_infer_batch_test COMMAND vp_infer_batch_test)
