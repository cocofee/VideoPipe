add_executable(vp_cmake_smoke cmake_smoke.cpp)
add_test(NAME vp_cmake_smoke COMMAND vp_cmake_smoke)

add_executable(vp_stream_profile_test test_stream_profile.cpp)
target_link_libraries(vp_stream_profile_test PRIVATE video_pipe)
add_test(NAME vp_stream_profile_test COMMAND vp_stream_profile_test)

add_executable(
    vp_cycling_upgrade_options_test
    test_cycling_upgrade_options.cpp
    ../samples/cycling_upgrade_options.cpp)
add_test(NAME vp_cycling_upgrade_options_test COMMAND vp_cycling_upgrade_options_test)

add_executable(
    vp_bytetrack_test
    test_bytetrack.cpp
    ../nodes/track/vp_bytetrack_node.cpp
    ../nodes/track/sort/Hungarian.cpp
    ../objects/shapes/vp_rect.cpp
    ../objects/shapes/vp_point.cpp
    ../objects/shapes/vp_size.cpp)
target_compile_definitions(vp_bytetrack_test PRIVATE VP_BYTETRACK_TEST_ONLY)
target_link_libraries(vp_bytetrack_test PRIVATE ${OpenCV_LIBS})
add_test(NAME vp_bytetrack_test COMMAND vp_bytetrack_test)

if(VP_WITH_YOLO11)
    add_executable(
        vp_yolo11_postprocess_test
        test_yolo11_postprocess.cpp
        ../nodes/infers/vp_yolo11_detector_node.cpp
        ../objects/vp_frame_target.cpp
        ../objects/shapes/vp_rect.cpp
        ../objects/shapes/vp_point.cpp
        ../objects/shapes/vp_size.cpp)
    target_compile_definitions(vp_yolo11_postprocess_test PRIVATE VP_YOLO11_POSTPROCESS_ONLY)
    target_link_libraries(vp_yolo11_postprocess_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME vp_yolo11_postprocess_test COMMAND vp_yolo11_postprocess_test)
endif()

if(VP_WITH_ASYNC_OCR)
    add_executable(
        vp_ocr_dispatch_test
        test_ocr_dispatch.cpp
        ../nodes/infers/vp_async_ocr_node.cpp
        ../objects/vp_ocr_job.cpp
        ../objects/vp_ocr_consensus.cpp
        ../objects/vp_frame_meta.cpp
        ../objects/vp_frame_target.cpp
        ../objects/vp_frame_pose_target.cpp
        ../objects/vp_frame_face_target.cpp
        ../objects/vp_frame_text_target.cpp
        ../objects/vp_sub_target.cpp
        ../objects/vp_meta.cpp
        ../objects/ba/vp_ba_result.cpp
        ../objects/shapes/vp_rect.cpp
        ../objects/shapes/vp_point.cpp
        ../objects/shapes/vp_size.cpp)
    target_compile_definitions(vp_ocr_dispatch_test PRIVATE VP_ASYNC_OCR_CORE_ONLY)
    target_link_libraries(vp_ocr_dispatch_test PRIVATE ${OpenCV_LIBS})
    add_test(NAME vp_ocr_dispatch_test COMMAND vp_ocr_dispatch_test)
endif()
