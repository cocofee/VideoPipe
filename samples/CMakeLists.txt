# can ONLY be called by parent CMakeLists.txt, because no environment settings for this script.#
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "can ONLY be called by parent CMakeLists.txt.")
endif()

message("start build for simple samples...")
# save all exe to 'build/bin'
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(1-1-1_sample "1-1-1_sample.cpp")
target_link_libraries(1-1-1_sample ${PROJECT_NAME})

add_executable(1-1-N_sample "1-1-N_sample.cpp")
target_link_libraries(1-1-N_sample ${PROJECT_NAME})

add_executable(1-N-N_sample "1-N-N_sample.cpp")
target_link_libraries(1-N-N_sample ${PROJECT_NAME})

add_executable(ba_crossline_sample "ba_crossline_sample.cpp")
target_link_libraries(ba_crossline_sample ${PROJECT_NAME})

add_executable(enet_seg_sample "enet_seg_sample.cpp")
target_link_libraries(enet_seg_sample ${PROJECT_NAME})

add_executable(face_tracking_sample "face_tracking_sample.cpp")
target_link_libraries(face_tracking_sample ${PROJECT_NAME})

add_executable(image_des_sample "image_des_sample.cpp")
target_link_libraries(image_des_sample ${PROJECT_NAME})

add_executable(image_src_sample "image_src_sample.cpp")
target_link_libraries(image_src_sample ${PROJECT_NAME})

add_executable(interaction_with_pipe_sample "interaction_with_pipe_sample.cpp")
target_link_libraries(interaction_with_pipe_sample ${PROJECT_NAME})

add_executable(mask_rcnn_sample "mask_rcnn_sample.cpp")
target_link_libraries(mask_rcnn_sample ${PROJECT_NAME})

add_executable(message_broker_sample "message_broker_sample.cpp")
target_link_libraries(message_broker_sample ${PROJECT_NAME})

add_executable(multi_detectors_and_classifiers_sample "multi_detectors_and_classifiers_sample.cpp")
target_link_libraries(multi_detectors_and_classifiers_sample ${PROJECT_NAME})

add_executable(N-1-N_sample "N-1-N_sample.cpp")
target_link_libraries(N-1-N_sample ${PROJECT_NAME})

add_executable(N-N_sample "N-N_sample.cpp")
target_link_libraries(N-N_sample ${PROJECT_NAME})

add_executable(openpose_sample "openpose_sample.cpp")
target_link_libraries(openpose_sample ${PROJECT_NAME})

add_executable(record_sample "record_sample.cpp")
target_link_libraries(record_sample ${PROJECT_NAME})

add_executable(rtsp_des_sample "rtsp_des_sample.cpp")
target_link_libraries(rtsp_des_sample ${PROJECT_NAME})

add_executable(rtsp_src_sample "rtsp_src_sample.cpp")
target_link_libraries(rtsp_src_sample ${PROJECT_NAME})

add_executable(vp_logger_sample "vp_logger_sample.cpp")
target_link_libraries(vp_logger_sample ${PROJECT_NAME})

add_executable(skip_sample "skip_sample.cpp")
target_link_libraries(skip_sample ${PROJECT_NAME})

add_executable(obstacle_detect_sample "obstacle_detect_sample.cpp")
target_link_libraries(obstacle_detect_sample ${PROJECT_NAME})

add_executable(multi_detectors_sample "multi_detectors_sample.cpp")
target_link_libraries(multi_detectors_sample ${PROJECT_NAME})

add_executable(firesmoke_detect_sample "firesmoke_detect_sample.cpp")
target_link_libraries(firesmoke_detect_sample ${PROJECT_NAME})

add_executable(face_swap_sample "face_swap_sample.cpp")
target_link_libraries(face_swap_sample ${PROJECT_NAME})

add_executable(video_restoration_sample "video_restoration_sample.cpp")
target_link_libraries(video_restoration_sample ${PROJECT_NAME})

add_executable(app_des_sample "app_des_sample.cpp")
target_link_libraries(app_des_sample ${PROJECT_NAME})

add_executable(app_src_des_sample "app_src_des_sample.cpp")
target_link_libraries(app_src_des_sample ${PROJECT_NAME})

add_executable(lane_detect_sample "lane_detect_sample.cpp")
target_link_libraries(lane_detect_sample ${PROJECT_NAME})

add_executable(frame_fusion_sample "frame_fusion_sample.cpp")
target_link_libraries(frame_fusion_sample ${PROJECT_NAME})

# samples depend on Kafka
if(VP_WITH_KAFKA)
    add_executable(message_broker_kafka_sample "message_broker_kafka_sample.cpp")
    target_link_libraries(message_broker_kafka_sample ${PROJECT_NAME})    
endif()

# samples depend on LLM
if(VP_WITH_LLM)
    add_executable(mllm_analyse_sample "mllm_analyse_sample.cpp")
    target_link_libraries(mllm_analyse_sample ${PROJECT_NAME})    

    add_executable(mllm_analyse_sample_openai "mllm_analyse_sample_openai.cpp")
    target_link_libraries(mllm_analyse_sample_openai ${PROJECT_NAME})    
endif()

# samples depend on FFmpeg
if(VP_WITH_FFMPEG)
    add_executable(ffmpeg_src_des_sample "ffmpeg_src_des_sample.cpp")
    target_link_libraries(ffmpeg_src_des_sample ${PROJECT_NAME})

    add_executable(ffmpeg_transcode_sample "ffmpeg_transcode_sample.cpp")
    target_link_libraries(ffmpeg_transcode_sample ${PROJECT_NAME})
endif()

# samples depend on PaddlePaddle
if(VP_WITH_PADDLE)
    add_executable(app_src_sample "app_src_sample.cpp")
    target_link_libraries(app_src_sample ${PROJECT_NAME})
    
    add_executable(paddle_infer_sample "paddle_infer_sample.cpp")
    target_link_libraries(paddle_infer_sample ${PROJECT_NAME})
endif()

# samples depend on CUDA(maybe other dependency required too)
if(VP_WITH_CUDA)
    # nvdec & nvenc plugins required from DeepStream
    add_executable(nv_hard_codec_sample "nv_hard_codec_sample.cpp")
    target_link_libraries(nv_hard_codec_sample ${PROJECT_NAME})    
endif()

# samples depend on TensorRT
if(VP_WITH_TRT)
    add_executable(ba_jam_sample "ba_jam_sample.cpp")
    target_link_libraries(ba_jam_sample ${PROJECT_NAME})

    add_executable(ba_stop_sample "ba_stop_sample.cpp")
    target_link_libraries(ba_stop_sample ${PROJECT_NAME})

    add_executable(body_scan_and_plate_detect_sample "body_scan_and_plate_detect_sample.cpp")
    target_link_libraries(body_scan_and_plate_detect_sample ${PROJECT_NAME})

    add_executable(dynamic_pipeline_sample "dynamic_pipeline_sample.cpp")
    target_link_libraries(dynamic_pipeline_sample ${PROJECT_NAME})

    add_executable(dynamic_pipeline_sample2 "dynamic_pipeline_sample2.cpp")
    target_link_libraries(dynamic_pipeline_sample2 ${PROJECT_NAME})    

    add_executable(message_broker_sample2 "message_broker_sample2.cpp")
    target_link_libraries(message_broker_sample2 ${PROJECT_NAME})

    add_executable(multi_trt_infer_nodes_sample "multi_trt_infer_nodes_sample.cpp")
    target_link_libraries(multi_trt_infer_nodes_sample ${PROJECT_NAME})

    add_executable(plate_recognize_sample "plate_recognize_sample.cpp")
    target_link_libraries(plate_recognize_sample ${PROJECT_NAME})

    add_executable(src_des_sample "src_des_sample.cpp")
    target_link_libraries(src_des_sample ${PROJECT_NAME})

    add_executable(trt_infer_sample "trt_infer_sample.cpp")
    target_link_libraries(trt_infer_sample ${PROJECT_NAME})

    add_executable(vehicle_body_scan_sample "vehicle_body_scan_sample.cpp")
    target_link_libraries(vehicle_body_scan_sample ${PROJECT_NAME})

    add_executable(vehicle_cluster_based_on_classify_encoding_sample "vehicle_cluster_based_on_classify_encoding_sample.cpp")
    target_link_libraries(vehicle_cluster_based_on_classify_encoding_sample ${PROJECT_NAME})

    add_executable(vehicle_tracking_sample "vehicle_tracking_sample.cpp")
    target_link_libraries(vehicle_tracking_sample ${PROJECT_NAME})

    add_executable(1-N-1_sample "1-N-1_sample.cpp")
    target_link_libraries(1-N-1_sample ${PROJECT_NAME})

    add_executable(1-N-1_sample2 "1-N-1_sample2.cpp")
    target_link_libraries(1-N-1_sample2 ${PROJECT_NAME})

    add_executable(1-N-1_sample3 "1-N-1_sample3.cpp")
    target_link_libraries(1-N-1_sample3 ${PROJECT_NAME})

    add_executable(trt_yolov8_sample "trt_yolov8_sample.cpp")
    target_link_libraries(trt_yolov8_sample ${PROJECT_NAME})

    add_executable(trt_yolov8_sample2 "trt_yolov8_sample2.cpp")
    target_link_libraries(trt_yolov8_sample2 ${PROJECT_NAME})

    add_executable(rtmp_src_sample "rtmp_src_sample.cpp")
    target_link_libraries(rtmp_src_sample ${PROJECT_NAME})
endif()
