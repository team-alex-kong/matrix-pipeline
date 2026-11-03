add_library(matrix_notifier
        matrix_notifier.cpp matrix_notifier.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(matrix_notifier PUBLIC nvjpeg_encoder matrix_sender Boost::headers)
target_link_libraries(matrix_notifier PRIVATE ${OpenCV_LIBS} CUDA::nvjpeg spdlog::spdlog cpr::cpr nlohmann_json::nlohmann_json)


add_library(pipe_writer
        pipe_writer.cpp pipe_writer.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(pipe_writer
        PUBLIC ${OpenCV_LIBS}
        PRIVATE CUDA::nvjpeg spdlog::spdlog cpr::cpr nlohmann_json::nlohmann_json)


add_library(video_writer
        video_writer.cpp video_writer.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(video_writer
        PUBLIC ${OpenCV_LIBS}
        PRIVATE CUDA::nvjpeg spdlog::spdlog cpr::cpr nlohmann_json::nlohmann_json)


add_library(asynchronous_processing_unit
        asynchronous_processing_unit.cpp asynchronous_processing_unit.h
        ../interfaces/i_asynchronous_processing_unit.h
)
target_link_libraries(asynchronous_processing_unit CUDA::nvjpeg
        utils
        overlay_text
        matrix_notifier
        debug_output
        rotate_and_flip
        resize
        collect_stats
        measure_latency
        yolo_detect
        yolo_prune_detection_results
        yolo_overlay
        yunet_detect
        yunet_overlay_landmarks
        cuda_helper
        pipe_writer
        sface_detect
        sface_overlay
        video_writer
        auto_zoom
        yolo_publish_mqtt
)
