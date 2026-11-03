add_library(utils
        misc.cpp misc.h
)
target_link_libraries(utils
        PUBLIC Drogon::Drogon fmt::fmt nlohmann_json::nlohmann_json spdlog::spdlog)

add_library(nvjpeg_encoder
        nvjpeg_encoder.cpp
        nvjpeg_encoder.h
)
target_link_libraries(nvjpeg_encoder
        PUBLIC
        CUDA::cudart
        CUDA::nvjpeg
        PRIVATE
        spdlog::spdlog)


add_library(matrix_sender
        matrix_sender.cpp
        matrix_sender.h
)
target_link_libraries(matrix_sender cpr::cpr spdlog::spdlog)


add_library(cuda_helper
        cuda_helper.cpp
        cuda_helper.h
)
target_include_directories(cuda_helper
        # the SYSTEM keyword can be used to silence warnings from system headers
        SYSTEM PUBLIC "$ENV{TRT_HOME}/include"
)
target_link_libraries(cuda_helper
        PUBLIC ${NVINFER_LIB} ${NVONNX_LIB} CUDA::cudart
        PRIVATE spdlog::spdlog)