#pragma once

#include "asynchronous_processing_units/asynchronous_processing_unit.h"
#include "entities/processing_context.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudacodec.hpp>

#include <string>

using njson = nlohmann::json;

namespace MatrixPipeline {

struct videoWritingContext {
  std::string evaluatedVideoPath;
  float fps;
};

class VideoFeedManager {

public:
  VideoFeedManager() = default;
  ~VideoFeedManager() = default;
  bool init();
  void feed_capture_ev();

private:
  ProcessingUnit::AsynchronousProcessingUnit m_apu{""};
  std::string deviceName;

  cv::Ptr<cv::cudacodec::VideoReader> vr{nullptr};
  std::chrono::steady_clock::time_point m_last_vc_open_attempt;
  void always_fill_in_frame(cv::cuda::GpuMat &frame,
                            ProcessingUnit::PipelineContext &ctx);
  void handle_video_capture(const ProcessingUnit::PipelineContext &ctx);
  std::chrono::time_point<std::chrono::steady_clock> m_last_warn_time;
};
} // namespace MatrixPipeline
