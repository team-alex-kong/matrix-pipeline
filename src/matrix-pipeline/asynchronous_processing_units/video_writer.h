#pragma once

#include "../entities/video_recording_state.h"
#include "../interfaces/i_asynchronous_processing_unit.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/videoio.hpp> // Needed for cv::VideoWriter::fourcc

#include <chrono>
#include <deque>
#include <string>

namespace MatrixPipeline::ProcessingUnit {

struct VideoWriterConfig {
  std::string m_file_path_template;
  double m_change_rate_threshold = 0.0;
  int m_cool_off_sec = 30;
  int m_max_length_sec = 60;
  double m_target_fps = 30.0;

  // N frames to keep
  int m_pre_record_frames = 30 * 5;
};

class VideoWriter final : public IAsynchronousProcessingUnit {
public:
  explicit VideoWriter(const std::string &unit_path);
  ~VideoWriter() override;

  bool init(const njson &config) override;

protected:
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;

private:
  bool start_recording(cv::Size frame_size, const PipelineContext &ctx);
  void stop_recording();
  void write_frame(const cv::cuda::GpuMat &frame) const;

  Utils::VideoRecordingState m_state = Utils::VideoRecordingState::IDLE;
  VideoWriterConfig m_config;
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer;

  // Stores GpuMat references.
  std::deque<cv::cuda::GpuMat> m_pre_roll_buffer;

  std::chrono::steady_clock::time_point m_record_start_time;
  std::chrono::steady_clock::time_point m_last_below_threshold_time;
  std::string m_video_path;
};

} // namespace MatrixPipeline::ProcessingUnit