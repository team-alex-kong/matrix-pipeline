#include "video_writer.h"
#include "../utils/misc.h"

#include <spdlog/fmt/chrono.h>
#include <spdlog/spdlog.h>

#include <ctime>
#include <regex>

namespace MatrixPipeline::ProcessingUnit {

VideoWriter::VideoWriter(const std::string &unit_path)
    : IAsynchronousProcessingUnit(unit_path + "/VideoWriter") {}

VideoWriter::~VideoWriter() {
  if (m_writer) {
    m_writer.release();
    SPDLOG_INFO("GPU Video writer released in destructor.");
  }
}

bool VideoWriter::init(const njson &config) {
  try {
    m_config.m_file_path_template =
        config.value("filePath", m_config.m_file_path_template);
    m_config.m_change_rate_threshold =
        config.value("changeRateThreshold", m_config.m_change_rate_threshold);
    m_config.m_cool_off_sec =
        config.value("coolOffSec", m_config.m_cool_off_sec);
    m_config.m_max_length_sec =
        config.value("maxLengthSec", m_config.m_max_length_sec);
    m_config.m_target_fps = config.value("targetFps", m_config.m_target_fps);

    m_config.m_pre_record_frames = config.value("preRecordFrames", 0);

    SPDLOG_INFO("change_rate_threshold: {}, pre_record_frames: {} frames, "
                "cool_off_sec: {}, max_length_sec: {}",
                m_config.m_change_rate_threshold, m_config.m_pre_record_frames,
                m_config.m_cool_off_sec, m_config.m_max_length_sec);
    return true;
  } catch (const njson::exception &e) {
    SPDLOG_ERROR("Failed to parse JSON config: {}", e.what());
    return false;
  }
}

void VideoWriter::on_frame_ready(cv::cuda::GpuMat &frame,
                                 PipelineContext &ctx) {
  if (frame.empty())
    return;

  const double change_rate = ctx.change_rate;
  const auto now = std::chrono::steady_clock::now();

  // --- IDLE STATE ---
  if (m_state == Utils::VideoRecordingState::IDLE) {

    // 1. Maintain Pre-Roll Buffer
    if (m_config.m_pre_record_frames > 0) {
      // Just push 'frame'. GpuMat acts like a shared_ptr/ref-counted object.
      // Gemini 3.1 Pro believe this line is perfectly safe if the frame is NOT
      // re-used, even if it goes out of scope first
      m_pre_roll_buffer.push_back(frame);

      // Remove old frames
      while (m_pre_roll_buffer.size() >
             static_cast<size_t>(m_config.m_pre_record_frames)) {
        m_pre_roll_buffer.pop_front();
      }
    }

    // 2. Check Trigger
    using namespace std::chrono_literals;
    if (change_rate >= m_config.m_change_rate_threshold &&
        ctx.captured_from_real_device &&
        std::chrono::steady_clock::now() - ctx.capture_from_this_device_since >
            10000ms) {
      if (!start_recording(frame.size(), ctx))
        return;

      // 3. FLUSH Pre-Roll Buffer to Writer
      if (!m_pre_roll_buffer.empty()) {
        SPDLOG_INFO("Flushing {} pre-roll frames.", m_pre_roll_buffer.size());
        while (!m_pre_roll_buffer.empty()) {
          write_frame(m_pre_roll_buffer.front());
          m_pre_roll_buffer.pop_front();
        }
      }

      m_state = Utils::VideoRecordingState::RECORDING;
      m_record_start_time = now;
      SPDLOG_INFO("Recording started (change_rate_threshold({}) vs "
                  "change_rate({:.3})), writing video to {}",
                  m_config.m_change_rate_threshold, change_rate, m_video_path);
    }
  }

  // --- RECORDING STATE ---
  if (m_state == Utils::VideoRecordingState::RECORDING) {
    const bool threshold_met = change_rate >= m_config.m_change_rate_threshold;

    if (!threshold_met) {
      if (m_last_below_threshold_time ==
          std::chrono::steady_clock::time_point{}) {
        m_last_below_threshold_time = now;
      }
    } else {
      m_last_below_threshold_time = std::chrono::steady_clock::time_point{};
    }

    const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                 now - m_record_start_time)
                                 .count();
    const auto cooldown_sec = std::chrono::duration_cast<std::chrono::seconds>(
                                  now - m_last_below_threshold_time)
                                  .count();

    if (elapsed_sec >= m_config.m_max_length_sec ||
        (!threshold_met && cooldown_sec >= m_config.m_cool_off_sec)) {
      stop_recording();
      m_state = Utils::VideoRecordingState::IDLE;
      return;
    }

    write_frame(frame);
  }
}

bool VideoWriter::start_recording(const cv::Size frame_size,
                                  const PipelineContext &ctx) {
  if (m_writer) {
    m_writer.release();
  }
  const auto video_path =
      Utils::evaluate_text_template(m_config.m_file_path_template, ctx);
  if (!video_path.has_value()) {
    SPDLOG_ERROR("evaluate_text_template(m_file_path_template) failed");
    return false;
  }

  m_video_path = video_path.value();

  SPDLOG_INFO("cv::cudacodec::createVideoWriter({})ing", m_video_path);
  try {
    cv::cudacodec::EncoderParams params;
    params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
    params.targetQuality = 30;
    m_writer = cv::cudacodec::createVideoWriter(
        m_video_path, frame_size, cv::cudacodec::Codec::HEVC,
        m_config.m_target_fps, cv::cudacodec::ColorFormat::BGR, params);
  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("cv::cudacodec::createVideoWriter({}) failed: {}",
                 m_video_path, e.what());
    if (m_writer)
      m_writer.release();
    SPDLOG_WARN("Disabling videoWriter unit");
    disable();
    return false;
  }
  return m_writer != nullptr;
}

void VideoWriter::stop_recording() {
  if (m_writer) {
    m_writer.release();
    SPDLOG_INFO("Recording stopped, video written to {}", m_video_path);
  }
  m_record_start_time = {};
  m_last_below_threshold_time = {};
  m_pre_roll_buffer.clear();
}

void VideoWriter::write_frame(const cv::cuda::GpuMat &frame) const {
  if (!m_writer)
    return;
  m_writer->write(frame);
}

} // namespace MatrixPipeline::ProcessingUnit