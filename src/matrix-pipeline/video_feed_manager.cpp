#include "video_feed_manager.h"
#include "global_vars.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudacodec.hpp>
#include <spdlog/spdlog.h>

#include <regex>
#include <sys/socket.h>

using namespace std;

namespace MatrixPipeline {

bool VideoFeedManager::init() {
  if (!m_apu.init(settings)) {
    return false;
  }
  m_apu.start();
  return true;
}

void VideoFeedManager::feed_capture_ev() {

  cv::cuda::GpuMat frame;

  ProcessingUnit::PipelineContext ctx;
  try {
    const auto device = settings.at("device");
    ctx.device_info = {.name = device.value("name", "Unnamed Device"),
                       .uri = device.at("uri").get<std::string>(),
                       .expected_frame_size = {
                           device["expectedFrameSize"]["width"].get<int>(),
                           device["expectedFrameSize"]["height"].get<int>()}};
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to parse device info: {}", e.what());
    return;
  }

  ctx.capture_from_this_device_since =
      std::chrono::time_point_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now());
  while (ev_flag == 0) {
    ctx.text_to_overlay = "";
    always_fill_in_frame(frame, ctx);
    handle_video_capture(ctx);
    m_apu.enqueue(frame, ctx);
  }

  vr.release();
  SPDLOG_INFO("thread quits gracefully");
}

void VideoFeedManager::always_fill_in_frame(
    cv::cuda::GpuMat &frame, ProcessingUnit::PipelineContext &ctx) {
  using namespace std::chrono_literals;
  using namespace std::chrono;
  auto captured_from_real_device = false;

  constexpr auto warn_interval = 10s;
  try {
    if (vr == nullptr) {
      if (steady_clock::now() - m_last_warn_time > warn_interval &&
          // give the video feed a few sec to open without complaining
          ctx.frame_seq_num > 90) {
        SPDLOG_WARN("vr == nullptr, frame_seq_num: {} (this "
                    "message is throttled to once per {} sec)",
                    ctx.frame_seq_num, warn_interval.count());
        m_last_warn_time = steady_clock::now();
      }
    } else if (!vr->nextFrame(frame)) {
      if (steady_clock::now() - m_last_warn_time > warn_interval) {
        SPDLOG_ERROR("VideoReader->nextFrame(frame) returns false, "
                     "frame_seq_num: {} (this "
                     "message is throttled to once per {} sec)",
                     ctx.frame_seq_num, warn_interval.count());
        m_last_warn_time = steady_clock::now();
      }
    } else if (frame.empty() ||
               frame.size() != ctx.device_info.expected_frame_size) {

      if (steady_clock::now() - m_last_warn_time > warn_interval) {
        SPDLOG_ERROR("VideoReader->nextFrame((frame) returns frame with "
                     "unexpected size. expect ({}x{}) vs actual ({}x{}) (this "
                     "message is throttled to once per {} sec)",
                     ctx.device_info.expected_frame_size.width,
                     ctx.device_info.expected_frame_size.height,
                     frame.empty() ? -1 : frame.size().width,
                     frame.empty() ? -1 : frame.size().height,
                     warn_interval.count());
        m_last_warn_time = steady_clock::now();
      }
    } else {
      captured_from_real_device = true;
    }
  } catch (const cv::Exception &e) {
    if (steady_clock::now() - m_last_warn_time > warn_interval) {
      SPDLOG_ERROR("VideoReader->nextFrame() failed: {} (this "
                   "message is throttled to once per {} sec)",
                   e.what(), warn_interval.count());
      m_last_warn_time = steady_clock::now();
    }
  }

  if (!ctx.captured_from_real_device) {
    // emulate an 30-fps video device lol
    this_thread::sleep_for(1000ms / 34);
    frame.create(ctx.device_info.expected_frame_size.height,
                 ctx.device_info.expected_frame_size.width, CV_8UC3);
    frame.setTo(cv::Scalar(128, 128, 128));
  }
  ctx.capture_timestamp =
      std::chrono::time_point_cast<milliseconds>(steady_clock::now());
  if (captured_from_real_device != ctx.captured_from_real_device) {
    ctx.capture_from_this_device_since = ctx.capture_timestamp;
  }
  ctx.captured_from_real_device = captured_from_real_device;
  ++ctx.frame_seq_num;
}

void VideoFeedManager::handle_video_capture(
    const ProcessingUnit::PipelineContext &ctx) {
  using namespace std::chrono;
  const auto now = steady_clock::now();
  if (ctx.captured_from_real_device)
    return;
  const auto since = ctx.capture_from_this_device_since;
  const auto down_for = duration_cast<seconds>(now - since);
  // outage length at the previous attempt: doubles per failure, <0 on a new one
  const auto backoff = std::clamp(
      duration_cast<seconds>(m_last_vc_open_attempt - since), 2s, 600s);
  if (now - m_last_vc_open_attempt < backoff)
    return;
  m_last_vc_open_attempt = now;
  // give the video feed a few sec to open without complaining
  if (ctx.frame_seq_num > 90)
    SPDLOG_WARN("device_down_for(sec): {}, invoking "
                "cv::cudacodec::createVideoReader({}), backoff(sec): {}",
                down_for.count(), ctx.device_info.uri, backoff.count());
  auto params = cv::cudacodec::VideoReaderInitParams();
  params.allowFrameDrop = true;
  try {
    vr = cv::cudacodec::createVideoReader(
        ctx.device_info.uri, {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000}, params);
    vr->set(cv::cudacodec::ColorFormat::BGR);
    SPDLOG_INFO("cv::cudacodec::createVideoReader({}) succeeded",
                ctx.device_info.uri);
  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("cudacodec::createVideoReader({}) failed: {}",
                 ctx.device_info.uri, e.what());
  }
}
} // namespace MatrixPipeline
