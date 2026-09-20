#include "video_feed_manager.h"
#include "global_vars.h"

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudacodec.hpp>
#include <spdlog/spdlog.h>

#include <gsl/gsl>
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
    std::lock_guard lock(mtx_vr);
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

  auto register_delayed_vc_open_retry = [this, ctx]() {
    // .exchange(true) atomically sets the flag to true and returns the PREVIOUS
    // value. If it was already true, someone else is already handling the
    // retry.
    if (delayed_vc_open_retry_registered.exchange(true)) {
      return;
    }
    using namespace std::chrono_literals;
    const auto video_feed_down_for =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() -
            ctx.capture_from_this_device_since);
    const std::chrono::seconds delay_before_attempt =
        std::min(std::max(video_feed_down_for, 2s), 60s * 10);
    // give the video feed a few sec to open without complaining
    if (ctx.frame_seq_num > 90)
      SPDLOG_WARN("captured_from_real_device: {}, device_down_for(sec): {}, "
                  "delay_before_attempt(sec): {}",
                  ctx.captured_from_real_device, video_feed_down_for.count(),
                  delay_before_attempt.count());

    // creating a shared_ptr from *this, s.t. the detach()'ed t will never
    // access *this after *this is deleted.
    auto self_ptr = shared_from_this();
    std::thread t(
        [self_ptr](const std::chrono::seconds delay_before_attempt_,
                   ProcessingUnit::PipelineContext ctx) {
          SPDLOG_INFO(
              "timer started, delay_before_attempt(sec) ({}) and then invoke "
              "cv::cudacodec::createVideoReader({})",
              delay_before_attempt_.count(), ctx.device_info.uri);
          std::this_thread::sleep_for(delay_before_attempt_);
          {
            std::lock_guard lock(self_ptr->mtx_vr);
            SPDLOG_INFO(
                "delay_before_attempt(sec) ({}) reached, about to invoke "
                "cv::cudacodec::createVideoReader({})",
                delay_before_attempt_.count(), ctx.device_info.uri);
            auto params = cv::cudacodec::VideoReaderInitParams();
            // https://docs.opencv.org/4.9.0/dd/d7d/structcv_1_1cudacodec_1_1VideoReaderInitParams.html
            params.allowFrameDrop = true;
            try {
              self_ptr->vr = cv::cudacodec::createVideoReader(
                  ctx.device_info.uri, {}, params);
              if (!self_ptr->vr)
                throw std::runtime_error("!self_ptr->vr");
              self_ptr->vr->set(cv::cudacodec::ColorFormat::BGR);
              SPDLOG_INFO("cv::cudacodec::createVideoReader({}) succeeded",
                          ctx.device_info.uri);
            } catch (const cv::Exception &e) {
              SPDLOG_ERROR("cudacodec::createVideoReader({}) failed: {}",
                           ctx.device_info.uri, e.what());
            }
            std::this_thread::sleep_for(
                100ms); // i.e., at least to skip one frame
            auto reset_flag = gsl::finally([self_ptr] {
              // Use 'release' to ensure all work done in this thread is
              // "committed" and visible to other threads before the flag is
              // reset to false.
              self_ptr->delayed_vc_open_retry_registered.store(
                  false, std::memory_order_release);
            });
          }
        },
        delay_before_attempt, ctx);
    t.detach();
  };

  if (!ctx.captured_from_real_device) [[unlikely]] {
    register_delayed_vc_open_retry();
  }
}
} // namespace MatrixPipeline
