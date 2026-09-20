#include "matrix_notifier.h"
#include "../global_vars.h"

#include <boost/uuid/uuid_io.hpp>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <gsl/gsl>
#include <nlohmann/json.hpp>

#include <absl/strings/internal/resize_uninitialized.h>
#include <chrono>
#include <string>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

RoiLookupResult MatrixNotifier::look_for_roi(const PipelineContext &ctx) const {
  // The order cant be switched, we need to check sface roi first, as it can
  // terminate the rest by returning Suppress;
  if (m_enable_sface_roi) {
    for (const auto &[detection, recognition] : ctx.yunet_sface.results) {
      if (!recognition.l2_norm_threshold_crossed ||
          !recognition.cosine_score_threshold_crossed)
        continue;

      if (m_sface_mark_authorized_identity_as_not_interesting &&
          recognition.category == IdentityCategory::Authorized)
        return Suppress;
      return Found;
    }
  }

  if (m_enable_yolo_roi) {
    if (std::ranges::any_of(ctx.yolo.indices, [&](auto idx) {
          return ctx.yolo.is_detection_interesting[idx];
        })) {
      return Found;
    }
  }
  // SPDLOG_INFO("NotFound");
  return NotFound;
}

// Post-step on purpose: the recording on disk must stay complete (see write)
std::optional<std::string>
MatrixNotifier::trim_video(const std::string &input_video_path,
                           const std::string &trimmed_video_path,
                           const int frames_to_remove) const {
  const auto remove_trimmed = gsl::finally([&trimmed_video_path] {
    std::error_code ec;
    std::filesystem::remove(trimmed_video_path, ec);
  });
  try {
    // 2. Probe Metadata (Standard VideoCapture for header info)
    cv::VideoCapture cap(input_video_path);
    if (!cap.isOpened()) {
      SPDLOG_ERROR("!cap.isOpened()");
      return std::nullopt;
    }

    const auto total_frames =
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    const cv::Size frame_size(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    cap.release();

    const int frames_after_trim = total_frames - frames_to_remove;
    if (frames_after_trim < 0) {
      SPDLOG_ERROR(
          "total_frames: {}, frames_to_remove: {}, frames_after_trim: {} < 0",
          total_frames, frames_to_remove, frames_after_trim);
      return std::nullopt;
    }

    cv::Ptr<cv::cudacodec::VideoReader> reader =
        cv::cudacodec::createVideoReader(input_video_path);
    reader->set(cv::cudacodec::ColorFormat::BGR);

    cv::cudacodec::EncoderParams params;
    // 2. Set Rate Control to Variable Bitrate
    params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
    params.targetQuality = m_target_quality;
    const auto writer = cv::cudacodec::createVideoWriter(
        trimmed_video_path, frame_size, cv::cudacodec::Codec::HEVC, m_fps,
        cv::cudacodec::ColorFormat::BGR, params);

    cv::cuda::GpuMat d_frame;
    int current_frame = 0;

    while (current_frame < frames_after_trim && reader->nextFrame(d_frame)) {
      if (d_frame.empty()) {
        SPDLOG_ERROR("!d_frame.empty()");
        break;
      }
      writer->write(d_frame);
      current_frame++;
    }

    writer->release();

    std::error_code ec;
    auto size = std::filesystem::file_size(trimmed_video_path, ec);
    std::ifstream file(trimmed_video_path, std::ios::binary);
    if (ec || !file) {
      SPDLOG_ERROR("IO error");
      return std::nullopt;
    }
    std::string buffer(size, '\0'); // 1. Pre-allocate exact size
    if (!file.read(buffer.data(), size)) {
      SPDLOG_ERROR("IO error");
      return std::nullopt;
    }
    return buffer;

  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what(): {}", e.what());
    return std::nullopt;
  }
}

void MatrixNotifier::finalize_video_then_send_out(
    const std::string temp_video_path, const std::string trimmed_video_path,
    const std::string jpeg_data, const cv::Size thumbnail_size,
    const size_t frame_count) const {
  using namespace std::chrono;

  const auto video_duration_ms =
      static_cast<long>(frame_count * 1000.0 / m_fps);
  const auto send_video_start_at = steady_clock::now();
  int frames_to_remove = m_roi_gap_tolerance_frames -
                         m_video_postcapture_frames - m_video_precapture_frames;
  if (frames_to_remove < 0)
    frames_to_remove = 0;
  if (const auto video =
          trim_video(temp_video_path, trimmed_video_path, frames_to_remove);
      video.has_value()) {
    m_sender->send_video_from_memory(
        video.value(),
        fmt::format("{:%Y-%m-%dT%H:%M:%S}.mp4", system_clock::now()),
        video_duration_ms,
        fmt::format("{:%Y-%m-%dT%H:%M:%S}", system_clock::now()), jpeg_data,
        thumbnail_size.width, thumbnail_size.height);
    const auto send_video_ms =
        duration_cast<milliseconds>(steady_clock::now() - send_video_start_at)
            .count();
    SPDLOG_INFO("video size: {}KB + thumbnail size {}KB, video_length(sec): "
                "{:.1f}, send_video_from_memory() took {}ms ({:.0f}KB/sec)",
                video.value().size() / 1024, jpeg_data.size() / 1024,
                video_duration_ms / 1000.0, send_video_ms,
                (video.value().size() + jpeg_data.size()) / 1024.0 /
                    std::max<long>(send_video_ms, 1) * 1000);
  } else {
    SPDLOG_ERROR("trim_video() failed");
  }
  try {
    if (!std::filesystem::remove(temp_video_path)) {
      SPDLOG_ERROR("std::filesystem::remove({}) returns false",
                   temp_video_path);
    }
  } catch (const std::filesystem::filesystem_error &e) {
    SPDLOG_ERROR("std::filesystem::remove({}) exception, e.what(): {}",
                 temp_video_path, e.what());
  }
}

void MatrixNotifier::consume_pending_sends(const bool wait) {
  std::erase_if(m_pending_sends, [wait](auto &f) {
    if (!wait && f.wait_for(0s) != std::future_status::ready)
      return false;
    try {
      f.get();
    } catch (const std::exception &e) {
      SPDLOG_ERROR("video send failed: {}", e.what());
    } catch (...) {
      SPDLOG_ERROR("video send failed");
    }
    return true;
  });
}

void MatrixNotifier::handle_video(const cv::cuda::GpuMat &frame,
                                  [[maybe_unused]] const PipelineContext &ctx,
                                  const RoiLookupResult roi_flag) {
  using namespace std::chrono;

  m_frames_queue.push({frame, ctx});
  while (m_frames_queue.size() >
         static_cast<size_t>(m_video_precapture_frames)) {
    m_frames_queue.pop();
  }

  // is_frame_changing and roi_flag rely on the current frame
  // const auto is_frame_changing = ctx.change_rate >=
  // m_activation_min_frame_change_rate;
  if (m_state == Utils::VideoRecordingState::IDLE &&
      (roi_flag != Found ||
       ctx.change_rate < m_activation_min_frame_change_rate))
    return;

  if (m_state == Utils::VideoRecordingState::IDLE) {
    try {
      // m_ram_buf = std::make_unique<Utils::RamVideoBuffer>();
      // cv::cudacodec::createVideoWriter expects a .mp4 file path, we must
      // emulate this behavior...
      m_temp_video_path =
          fmt::format("/tmp/nvenc_buffer_{}.mp4",
                      boost::uuids::to_string(m_uuid_generator()));

      cv::cudacodec::EncoderParams params;
      // 2. Set Rate Control to Variable Bitrate
      params.rateControlMode = cv::cudacodec::ENC_PARAMS_RC_VBR;
      params.targetQuality = m_target_quality;
      m_writer = cv::cudacodec::createVideoWriter(
          m_temp_video_path, frame.size(), cv::cudacodec::Codec::HEVC, m_fps,
          cv::cudacodec::ColorFormat::BGR, params);

      m_current_video_start_at = m_frames_queue.front().ctx.capture_timestamp;
      m_current_video_without_detection_frames = 0;
      m_current_video_frame_count = 0;
      m_current_video_should_be_suppressed = false;
      m_max_roi_score = -1;
      SPDLOG_INFO("Start video recording for matrix message, saved file to "
                  "temp_video_path {}, video_max_length(sec): {}",
                  m_temp_video_path, m_video_max_length);
      m_state = Utils::VideoRecordingState::RECORDING;
    } catch (const cv::Exception &e) {
      SPDLOG_ERROR("cv::cudacodec::VideoWriter({}) failed: {}",
                   m_temp_video_path, e.what());
      if (m_writer)
        m_writer.release();
      m_writer = nullptr;
      disable();
      unlink(m_temp_video_path.c_str());
      SPDLOG_WARN("{} turned off", m_unit_path);
      return;
    }
  }

  const auto is_max_length_reached_or_program_exiting =
      steady_clock::now() - m_current_video_start_at >= m_video_max_length ||
      ev_flag != 0;
  const auto is_roi_gap_tolerance_reached =
      m_current_video_without_detection_frames++ >= m_roi_gap_tolerance_frames;
  // We have to store the suppression flag, instead of terminating the video
  // immediately, as we want to suppress the entire video, not just until now
  m_current_video_should_be_suppressed |= (roi_flag == Suppress);
  if (is_max_length_reached_or_program_exiting ||
      is_roi_gap_tolerance_reached) {

    m_writer.release();
    SPDLOG_INFO("video stopped, "
                "is_max_length_reached_or_program_exiting: {}, "
                "is_roi_gap_tolerance_reached: {}, "
                "m_current_video_should_be_suppressed: {}",
                is_max_length_reached_or_program_exiting,
                is_roi_gap_tolerance_reached,
                m_current_video_should_be_suppressed);
    m_state = Utils::VideoRecordingState::IDLE;
    if (!m_current_video_should_be_suppressed &&
        m_current_video_frame_count > 0) {
      std::string jpeg_data;
      if (!m_gpu_encoder->encode(m_max_roi_score_frame, jpeg_data, 90))
        SPDLOG_ERROR("m_gpu_encoder->encode() failed");
      consume_pending_sends(false);
      m_pending_sends.push_back(std::async(
          std::launch::async, &MatrixNotifier::finalize_video_then_send_out,
          this, m_temp_video_path,
          fmt::format("/tmp/nvenc_buffer_{}.mp4",
                      boost::uuids::to_string(m_uuid_generator())),
          std::move(jpeg_data), m_max_roi_score_frame.size(),
          m_current_video_frame_count));
    } else {
      try {
        if (!std::filesystem::remove(m_temp_video_path)) {
          SPDLOG_ERROR("std::filesystem::remove({}) returns false",
                       m_temp_video_path);
        } else {
          SPDLOG_INFO("std::filesystem::remove({})'ed, video not sent "
                      "(suppressed: {}, frames: {})",
                      m_temp_video_path, m_current_video_should_be_suppressed,
                      m_current_video_frame_count);
        }
      } catch (const std::filesystem::filesystem_error &e) {
        SPDLOG_ERROR("std::filesystem::remove({}) exception, e.what(): {}",
                     m_temp_video_path, e.what());
      }
    }
    return;
  }
  if (const auto roi_score = calculate_roi_score(m_frames_queue.front().ctx);
      roi_score >= m_max_roi_score /* must be >=, not >*/) {
    m_max_roi_score_frame = m_frames_queue.front().frame;
    m_max_roi_score = roi_score;
  }

  // Always write the tail: it is trimmed only on the uploaded copy so that
  // consecutive recordings stay concatenable without a gap.
  m_writer->write(m_frames_queue.front().frame);
  m_frames_queue.pop();
  ++m_current_video_frame_count;
  if (roi_flag == Found &&
      ctx.change_rate > m_maintenance_min_frame_change_rate)
    m_current_video_without_detection_frames = 0;
}

double MatrixNotifier::calculate_roi_score(const PipelineContext &ctx) const {
  const auto &yolo = ctx.yolo;
  double roi_value = 0.0;

  if (m_enable_yolo_roi) {
    for (const auto idx : yolo.indices) {

      if (yolo.class_ids[idx] == 0 && yolo.is_detection_interesting[idx]) {
        const auto normalized_area =
            static_cast<double>(yolo.bounding_boxes[idx].area()) /
            yolo.inference_input_size.area();
        roi_value += normalized_area * yolo.confidences[idx] *
                     pow(yolo.indices.size(), 0.5);
      }
    }
  }

  if (m_enable_sface_roi) {
    for (const auto &[detection, recognition] : ctx.yunet_sface.results) {

      const auto normalized_area =
          detection.bounding_box.area() /
          static_cast<float>(ctx.yunet_sface.yunet_input_frame_size.area());
      if (std::isnan(recognition.cosine_score))
        continue;

      roi_value += normalized_area * recognition.cosine_score *
                   m_identity_to_weight_map.at(recognition.category) *
                   m_detection_recognition_weight_ratio *
                   pow(ctx.yunet_sface.results.size(), 0.5);
    }
  }

  return roi_value;
}

bool MatrixNotifier::init(const njson &config) {
  if (!config.contains("matrixHomeServer") ||
      !config.contains("matrixRoomId") ||
      !config.contains("matrixAccessToken")) {
    SPDLOG_ERROR("Missing matrix credentials");
    return false;
  }
  m_matrix_homeserver = config.at("matrixHomeServer");
  m_matrix_room_id = config.at("matrixRoomId");
  m_matrix_access_token = config.at("matrixAccessToken");

  m_video_max_length = std::chrono::seconds(
      config.value("videoMaxLengthInSeconds", m_video_max_length.count()));
  m_target_quality = config.value("videoTargetQuality", m_target_quality);
  m_detection_recognition_weight_ratio = config.value(
      "detectionRecognitionWeightRatio", m_detection_recognition_weight_ratio);
  m_activation_min_frame_change_rate = config.value(
      "activationMinFrameChangeRate", m_activation_min_frame_change_rate);
  m_maintenance_min_frame_change_rate = config.value(
      "maintenanceMinFrameChangeRate", m_maintenance_min_frame_change_rate);
  m_fps = config.value("fps", m_fps);
  m_roi_gap_tolerance_frames =
      config.value("roiGapToleranceFrames", m_roi_gap_tolerance_frames);
  m_video_precapture_frames =
      config.value("videoPrecaptureFrames", m_video_precapture_frames);
  m_video_postcapture_frames =
      config.value("videoPostcaptureFrames", m_video_postcapture_frames);
  m_enable_yolo_roi =
      config.value("/roi/enableYoloRoi"_json_pointer, m_enable_yolo_roi);
  m_enable_sface_roi =
      config.value("/roi/enableSFaceRoi"_json_pointer, m_enable_sface_roi);
  m_sface_mark_authorized_identity_as_not_interesting = config.value(
      "/roi/sFaceMarkAuthorizedIdentityAsNotInteresting"_json_pointer,
      m_sface_mark_authorized_identity_as_not_interesting);
  SPDLOG_INFO("matrix_homeserver: {}, matrix_room_id: {}, matrix_access_token: "
              "{}",
              m_matrix_homeserver, m_matrix_room_id, m_matrix_access_token);

  SPDLOG_INFO(
      "video_max_length(sec): {}, m_roi_gap_tolerance_frames: {}, "
      "video_precapture_frames: {}, video_postcapture_frames: {}, fps: {}, "
      "activation_min_frame_change_rate: {}, "
      "maintenance_min_frame_change_rate: {}, target_quality: {} (0-51, "
      "lower is better)",
      m_video_max_length, m_roi_gap_tolerance_frames, m_video_precapture_frames,
      m_video_postcapture_frames, m_fps, m_activation_min_frame_change_rate,
      m_maintenance_min_frame_change_rate, m_target_quality);
  SPDLOG_INFO("enable_yolo_roi : {}, enable_sface_roi : {}, "
              "sface_mark_authorized_identity_as_not_interesting: {}",
              m_enable_yolo_roi, m_enable_sface_roi,
              m_sface_mark_authorized_identity_as_not_interesting);
  m_sender = std::make_unique<Utils::MatrixSender>(
      m_matrix_homeserver, m_matrix_access_token, m_matrix_room_id);
  m_gpu_encoder = std::make_unique<Utils::NvJpegEncoder>();
  if (config.value("testMatrixConnectivity", false))
    m_sender->sendText("MatrixPipeline started");
  return true;
}

void MatrixNotifier::on_frame_ready(cv::cuda::GpuMat &frame,
                                    [[maybe_unused]] PipelineContext &ctx) {

  handle_video(frame, ctx, look_for_roi(ctx));
}

} // namespace MatrixPipeline::ProcessingUnit