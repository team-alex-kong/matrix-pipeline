#pragma once

#include "../entities/video_recording_state.h"
#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../utils/nvjpeg_encoder.h"
#include "../utils/ram_video_buffer.h"

#include <boost/uuid/uuid.hpp> // main uuid family
#include <boost/uuid/uuid_generators.hpp>
#include <opencv2/cudacodec.hpp>

#include <future>

namespace MatrixPipeline::Utils {
struct RamVideoBuffer;
} // namespace MatrixPipeline::Utils
namespace MatrixPipeline::ProcessingUnit {
enum RoiLookupResult {
  Found,
  NotFound,
  Suppress // Find an ROI that is supposed to suppress the entire video
};
using namespace std::chrono_literals;
class MatrixNotifier final : public IAsynchronousProcessingUnit {
  boost::uuids::random_generator m_uuid_generator;
  std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder{nullptr};
  std::unique_ptr<Utils::MatrixSender> m_sender{nullptr};
  std::string m_matrix_homeserver;
  std::string m_matrix_room_id;
  std::string m_matrix_access_token;
  bool m_enable_yolo_roi{true};
  bool m_enable_sface_roi{true};
  bool m_sface_mark_authorized_identity_as_not_interesting{false};
  double m_activation_min_frame_change_rate{0.1};
  double m_maintenance_min_frame_change_rate{0.01};
  double m_detection_recognition_weight_ratio{3.14};
  std::chrono::seconds m_video_max_length{60s};
  static inline std::unordered_map<IdentityCategory, float>
      m_identity_to_weight_map = {{IdentityCategory::Unknown, 1.0},
                                  {IdentityCategory::Unauthorized, 1.73},
                                  {IdentityCategory::Authorized, 3.14}};
  // using second as unit is more user-friendly but impose significant
  // difficulty on implementation. The problem is that there are two FPSes, one
  // from the video feed, which is dynamic; the other from videoWriter, which is
  // static. If we want to trim the back of the final video, two FPSes cause
  // confusion
  int m_video_precapture_frames{45};
  int m_roi_gap_tolerance_frames{120};
  int m_video_postcapture_frames{45};

  std::chrono::time_point<std::chrono::steady_clock> m_current_video_start_at;
  size_t m_current_video_frame_count{0};
  bool m_current_video_should_be_suppressed{false};
  int m_current_video_without_detection_frames{-1};
  double m_max_roi_score{0.0};
  cv::cuda::GpuMat m_max_roi_score_frame{-1};
  double m_fps{25.0};
  // 0-51, lower is better, effectively acts like CRF.
  // A value of ~25-30 is usually good for NVENC H.264.
  uint8_t m_target_quality{30};
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer{nullptr};
  // std::unique_ptr<Utils::RamVideoBuffer> m_ram_buf{nullptr};
  std::string m_temp_video_path;
  Utils::VideoRecordingState m_state{Utils::IDLE};
  std::queue<AsyncPayload> m_frames_queue;
  // last member: joined first, before anything the sends still use
  std::vector<std::future<void>> m_pending_sends;

  void consume_pending_sends(bool wait);
  RoiLookupResult look_for_roi(const PipelineContext &ctx) const;

  void handle_video(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    RoiLookupResult roi_flag);

  double calculate_roi_score(const PipelineContext &ctx) const;
  void finalize_video_then_send_out(std::string temp_video_path,
                                    std::string trimmed_video_path,
                                    std::string jpeg_data,
                                    cv::Size thumbnail_size,
                                    size_t frame_count) const;
  std::optional<std::string> trim_video(const std::string &input_video_path,
                                        const std::string &trimmed_video_path,
                                        int frames_to_remove) const;

public:
  explicit MatrixNotifier(const std::string &unit_path)
      : IAsynchronousProcessingUnit(unit_path + "/MatrixNotifier") {}
  ~MatrixNotifier() override { consume_pending_sends(true); }
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit