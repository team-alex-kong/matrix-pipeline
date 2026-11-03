#pragma once

#include <opencv2/opencv.hpp>

namespace MatrixPipeline::ProcessingUnit {

struct DeviceInfo {
  std::string name;
  std::string uri;
  cv::Size expected_frame_size;
};

enum class IdentityCategory { Unknown, Authorized, Unauthorized };

struct SFaceRecognition {
  cv::Mat embedding; // The 128-d vector
  std::string identity{"Unknown"};
  double cosine_score{std::numeric_limits<double>::quiet_NaN()};
  bool cosine_score_threshold_crossed{false};
  IdentityCategory category = IdentityCategory::Unknown;
  double l2_norm{std::numeric_limits<double>::quiet_NaN()};
  bool l2_norm_threshold_crossed{false};
};
struct YuNetDetection {
  // we need this raw output because cv::FaceRecognizerSF::alignCrop() expects
  // it as an input
  cv::Mat yunet_output;
  cv::Rect2f bounding_box;
  std::array<cv::Point2f, 5> landmarks; // 5 points: eyes, nose, mouth corners
  float face_score;
};

struct YuNetSFaceResult {
  YuNetDetection detection;
  SFaceRecognition recognition;
};

struct YuNetSFaceContext {
  cv::Size2i yunet_input_frame_size;
  std::vector<YuNetSFaceResult> results;
};

struct YoloContext {
  cv::Size inference_input_size;
  std::vector<cv::Rect> bounding_boxes;
  std::vector<size_t> class_ids;
  std::vector<short> is_detection_interesting;
  // Must be float as mandated by cv::dnn::NMSBoxes
  std::vector<float> confidences;
  std::vector<int> indices;
};

struct PipelineContext {
  DeviceInfo device_info;
  // false means it is a gray image
  bool captured_from_real_device = false;
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>
      capture_timestamp;
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>
      capture_from_this_device_since;
  uint32_t frame_seq_num = 0;
  size_t processing_unit_idx = 0;
  float change_rate = -1;
  float fps = 0.0;
  std::chrono::steady_clock::time_point latency_start_time;

  YoloContext yolo;
  YuNetSFaceContext yunet_sface;
  std::string text_to_overlay;
  // SFaceContext sface;
};
} // namespace MatrixPipeline::ProcessingUnit