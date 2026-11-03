#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/objdetect.hpp>

namespace MatrixPipeline::ProcessingUnit {

class YuNetDetect : public ISynchronousProcessingUnit {
private:
  cv::Ptr<cv::FaceDetectorYN> m_detector;

  float m_face_score_threshold = 0.9f;
  float m_nms_threshold = 0.3f;
  // m_top_k typical ranges:
  // Industry Standard (Typical)  400 - 1,000
  // Dense Crowd Analysis	  2,000 - 5,000
  // Real-time / Embedded	  50 - 100
  // Aggressive Filtering	  10 - 20
  int m_top_k = 100;
  cv::cuda::HostMem m_pinned_buffer;

public:
  explicit YuNetDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YuNetDetect") {}

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit