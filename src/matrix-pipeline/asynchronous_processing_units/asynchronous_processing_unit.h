#pragma once

#include "../entities/processing_units_variant.h"
#include "../interfaces/i_asynchronous_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class AsynchronousProcessingUnit final : public IAsynchronousProcessingUnit {
  std::vector<ProcessingUnitVariant> m_processing_units;

public:
  explicit AsynchronousProcessingUnit(const std::string &unit_path)
      : IAsynchronousProcessingUnit(unit_path + "/AsynchronousProcessingUnit") {
  }
  ~AsynchronousProcessingUnit() override;
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit