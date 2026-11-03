#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"
#include "../utils/cuda_helper.h"

#include <NvInfer.h> // TensorRT Core Header
#include <NvOnnxParser.h> // ONNX Parser Header (Required to build the engine from .onnx at runtime)
#include <opencv2/core/cuda.hpp>

namespace MatrixPipeline::ProcessingUnit {

using namespace std::chrono_literals;

struct BoundingBoxScaleParams {
  double scale;
  int x_offset;
  int y_offset;
};

class YoloDetect final : public ISynchronousProcessingUnit {
private:
  // --- TensorRT Smart Pointers ---
  // We use unique_ptrs with custom deleters or shared_ptrs for TRT objects
  std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
  std::unique_ptr<nvinfer1::IExecutionContext> m_context;

  // --- GPU Memory Management ---
  // Raw pointer for TRT binding
  std::unique_ptr<float, Utils::CudaDeleter> m_output_buffer_gpu = nullptr;
  std::unique_ptr<float, Utils::CudaDeleter> m_input_buffer_gpu = nullptr;
  std::vector<float> m_output_cpu; // For downloading inference results
  size_t m_output_count = 0;       // Total floats in the output tensor

  cudaStream_t m_cuda_stream = nullptr; // Async execution stream
  cv::cuda::Stream m_cv_stream;
  int m_output_dimensions{-1};
  int m_output_rows{-1};
  // --- OpenCV GpuMat Buffers (Reuse these to avoid re-allocation) ---
  cv::cuda::GpuMat m_resized_gpu, m_resized_gpu_buffer;
  cv::cuda::GpuMat m_normalized_gpu;
  cv::cuda::GpuMat m_rgb;
  std::vector<cv::cuda::GpuMat> m_channels;

  // Non-TRT-related
  cv::Size m_model_input_size = {640, 640}; // Default YOLO size
  float m_confidence_threshold = 0.5f;
  float m_nms_thres = 0.45f;
  std::chrono::milliseconds m_inference_interval = 100ms;
  std::chrono::time_point<std::chrono::steady_clock> m_last_inference_time;
  YoloContext m_prev_yolo_ctx;

  void post_process_yolo(PipelineContext &ctx) const;

public:
  explicit YoloDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YoloDetect") {}

  bool init(const njson &config) override;

  ~YoloDetect() override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
  static BoundingBoxScaleParams get_bounding_box_scale(const cv::cuda::GpuMat &,
                                                       const PipelineContext &);

  static cv::Rect
  get_scaled_bounding_box_coordinates(const cv::Rect &orig_box,
                                      const BoundingBoxScaleParams &params);
};

} // namespace MatrixPipeline::ProcessingUnit