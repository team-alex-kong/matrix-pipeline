#include "overlay_text.h"
#include "../utils/misc.h"

#include <fmt/chrono.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

bool OverlayText::init(const nlohmann::json &config) {
  try {
    m_text_height_ratio = config.value("textHeightRatio", m_text_height_ratio);
    m_outline_ratio = config.value("outlineRatio", m_outline_ratio);

    m_overlay_interval = std::chrono::milliseconds(
        config.value("overlayIntervalMs", m_overlay_interval.count()));

    SPDLOG_INFO(
        "outline_ratio: {}, text_height_ratio: {}, overlay_interval(ms): {}",
        m_outline_ratio, m_text_height_ratio, m_overlay_interval.count());
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("error: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult OverlayText::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  if (frame.empty())
    return success_and_continue;

  using namespace std::chrono_literals;
  if (std::chrono::steady_clock::now() - m_last_overlay_at >
      m_overlay_interval) {
    m_last_overlay_at = std::chrono::steady_clock::now();

    // --- 2. Split into Lines ---
    std::vector<std::string> lines;
    std::stringstream ss(ctx.text_to_overlay);
    std::string line;
    while (std::getline(ss, line, '\n')) {
      lines.push_back(line);
    }

    if (lines.empty())
      return success_and_continue;

    // --- 3. Update Font Metrics ---
    update_font_metrics(frame.rows);

    // --- 4. Prepare Rendering Surface ---
    m_stripHeight =
        (static_cast<int>(lines.size()) * m_line_height_px) + (2 * m_margin_y);
    if (m_stripHeight > frame.rows)
      m_stripHeight = frame.rows;

    if (m_h_text_strip.cols != frame.cols ||
        m_h_text_strip.rows != m_stripHeight ||
        m_h_text_strip.type() != CV_8UC3) {
      m_h_text_strip.create(m_stripHeight, frame.cols, CV_8UC3);
    }

    m_h_text_strip.setTo(cv::Scalar::all(0));

    // --- 5. Draw Lines (Left Aligned) ---
    int currentY = m_margin_y + static_cast<int>(BASE_FONT_HEIGHT_PX *
                                                 m_current_opencv_scale);
    int x = m_margin_x;

    for (const auto &txt : lines) {
      if (txt.empty()) {
        currentY += m_line_height_px;
        continue;
      }

      cv::Point org(x, currentY);

      if (m_outline_ratio > 0.0f) {
        // Draw thicker outline behind
        cv::putText(m_h_text_strip, txt, org, cv::FONT_HERSHEY_DUPLEX,
                    m_current_opencv_scale, m_glow_color,
                    m_current_outline_thickness, cv::LINE_AA);
      }
      // Draw main text
      cv::putText(m_h_text_strip, txt, org, cv::FONT_HERSHEY_DUPLEX,
                  m_current_opencv_scale, m_text_color, m_current_thickness,
                  cv::LINE_AA);

      currentY += m_line_height_px;
    }
  }

  upload_and_overlay(frame, cv::Rect(0, 0, frame.cols, m_stripHeight));

  return success_and_continue;
}

void OverlayText::update_font_metrics(const int frameRows) {
  // 1. Calculate base text height
  float final_px_height = std::max(frameRows * m_text_height_ratio, 6.0f);

  m_current_opencv_scale = final_px_height / BASE_FONT_HEIGHT_PX;
  m_current_thickness = std::max(1, static_cast<int>(final_px_height / 20.0f));

  // 2. Calculate Border Size
  int border_px = 0;
  if (m_outline_ratio > 0.0f) {
    border_px = static_cast<int>(final_px_height * m_outline_ratio);
    // Safety: Ensure at least 1px if ratio is set
    if (border_px < 1)
      border_px = 1;

    m_current_outline_thickness = m_current_thickness + (2 * border_px);
  } else {
    m_current_outline_thickness = 0;
  }

  // 3. Fix Line Height
  // Logic: (Font Height + Standard Spacing) + (Top Border + Bottom Border)
  m_line_height_px = static_cast<int>(final_px_height * 1.2f) + (2 * border_px);
}

void OverlayText::upload_and_overlay(const cv::cuda::GpuMat &frame,
                                     const cv::Rect roi_rect) {
  cv::Rect validRoi = roi_rect & cv::Rect(0, 0, frame.cols, frame.rows);
  if (validRoi.empty())
    return;

  cv::Mat cpuSrc =
      m_h_text_strip(cv::Rect(0, 0, validRoi.width, validRoi.height));
  m_d_text_strip.upload(cpuSrc);

  cv::cuda::cvtColor(m_d_text_strip, m_d_strip_gray, cv::COLOR_BGR2GRAY);
  cv::cuda::threshold(m_d_strip_gray, m_d_mask, 1, 255, cv::THRESH_BINARY);

  cv::cuda::GpuMat roi = frame(validRoi);
  m_d_text_strip.copyTo(roi, m_d_mask);
}

} // namespace MatrixPipeline::ProcessingUnit