#include "sface_overlay.h"
#include "../utils/misc.h"

#include <fmt/format.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

bool SFaceOverlay::init(const njson &config) {
  identity_to_box_color_bgr[IdentityCategory::Unknown] =
      cv::Scalar(127, 127, 127);
  identity_to_box_color_bgr[IdentityCategory::Unauthorized] =
      cv::Scalar(0, 179, 255);
  identity_to_box_color_bgr[IdentityCategory::Authorized] =
      cv::Scalar(0, 181, 0);

  try {
    m_bounding_box_border_thickness = config.value(
        "boundingBoxBorderThickness", m_bounding_box_border_thickness);
    m_label_font_scale = config.value("labelFontScale", m_label_font_scale);
    m_label_font_thickness =
        config.value("labelFontThickness", m_label_font_thickness);

    SPDLOG_INFO("bounding_box_border_thickness: {}, label_font_scale: {}, "
                "label_font_thickness: {}",
                m_bounding_box_border_thickness, m_label_font_scale,
                m_label_font_thickness);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to init SFaceOverlay: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult SFaceOverlay::process(cv::cuda::GpuMat &frame,
                                                  PipelineContext &ctx) {

  njson yunet_jsons;
  njson sface_jsons;
  // We also check YuNet because SFace results rely on YuNet geometry.
  if (ctx.yunet_sface.results.empty()) {
    ctx.text_to_overlay += fmt::format("YuNet: []\nSFace: []\n");
    return failure_and_continue;
  }

  frame.download(m_pinned_mem_for_cpu_frame);
  // point the cv::Mat directly to the pinned memory
  m_frame_cpu = m_pinned_mem_for_cpu_frame.createMatHeader();

  for (const auto &[detection, recognition] : ctx.yunet_sface.results) {

    njson yunet_json;
    yunet_json["conf"] = std::round(detection.face_score * 100.0) / 100.0;
    yunet_jsons.emplace_back(yunet_json);
    // Convert Rect2f (float) to Rect (int) for cleaner pixel drawing
    const cv::Rect bounding_box = detection.bounding_box;

    // Draw the bounding box
    cv::rectangle(m_frame_cpu, bounding_box,
                  identity_to_box_color_bgr[recognition.category],
                  m_bounding_box_border_thickness);
    njson sface_json;
    sface_json["ID"] = recognition.identity;
    sface_json["cos"] = std::round(recognition.cosine_score * 100.0) / 100.0;
    sface_json["L2"] = std::round(recognition.l2_norm * 100.0) / 100.0;
    sface_jsons.emplace_back(sface_json);
    if (!recognition.l2_norm_threshold_crossed ||
        !recognition.cosine_score_threshold_crossed)
      continue;

    const auto bounding_box_label_text =
        recognition.category != IdentityCategory::Unknown
            ? fmt::format("{}", recognition.identity)
            : "?";

    // --- POSITIONING LOGIC FIX START ---

    int baseLine;
    cv::Size label_size =
        cv::getTextSize(bounding_box_label_text, cv::FONT_HERSHEY_SIMPLEX,
                        m_label_font_scale, m_label_font_thickness, &baseLine);

    const auto label_x = bounding_box.x;
    int label_y = 0;

    // SAFETY MARGIN: 5px padding + the thickness of the face border + the text
    // tail size This ensures the green box's bottom edge barely touches the top
    // of the face border
    int margin_above = 5 + m_bounding_box_border_thickness + baseLine;

    // 1. Try placing ABOVE the face
    // We subtract 'margin_above' to lift the baseline high enough so the
    // descenders don't touch
    int y_above = bounding_box.y - margin_above;

    // Check if the TOP of the green box (y_above - height) stays on screen
    bool fits_above = (y_above - label_size.height) >= 0;

    if (fits_above) {
      // Option A: Place Above
      // We use the calculated y_above which already accounts for the
      // baseline/tails
      label_y = bounding_box.y - 5 - m_bounding_box_border_thickness - baseLine;
    } else {
      // Option B: Place Below
      // Calculate position so the TOP of the text box starts below the face
      // Face Bottom + Border + 5px Margin + Text Height
      label_y = (bounding_box.y + bounding_box.height) +
                m_bounding_box_border_thickness + 5 + label_size.height;
    }

    // Draw label background box
    // We explicitly calculate the rectangle coordinates to ensure they match
    // the text
    cv::Point box_top_left(label_x, label_y - label_size.height);
    cv::Point box_bottom_right(label_x + label_size.width, label_y + baseLine);

    cv::rectangle(m_frame_cpu, box_top_left, box_bottom_right,
                  identity_to_box_color_bgr[recognition.category], cv::FILLED);

    // Draw label text
    cv::putText(m_frame_cpu, bounding_box_label_text,
                cv::Point(label_x, label_y), cv::FONT_HERSHEY_SIMPLEX,
                m_label_font_scale, m_text_color_bgr, m_label_font_thickness);

    // --- POSITIONING LOGIC FIX END ---
  }

  ctx.text_to_overlay +=
      fmt::format("YuNet: {}\n", Utils::hybrid_njson_array_dump(yunet_jsons));
  ctx.text_to_overlay +=
      fmt::format("SFace: {}\n", Utils::hybrid_njson_array_dump(sface_jsons));

  frame.upload(m_frame_cpu);

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit