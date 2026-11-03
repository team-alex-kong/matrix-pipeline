#pragma once

#include "../entities/processing_context.h"

#include <drogon/drogon.h>
#include <fmt/args.h>
#include <fmt/base.h>
#include <fmt/chrono.h>
#include <fmt/compile.h>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <string>

namespace MatrixPipeline::Utils {
using njson = nlohmann::json;
typedef auto (*signal_handler_callback)(int) -> void;

void install_signal_handler(signal_handler_callback cb);

std::string hybrid_njson_array_dump(const njson &j_array);

// seems std::format() wont work if defined in .cpp file
inline std::optional<std::string> evaluate_text_template(
    const std::string &string_template,
    const std::optional<ProcessingUnit::PipelineContext> &ctx = std::nullopt,
    const std::chrono::system_clock::time_point timestamp =
        std::chrono::system_clock::now()) {
  std::string evaluated_string = string_template;
  static const std::regex placeholder_regex(R"(\{timestamp(?::([^}]+))?\})");
  std::smatch match;
  if (std::regex_search(evaluated_string, match, placeholder_regex)) {
    const auto ms_part = std::chrono::duration_cast<std::chrono::milliseconds>(
                             timestamp.time_since_epoch()) %
                         1000;
    auto t_c = std::chrono::system_clock::to_time_t(timestamp);
    std::tm tm = *std::localtime(&t_c);
    std::string time_format = "{:%Y%m%d-%H%M%S}";
    if (match[1].matched) {
      std::string user_fmt = match[1].str();
      size_t f_pos = user_fmt.find("%f");
      if (f_pos != std::string::npos)
        user_fmt.replace(f_pos, 2, fmt::format("{:03d}", ms_part.count()));
      time_format = "{:" + user_fmt + "}";
    }
    try {
      const std::string formatted_time =
          fmt::format(fmt::runtime(time_format), tm);
      evaluated_string.replace(match.position(), match.length(),
                               formatted_time);
    } catch (const std::exception &e) {
      SPDLOG_ERROR("e.what(): {}", e.what());
      return std::nullopt;
    }
  }
  if (!ctx.has_value())
    return evaluated_string;
  try {
    evaluated_string = fmt::format(
        fmt::runtime(evaluated_string),
        fmt::arg("deviceName", ctx->device_info.name),
        fmt::arg("fps", ctx->fps), fmt::arg("changeRate", ctx->change_rate),
        fmt::arg("changeRatePct", ctx->change_rate * 100));
  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what(): {}, evaluated_string: {}", e.what(),
                 evaluated_string);
    return std::nullopt;
  }
  return evaluated_string;
}

} // namespace MatrixPipeline::Utils