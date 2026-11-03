#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Define alias if not already defined in a common header
using njson = nlohmann::json;
using uchar = unsigned char;

namespace MatrixPipeline::Utils {

class MatrixSender {
private:
  std::string homeServer;
  std::string accessToken;
  std::string roomId;

  // Internal helper to perform HTTP POST for media
  std::string upload(const std::string &data,
                     const std::string &contentType) const;

  // Internal helper to perform HTTP PUT for events
  bool send_event(const njson &contentBody, const std::string &msgType) const;

  // Helper to clean URL strings
  static std::string sanitizeUrl(std::string url);

public:
  // Constructor
  MatrixSender(std::string url, std::string token, std::string room);

  // Send plain text message
  void sendText(const std::string &message) const;

  void send_video_from_memory(
      const std::string &video_data, const std::string &caption,
      size_t duration_ms = 0, std::string body = "",
      const std::string &thumbnail_data = {}, int width = 0, int height = 0,
      const std::string &thumb_mime = "image/jpeg") const;
};

} // namespace MatrixPipeline::Utils