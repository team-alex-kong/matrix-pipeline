#include "matrix_sender.h"
#include "../global_vars.h"

#include <cpr/cpr.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace MatrixPipeline::Utils {

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------
/*
void MatrixSender::stbiWriteFunc(void *context, void *data, int size) {
    auto *buffer = static_cast<std::vector<unsigned char>*>(context);
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    buffer->insert(buffer->end(), bytes, bytes + size);
}*/

std::string MatrixSender::sanitizeUrl(std::string url) {
  while (!url.empty() && url.back() == '/') {
    url.pop_back();
  }
  return url;
}

std::string MatrixSender::readFile(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs)
    return "";
  auto pos = ifs.tellg();
  std::string result(pos, '\0');
  ifs.seekg(0, std::ios::beg);
  ifs.read(&result[0], pos);
  return result;
}

std::string MatrixSender::upload(const std::string &data,
                                 const std::string &contentType) const {
  std::string url = homeServer + "/_matrix/media/v3/upload";

  constexpr int max_retry_count = 10;

  cpr::Response r;
  for (int i = 0; i < max_retry_count && (i == 0 || ev_flag == 0); ++i) {
    r = cpr::Post(cpr::Url{url},
                  cpr::Header{{"Authorization", "Bearer " + accessToken},
                              {"Content-Type", contentType}},
                  /*cpr::Proxies{
                      {"http", "socks5://localhost:7890"},
                      {"https", "socks5://localhost:7890"},
                  },*/
                  cpr::Body{data}, cpr::LowSpeed{1024, 30});
    if (r.status_code == 200) {
      try {
        auto j = njson::parse(r.text);
        return j["content_uri"];
      } catch (const std::exception &e) {
        SPDLOG_ERROR("Unable to parse response from matrix server: {}",
                     e.what());
      }
    }
    SPDLOG_ERROR("Upload failed, status_code: {}, text: {}, retry {}/{}",
                 r.status_code, r.text, i + 1, max_retry_count);
    std::this_thread::sleep_for(std::chrono::seconds(i * 10));
  }
  return "";
}

bool MatrixSender::send_event(const njson &contentBody,
                              const std::string &msgType) const {
  auto now = std::chrono::system_clock::now().time_since_epoch().count();
  const std::string txnId = std::to_string(now);
  const std::string url = homeServer + "/_matrix/client/v3/rooms/" + roomId +
                          "/send/m.room.message/" + txnId;

  njson payload;
  payload["msgtype"] = msgType;
  payload["body"] = contentBody["body"];

  if (contentBody.contains("url"))
    payload["url"] = contentBody["url"];

  if (contentBody.contains("info")) {
    payload["info"] = contentBody["info"];
  }

  // 1. Create the Session
  cpr::Session session;
  session.SetUrl(cpr::Url{url});
  session.SetHeader(cpr::Header{
      {"Authorization", "Bearer " + accessToken},
      {"Content-Type", "application/json"} // Fixed typo here too
  });
  session.SetBody(cpr::Body{payload.dump()});
  session.SetLowSpeed(cpr::LowSpeed{1024, 30});
  // session.SetVerbose(cpr::Verbose{true});
  // session.SetProxies(cpr::Proxies{{"http", "socks5://localhost:7890"},
  //                                 {"https", "socks5://localhost:7890"}});

  // 3. Send the request
  if (cpr::Response r = session.Put(); r.status_code != 200) {
    SPDLOG_ERROR("Send message failed.  status_code: {}, reason: {}",
                 r.status_code, r.reason);
    return false;
  } else {
    SPDLOG_INFO("Send message {} successfully", msgType);
    SPDLOG_DEBUG("payload: {}", payload.dump(2));
    return true;
  }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

MatrixSender::MatrixSender(std::string url, std::string token,
                           std::string room) {
  if (url.empty() || token.empty() || room.empty()) {
    throw std::runtime_error("MatrixSender requires URL, Token, and RoomID");
  }
  homeServer = sanitizeUrl(url);
  accessToken = token;
  roomId = room;
}

void MatrixSender::sendText(const std::string &message) const {
  if (message.empty())
    return;
  njson content;
  content["body"] = message;
  send_event(content, "m.text");
}

void MatrixSender::send_jpeg(const std::string &jpeg_bytes, int width,
                             int height, const std::string &caption) const {
  if (std::string mxc = upload(jpeg_bytes, "image/jpeg"); !mxc.empty()) {
    njson content;
    content["body"] = caption;
    content["url"] = mxc;
    content["info"]["w"] = width;
    content["info"]["h"] = height;
    content["info"]["mimetype"] = "image/jpeg";
    content["info"]["size"] = jpeg_bytes.size();
    send_event(content, "m.image");
  }
}

void MatrixSender::send_video(const std::string &filepath,
                              const std::string &caption, size_t duration_ms,
                              const std::string &body,
                              const std::string &thumbnail_data, int width,
                              int height, const std::string &thumb_mime) {
  std::string data = readFile(filepath);
  if (data.empty()) {
    SPDLOG_ERROR("readFile({}) is empty, returning", filepath);
    return;
  }
  send_video_from_memory(data, caption, duration_ms, body, thumbnail_data,
                         width, height, thumb_mime);
}

void MatrixSender::send_video_from_memory(const std::string &video_data,
                                          const std::string &caption,
                                          size_t duration_ms, std::string body,
                                          const std::string &thumbnail_data,
                                          int width, int height,
                                          const std::string &thumb_mime) const {
  if (video_data.empty())
    return;

  // 1. Upload Video
  std::string video_mxc = upload(video_data, "video/mp4");
  if (video_mxc.empty())
    return;

  njson content;
  content["body"] = caption;
  content["url"] = video_mxc;

  njson info;
  info["mimetype"] = "video/mp4";
  info["size"] = video_data.size();
  if (duration_ms > 0)
    info["duration"] = duration_ms;

  info["w"] = width;
  info["h"] = height;

  // 2. Upload Thumbnail (if present)
  if (!thumbnail_data.empty()) {

    if (std::string thumb_mxc = upload(thumbnail_data, thumb_mime);
        !thumb_mxc.empty()) {
      info["thumbnail_url"] = thumb_mxc;

      njson thumb_info;
      thumb_info["mimetype"] = thumb_mime;
      thumb_info["size"] = thumbnail_data.size();
      if (width > 0 && height > 0) {
        thumb_info["w"] = width;
        thumb_info["h"] = height;
      }
      info["thumbnail_info"] = thumb_info;
    }
  }

  content["info"] = info; /*
   if (!body.empty()) {
     content["info"]["filename"] = body.ends_with(".mp4") ? body : body +
   ".mp4"; content["body"] = body.ends_with(".mp4") ? body : body + ".mp4";
     content["format"] = "org.matrix.custom.html";
     content["formatted_body"] = "Your Caption Here";
   }*/
  // SPDLOG_INFO("content: {}", content.dump());
  constexpr size_t max_retry_count = 5;
  for (size_t i = 0; i < max_retry_count && (i == 0 || ev_flag == 0); ++i) {
    if (send_event(content, "m.video"))
      break;
    const auto retry_delay_sec = i * 5;
    SPDLOG_ERROR("failed, {}-th retry in {} sec. max_retry_count: {}", i + 1,
                 retry_delay_sec, max_retry_count);
    std::this_thread::sleep_for(std::chrono::seconds(retry_delay_sec));
  }
}

} // namespace MatrixPipeline::Utils