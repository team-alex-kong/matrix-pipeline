#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../utils/misc.h"
#include "../utils/nvjpeg_encoder.h"

#include <drogon/drogon.h>
#include <drogon/utils/Utilities.h>
#include <nvjpeg.h>

#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <spdlog/spdlog.h>
#include <string>

using namespace drogon;

namespace MatrixPipeline::ProcessingUnit {

class HttpService;

// --- Static Global State ---
inline std::map<uint16_t, HttpService *> s_service_registry;
inline std::mutex s_registry_mutex;
inline std::atomic s_global_handler_registered{false};

/**
 * @brief HTTP Server implementation.
 * Uses shared_ptr for zero-contention snapshot serving.
 * Supports /stream for VLC MJPEG streaming.
 */
class HttpService : public IAsynchronousProcessingUnit {
public:
  explicit HttpService(const std::string &unit_path)
      : IAsynchronousProcessingUnit(unit_path + "/HttpService") {}

  ~HttpService() override {
    std::lock_guard<std::mutex> lock(s_registry_mutex);
    if (m_port > 0) {
      s_service_registry.erase(m_port);
    }
  }

  bool init(const njson &config) override {
    try {
      m_gpu_encoder = std::make_unique<Utils::NvJpegEncoder>();

      m_ip = config.value("bindAddr", "127.0.0.1");
      m_port = config.value("port", 8080);

      if (config.contains("username") && config.contains("password")) {
        m_auth_enabled = true;
        m_username = config["username"];
        m_password = config["password"];
      }

      bool use_https = config.value("useHttps", false);
      std::string cert_path = config.value("certPath", "");
      std::string key_path = config.value("keyPath", "");

      m_refresh_interval_sec = std::chrono::duration<double>(
          config.value("refreshIntervalSec", 10.0));

      {
        std::lock_guard<std::mutex> lock(s_registry_mutex);
        if (s_service_registry.find(m_port) != s_service_registry.end()) {
          SPDLOG_WARN("Port {} is already claimed!", m_port);
        }
        s_service_registry[m_port] = this;
      }

      app().addListener(m_ip, m_port, use_https, cert_path, key_path);

      bool expected = false;
      if (s_global_handler_registered.compare_exchange_strong(expected, true)) {

        // 1. Snapshot Handler (GET /)
        app().registerHandler(
            "/",
            [](const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback) {
              dispatch_request(req, std::move(callback), false);
            },
            {Get});

        // 2. Stream Handler (GET /stream) for VLC
        app().registerHandler(
            "/stream",
            [](const HttpRequestPtr &req,
               std::function<void(const HttpResponsePtr &)> &&callback) {
              dispatch_request(req, std::move(callback), true);
            },
            {Get});
      }

      SPDLOG_INFO("HttpService initialized on {}:{} (HTTPS: {}), "
                  "refresh_interval_sec: {}",
                  m_ip, m_port, use_https, m_refresh_interval_sec.count());
      return true;
    } catch (const std::exception &e) {
      SPDLOG_ERROR("Failed to init HttpService: {}", e.what());
      return false;
    }
  }

  // Static helper to route request to correct instance
  static void
  dispatch_request(const HttpRequestPtr &req,
                   std::function<void(const HttpResponsePtr &)> &&callback,
                   bool is_stream) {
    uint16_t local_port = req->getLocalAddr().toPort();
    HttpService *service = nullptr;
    {
      std::lock_guard<std::mutex> lock(s_registry_mutex);
      auto it = s_service_registry.find(local_port);
      if (it != s_service_registry.end())
        service = it->second;
    }

    if (service) {
      if (is_stream)
        service->handle_stream_request(req, std::move(callback));
      else
        service->handle_snapshot_request(req, std::move(callback));
    } else {
      auto resp = HttpResponse::newHttpResponse();
      resp->setStatusCode(k404NotFound);
      resp->setBody("No Service attached to this port");
      callback(resp);
    }
  }

protected:
  void on_frame_ready(cv::cuda::GpuMat &frame,
                      [[maybe_unused]] PipelineContext &ctx) override {
    auto now = std::chrono::steady_clock::now();
    if (now - m_last_update_time < m_refresh_interval_sec)
      return;
    if (frame.empty())
      return;

    if (!m_gpu_encoder) {
      SPDLOG_ERROR("NvJpegEncoder not initialized!");
      return;
    }

    try {
      std::string temp_buffer;
      bool success = m_gpu_encoder->encode(frame, temp_buffer, 90);

      if (success) {
        // Optimization: Create the shared string OUTSIDE the lock.
        // This is the heavy memory allocation and copy.
        auto new_image_ptr = std::make_shared<std::string>(temp_buffer.begin(),
                                                           temp_buffer.end());

        {
          std::lock_guard<std::mutex> lock(m_image_mutex);
          m_latest_jpeg_ptr = std::move(new_image_ptr);
        }

        m_last_update_time = now;
        SPDLOG_DEBUG("HttpService Port {}: Updated image.", m_port);

        // Broadcast to any connected VLC/Stream clients
        broadcast_stream(temp_buffer);
      }
    } catch (const std::exception &e) {
      SPDLOG_ERROR("Processing error: {}", e.what());
    }
  }

private:
  // Handles standard browser image requests (GET /)
  void handle_snapshot_request(
      const HttpRequestPtr &req,
      std::function<void(const HttpResponsePtr &)> &&callback) {

    if (!perform_auth(req, callback))
      return;

    // Optimization: Low-contention read
    std::shared_ptr<std::string> img_ptr_copy;
    {
      std::lock_guard<std::mutex> lock(m_image_mutex);
      img_ptr_copy = m_latest_jpeg_ptr;
    }

    if (!img_ptr_copy) {
      auto resp = HttpResponse::newHttpResponse();
      resp->setStatusCode(k503ServiceUnavailable);
      resp->setBody("No frame available yet.");
      callback(resp);
      return;
    }

    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k200OK);
    // Explicitly used your requested enum
    resp->setContentTypeCode(CT_IMAGE_JPG);
    resp->setBody(*img_ptr_copy);

    callback(resp);
  }

  // Handles VLC streaming requests (GET /stream)
  void handle_stream_request(
      const HttpRequestPtr &req,
      std::function<void(const HttpResponsePtr &)> &&callback) {

    if (!perform_auth(req, callback))
      return;

    const auto conn_weak = req->getConnectionPtr(); // Get weak_ptr
    auto conn = conn_weak.lock();                   // Convert to shared_ptr
    if (!conn)
      return;

    // 1. Send MJPEG Header manually to hijack the stream
    std::string header = "HTTP/1.1 200 OK\r\n"
                         "Content-Type: multipart/x-mixed-replace; "
                         "boundary=matrix_pipeline_frame\r\n"
                         "Connection: keep-alive\r\n"
                         "Pragma: no-cache\r\n"
                         "Cache-Control: no-cache\r\n"
                         "\r\n";
    conn->send(header);

    // 2. Register Connection
    {
      std::lock_guard<std::mutex> lock(m_stream_mutex);
      m_stream_clients.push_back(conn);
    }
  }

  void broadcast_stream(const std::string &buffer) {
    std::lock_guard<std::mutex> lock(m_stream_mutex);
    if (m_stream_clients.empty())
      return;

    // OPTIMIZATION: Construct the payload string ONCE
    std::string chunk_header = "--matrix_pipeline_frame\r\n"
                               "Content-Type: image/jpeg\r\n"
                               "Content-Length: " +
                               std::to_string(buffer.size()) + "\r\n\r\n";

    std::string chunk_footer = "\r\n";

    auto it = m_stream_clients.begin();
    while (it != m_stream_clients.end()) {
      auto conn = it->lock();
      if (conn && conn->connected()) {
        conn->send(chunk_header);
        conn->send(buffer); // Send the same string object
        conn->send(chunk_footer);
        ++it;
      } else {
        it = m_stream_clients.erase(it);
      }
    }
  }

  bool perform_auth(
      const HttpRequestPtr &req,
      const std::function<void(const HttpResponsePtr &)> &callback) const {
    if (m_auth_enabled) {
      if (const std::string auth_header = req->getHeader("Authorization");
          !check_auth(auth_header)) {
        auto resp = HttpResponse::newHttpResponse();
        resp->setStatusCode(k401Unauthorized);
        resp->addHeader("WWW-Authenticate", "Basic realm=\"MatrixPipeline\"");
        resp->setBody("Unauthorized");
        callback(resp);
        return false;
      }
    }
    return true;
  }

  bool check_auth(const std::string &header_val) const {
    if (header_val.empty())
      return false;
    const size_t split_pos = header_val.find(' ');
    if (split_pos == std::string::npos ||
        header_val.substr(0, split_pos) != "Basic")
      return false;
    const std::string encoded = header_val.substr(split_pos + 1);
    std::string decoded = drogon::utils::base64Decode(encoded);
    const size_t colon_pos = decoded.find(':');
    if (colon_pos == std::string::npos)
      return false;
    const std::string u = decoded.substr(0, colon_pos);
    const std::string p = decoded.substr(colon_pos + 1);
    return (u == m_username && p == m_password);
  }

  std::string m_ip;
  uint16_t m_port = 0;
  bool m_auth_enabled = false;
  std::string m_username;
  std::string m_password;

  std::chrono::duration<double> m_refresh_interval_sec{10.0};
  std::chrono::steady_clock::time_point m_last_update_time;

  std::mutex m_image_mutex;
  std::shared_ptr<std::string> m_latest_jpeg_ptr;

  std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder{nullptr};

  // Streaming State (VLC Support)
  std::mutex m_stream_mutex;
  std::list<std::weak_ptr<trantor::TcpConnection>> m_stream_clients;
};

} // namespace MatrixPipeline::ProcessingUnit