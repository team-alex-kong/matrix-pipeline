#include "yolo_publish_mqtt.h"

#include <cpr/cpr.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <csignal>
#include <fmt/chrono.h>
#include <thread>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {
void mosq_log_callback([[maybe_unused]] mosquitto *mosq, void *obj, int level,
                       const char *str) {

  switch (level) {
    // case MOSQ_LOG_DEBUG:
  case MOSQ_LOG_INFO:
    SPDLOG_INFO("{}", str);
  case MOSQ_LOG_NOTICE:
    SPDLOG_INFO("{}", str);
  case MOSQ_LOG_WARNING:
    SPDLOG_WARN("{}", str);
  case MOSQ_LOG_ERR: {
    SPDLOG_ERROR("{}", str);
  }
  default:;
  }
}

void on_connect([[maybe_unused]] mosquitto *mosq, [[maybe_unused]] void *obj,
                int reason_code) {
  /* Print out the connection result. mosquitto_connack_string() produces an
   * appropriate string for MQTT v3.x clients, the equivalent for MQTT v5.0
   * clients is mosquitto_reason_string().
   */
  SPDLOG_INFO("on_connect(): {}({})", reason_code,
              mosquitto_connack_string(reason_code));
}

void on_disconnect([[maybe_unused]] mosquitto *mosq, [[maybe_unused]] void *obj,
                   int reason_code) {
  /* Print out the connection result. mosquitto_connack_string() produces an
   * appropriate string for MQTT v3.x clients, the equivalent for MQTT v5.0
   * clients is mosquitto_reason_string().
   */
  SPDLOG_INFO("on_disconnect(): {}({})", reason_code,
              mosquitto_connack_string(reason_code));
}

/* Callback called when the client knows to the best of its abilities that a
 * PUBLISH has been successfully sent. For QoS 0 this means the message has
 * been completely written to the operating system. For QoS 1 this means we
 * have received a PUBACK from the broker. For QoS 2 this means we have
 * received a PUBCOMP from the broker. */
void on_publish([[maybe_unused]] mosquitto *mosq, void *obj, int msg_id) {
  const YoloPublishMqtt *self = static_cast<YoloPublishMqtt *>(obj);
  SPDLOG_INFO("Message (msg_id: {}) has been published to {}/{}", msg_id,
              self->m_mqtt_broker_url, self->m_mqtt_topic);
}

mosquitto *init_mosquitto(void *obj) {
  const YoloPublishMqtt *self = static_cast<YoloPublishMqtt *>(obj);
  int rc;
  struct mosquitto *mosq;
  /* Required before calling other mosquitto functions */
  if ((rc = mosquitto_lib_init()) != MOSQ_ERR_SUCCESS) {
    SPDLOG_ERROR("mosquitto_lib_init() failed: %s", mosquitto_strerror(rc));
    goto err_init_mosquitto;
  }

  mosq = mosquitto_new(nullptr, true, obj);
  if (mosq == nullptr) {
    SPDLOG_ERROR("mosquitto_new() failed");
    goto err_init_mosquitto;
  }
  if ((rc = mosquitto_username_pw_set(mosq, self->m_mqtt_username.c_str(),
                                      self->m_mqtt_password.c_str())) !=
      MOSQ_ERR_SUCCESS) {
    SPDLOG_ERROR("mosquitto_username_pw_set() failed: {}",
                 mosquitto_strerror(rc));
    goto err_mosquitto_config;
  }
  if ((rc = mosquitto_tls_set(mosq, self->m_mqtt_ca_file.c_str(), nullptr,
                              nullptr, nullptr, nullptr)) != MOSQ_ERR_SUCCESS) {
    SPDLOG_ERROR("mosquitto_tls_set() failed: {}", mosquitto_strerror(rc));
    goto err_mosquitto_config;
  }
  mosquitto_connect_callback_set(mosq, on_connect);
  mosquitto_disconnect_callback_set(mosq, on_disconnect);
  mosquitto_publish_callback_set(mosq, on_publish);
  mosquitto_log_callback_set(mosq, mosq_log_callback);

  /* Connect to test.mosquitto.org on port 1883, with a keepalive of 60
   * seconds. This call makes the socket connection only, it does not complete
   * the MQTT CONNECT/CONNACK flow, you should use mosquitto_loop_start() or
   * mosquitto_loop_forever() for processing net traffic. */
  rc = mosquitto_connect(mosq, self->m_mqtt_broker_url.c_str(), 8883, 60);
  if (rc != MOSQ_ERR_SUCCESS) {
    SPDLOG_ERROR("mosquitto_connect() failed: {}", mosquitto_strerror(rc));
    goto err_mosquitto_connect;
  }

  /* Run the network loop in a background thread, this call returns quickly.
   */
  rc = mosquitto_loop_start(mosq);
  if (rc != MOSQ_ERR_SUCCESS) {
    SPDLOG_ERROR("mosquitto_loop_start() failed: %s", mosquitto_strerror(rc));
    goto err_mosquitto_connect;
  }
  SPDLOG_INFO("mosquitto_loop_start()ed");

  return mosq;

err_mosquitto_config:
err_mosquitto_connect:
  mosquitto_destroy(mosq);
err_init_mosquitto:
  return nullptr;
}

bool YoloPublishMqtt::init(const nlohmann::json &config) {
  try {
    if (const std::string key = "mqttBrokerUrl"; config.contains(key)) {
      m_mqtt_broker_url = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} not defined", key);
      return false;
    }

    if (const std::string key = "mqttUsername"; config.contains(key)) {
      m_mqtt_username = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} not defined", key);
      return false;
    }

    if (const std::string key = "mqttPassword"; config.contains(key)) {
      m_mqtt_password = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} not defined", key);
      return false;
    }

    if (const std::string key = "mqttCaFile"; config.contains(key)) {
      m_mqtt_ca_file = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} not defined", key);
      return false;
    }

    if (const std::string key = "mqttTopic"; config.contains(key)) {
      m_mqtt_topic = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} not defined", key);
      return false;
    }
  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what: {}", e.what());
    return false;
  }
  SPDLOG_INFO("mqtt_broker_url: {}, mqtt_username: {}, mqtt_ca_file: {}, "
              "mqtt_topic: {}",
              m_mqtt_broker_url, m_mqtt_username, m_mqtt_ca_file, m_mqtt_topic);
  m_mosq = init_mosquitto(this);
  if (m_mosq == nullptr) {
    SPDLOG_ERROR("init_mosquitto() failed");
    return false;
  }
  SPDLOG_INFO("init_mosquitto()'ed");
  return true;
}

SynchronousProcessingResult
YoloPublishMqtt::process([[maybe_unused]] cv::cuda::GpuMat &frame,
                         PipelineContext &ctx) {
  using namespace std::chrono;

  njson payload;
  payload["bounding_boxes"] = {};
  for (const auto idx : ctx.yolo.indices) {
    if (!ctx.yolo.is_detection_interesting[idx])
      continue;
    njson box;
    box["x"] = ctx.yolo.bounding_boxes[idx].x;
    box["y"] = ctx.yolo.bounding_boxes[idx].y;
    box["w"] = ctx.yolo.bounding_boxes[idx].width;
    box["h"] = ctx.yolo.bounding_boxes[idx].height;
    payload["bounding_boxes"].push_back(box);
  }
  if (payload["bounding_boxes"].empty())
    return success_and_continue;

  payload["unix_time_ms"] =
      duration_cast<milliseconds>(system_clock::now().time_since_epoch())
          .count();
  payload["signal_source"] = "matrix-pipeline";
  // const auto payload_mp = njson::to_msgpack(payload);
  mosquitto_publish(m_mosq, nullptr, m_mqtt_topic.c_str(),
                    payload.dump().length(), payload.dump().c_str(), 2, false);
  return success_and_continue;
}
YoloPublishMqtt::~YoloPublishMqtt() {
  if (m_mosq != nullptr) {
    mosquitto_loop_stop(m_mosq, true);
    SPDLOG_INFO("mosquitto_loop_stop()'ed");
    mosquitto_destroy(m_mosq);
    SPDLOG_INFO("mosquitto_destroy()'ed");
  }
  mosquitto_lib_cleanup();
  SPDLOG_INFO("mosquitto_lib_cleanup()'ed");
}
} // namespace MatrixPipeline::ProcessingUnit
