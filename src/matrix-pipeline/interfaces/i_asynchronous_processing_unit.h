#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"
#include "../global_vars.h"
#include "../utils/matrix_sender.h"
#include "i_processing_unit.h"

#include <boost/stacktrace.hpp>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace MatrixPipeline::Utils {

class NvJpegEncoder;
}

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

struct AsyncPayload {
  cv::cuda::GpuMat frame;
  PipelineContext ctx;
};

class IAsynchronousProcessingUnit : public IProcessingUnit {
protected:
  // const std::string m_unit_path;
  std::array<bool, 24> m_turned_on_hours{
      true, true, true, true, true, true, true, true, true, true, true, true,
      true, true, true, true, true, true, true, true, true, true, true, true};

public:
  ~IAsynchronousProcessingUnit() override {
    stop();
    SPDLOG_INFO("processing_unit {}::m_worker_thread stop()ed", m_unit_path);
  }

  explicit IAsynchronousProcessingUnit(const std::string &unit_path)
      : IProcessingUnit(unit_path) {};
  /**
   * @brief Pushes a frame into the processing queue.
   * * IMPORTANT: This performs a DEEP COPY (clone) of the GPU frame.
   * This ensures the worker thread owns independent data, allowing the caller
   * to reuse or destroy the original frame immediately after this call returns.
   * * Note: This involves GPU memory allocation, so it has a higher cost than a
   * shallow copy.
   */
  SynchronousProcessingResult enqueue(const cv::cuda::GpuMat &frame,
                                      const PipelineContext &ctx) {
    if (!m_turned_on_hours[get_hours_from_local_time()])
      return success_and_continue;
    if (ev_flag != 0)
      return failure_and_stop;

    using namespace std::chrono_literals;
    cv::cuda::GpuMat frame_clone;
    try {
      frame_clone = frame.clone();
    } catch (const cv::Exception &e) {
      SPDLOG_WARN(
          "e.what(): {}\n{}", e.what(),
          boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
      return failure_and_continue;
    }
    {
      std::lock_guard lock(m_queue_mutex);

      auto queue_size = m_processing_queue.size();
      if (constexpr auto warning_queue_size = 10;
          queue_size > warning_queue_size) {
        constexpr auto warning_throttle_interval = 5s;
        if (std::chrono::steady_clock::now() - m_last_warning_time >
            warning_throttle_interval) {
          SPDLOG_WARN("{}: queue_size ({}) is above warning_queue_size ({}). "
                      "(This message is throttled to once per {} sec)",
                      m_unit_path, queue_size, warning_queue_size,
                      warning_throttle_interval.count());
          m_last_warning_time = std::chrono::steady_clock::now();
        }
        if (constexpr auto critical_queue_size = 30;
            queue_size > critical_queue_size) {
          SPDLOG_ERROR("{}: queue_size ({}) is above critical_queue_size ({}), "
                       "discard {} frames to avoid OOM",
                       m_unit_path, queue_size, critical_queue_size,
                       queue_size - warning_queue_size);
          while (queue_size > warning_queue_size) {
            // auto [frame, ctx] = m_processing_queue.front();
            m_processing_queue.pop();
            queue_size = m_processing_queue.size();
          }
        }
      }

      m_processing_queue.push(AsyncPayload{frame_clone, ctx});
    }

    // 3. Wake up the worker thread
    m_cv.notify_one();
    return success_and_continue;
  }

  /**
   * @brief Starts the internal worker thread.
   */
  void start() {
    if (m_running.load()) {
      return; // Already running
    }
    m_running.store(true);
    m_worker_thread =
        std::thread(&IAsynchronousProcessingUnit::dequeue_loop, this);
    SPDLOG_INFO("asynchronous_processing_unit {} started", m_unit_path);
  }

  /**
   * @brief Stops the worker thread and waits for it to finish.
   */
  void stop() {
    if (!m_running.load()) {
      return;
    }

    m_running.store(false);
    m_cv.notify_all(); // Wake up thread if it is sleeping

    if (m_worker_thread.joinable()) {
      m_worker_thread.join();
    }
  }

protected:
  /**
   * @brief The actual processing logic to be implemented by concrete classes.
   * This is called automatically by the internal thread when data is dequeued.
   */
  virtual void on_frame_ready(cv::cuda::GpuMat &frame,
                              PipelineContext &ctx) = 0;

private:
  /**
   * @brief The internal thread loop.
   * Pops items from the queue and delegates to process_frame().
   */
  void dequeue_loop() {
    while (m_running.load() || !is_queue_empty()) {
      AsyncPayload payload;

      {
        std::unique_lock lock(m_queue_mutex);
        m_cv.wait(lock, [this] {
          return !m_processing_queue.empty() || !m_running.load();
        });
        // empty here implies !m_running
        if (m_processing_queue.empty())
          break;
        payload = m_processing_queue.front();
        m_processing_queue.pop();
      }
      try {
        on_frame_ready(payload.frame, payload.ctx);
      } catch (const std::exception &e) {
        SPDLOG_ERROR("std::exception from {}: e.what(): {}", m_unit_path,
                     e.what());
        disable();
      }
    }
  }

  bool is_queue_empty() {
    std::lock_guard<std::mutex> lock(m_queue_mutex);
    return m_processing_queue.empty();
  }

  std::queue<AsyncPayload> m_processing_queue;
  std::mutex m_queue_mutex;
  std::condition_variable m_cv;
  std::atomic<bool> m_running{false};
  std::thread m_worker_thread;
  std::chrono::steady_clock::time_point m_last_warning_time;

  static int get_hours_from_local_time() {
    const auto local_datetime = std::chrono::zoned_time{
        std::chrono::current_zone(), std::chrono::system_clock::now()};
    const auto lt = local_datetime.get_local_time();
    // remove the date part
    const std::chrono::hh_mm_ss time_of_day{
        lt - std::chrono::floor<std::chrono::days>(lt)};
    return time_of_day.hours().count();
  }
};

} // namespace MatrixPipeline::ProcessingUnit