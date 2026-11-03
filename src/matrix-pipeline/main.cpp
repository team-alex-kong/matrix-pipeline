#include "global_vars.h"
#include "utils/misc.h"
#include "video_feed_manager.h"

#include <cxxopts.hpp>
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace drogon;
using namespace std;
using json = nlohmann::json;

string config_path =
    string(getenv("HOME")) + "/.config/ak-studio/cuda-motion.jsonc";

void signal_handler_cb(__attribute__((unused)) int signum) {
  drogon::app().quit();
  ev_flag = 1;
}

void configure_spdlog() {

  // Define the queue size (the buffer for log messages)
  // A larger queue prevents blocking if the worker thread falls behind
  // temporarily.
  constexpr size_t queue_size = 9527;
  constexpr size_t thread_count = 1; // One thread is usually enough for logging
  spdlog::init_thread_pool(queue_size, thread_count);
  try {
    // 1. Define the sinks (destinations)
    const auto console_sink =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    std::vector<spdlog::sink_ptr> sinks{console_sink};

    // 2. Create the async logger
    // Note the use of create_async
    const auto async_logger = std::make_shared<spdlog::async_logger>(
        "matrix-pipeline", sinks.begin(), sinks.end(),
        spdlog::thread_pool(), // Pass the globally initialized thread pool
        spdlog::async_overflow_policy::overrun_oldest // Choose the overflow
                                                      // policy
    );

    // 3. Register it globally
    spdlog::set_default_logger(async_logger);

    // 4. Set pattern and level
    spdlog::set_level(spdlog::level::info);
    // Doc: https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
    // Including microseconds is handy for naive profiling
    spdlog::set_pattern("%Y-%m-%dT%T.%e | %5t | %8l | %30s:%-3#:%-12! | %v");

  } catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "Log initialization failed: " << ex.what() << std::endl;
  }
}

int main(int argc, char *argv[]) {
  cxxopts::Options options(argv[0], "video feed handler that uses CUDA");
  // clang-format off
  options.add_options()
    ("h,help", "print help message")
    ("c,config-path", "path of the config file", cxxopts::value<string>()->default_value(config_path));
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  config_path = result["config-path"].as<std::string>();

  configure_spdlog();

  SPDLOG_INFO("matrix-pipeline started (git commit: {})", GIT_COMMIT_HASH);
  MatrixPipeline::Utils::install_signal_handler(signal_handler_cb);
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  SPDLOG_INFO("cv::getBuildInformation(): {}",
              string(cv::getBuildInformation()));

  SPDLOG_INFO("Loading json settings from {}", config_path);
  ifstream is(config_path);
  try {
    settings = json::parse(is, nullptr, true, true);
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to parse json from {}: {}", config_path, e.what());
    return EXIT_FAILURE;
  }

  auto mgr = std::make_shared<MatrixPipeline::VideoFeedManager>();
  if (!mgr->init()) {
    SPDLOG_ERROR("Failed to initialize video feed manager");
    return EXIT_FAILURE;
  }

  SPDLOG_INFO("Starting Drogon web server");
  auto th_drogon = std::thread([] {
    app()
        .setLogLevel(trantor::Logger::kWarn)
        .setThreadNum(4)
        .disableSigtermHandling()
        .run();
  });

  SPDLOG_INFO("Starting VideoFeedManager's event loop thread");
  mgr->feed_capture_ev();
  SPDLOG_INFO("VideoFeedManager's event loop exited gracefully");

  th_drogon.join();
  SPDLOG_INFO("Drogon exited");

  SPDLOG_INFO("matrix-pipeline will now exit gracefully");
  return EXIT_SUCCESS;
}
