#include "misc.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <signal.h>
#include <time.h>

using namespace std;

namespace MatrixPipeline::Utils {

signal_handler_callback sh_callback;

atomic<ssize_t> executionCounter = -1;

static void signal_handler(int signum) noexcept {
  if (signum == SIGCHLD) {
    // When a child process stops or terminates, SIGCHLD is sent to the parent
    // process. The default response to the signal is to ignore it.
    return;
  }
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + signum / 10;
  msg[9] = '0' + signum % 10;
  size_t len = sizeof(msg) - 1;
  size_t written = 0;
  while (written < len) {
    ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
    if (ret == -1)
      break;
    written += ret;
  }
  sh_callback(signum);
}

void install_signal_handler(const signal_handler_callback cb) {
  static_assert(_NSIG < 99,
                "signal_handler() can't handle more than 99 signals");

  sh_callback = cb;
  struct sigaction act;
  // Initialize the signal set to empty, similar to memset(0)
  if (sigemptyset(&act.sa_mask) == -1) {
    perror("sigemptyset()");
    abort();
  }
  act.sa_handler = signal_handler;
  // writers handle EPIPE themselves
  signal(SIGPIPE, SIG_IGN);
  if (sigaction(SIGINT, &act, nullptr) + sigaction(SIGABRT, &act, nullptr) +
          sigaction(SIGQUIT, &act, nullptr) +
          sigaction(SIGTERM, &act, nullptr) +
          sigaction(SIGCHLD, &act, nullptr) +
          sigaction(SIGTRAP, &act, nullptr) <
      0) {
    throw runtime_error("sigaction() called failed: " + to_string(errno) + "(" +
                        strerror(errno) + ")");
  }
}

std::string hybrid_njson_array_dump(const njson &j_array) {
  if (!j_array.is_array())
    return j_array.dump();

  std::string out = "[\n";
  for (size_t i = 0; i < j_array.size(); ++i) {
    out += fmt::format("  {}{}\n", j_array[i].dump(),
                       (i < j_array.size() - 1 ? ", " : ""));
  }
  out += "]";
  return out;
}

} // namespace MatrixPipeline::Utils