// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_RUNTIME_DEBUGGER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_RUNTIME_DEBUGGER_H_

#include <atomic>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/io_types.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"

namespace litert::lm {

// RuntimeDebugger manages diagnostic tracing, intermediate layer tensor
// capturing, and step-by-step token observation during model inference.
class RuntimeDebugger {
 public:
  // Monitoring phase for tensor observation.
  enum class MonitorPhase {
    kPre,
    kPost,
  };

  // Creates and initializes a RuntimeDebugger instance under the given cache
  // directory, or returns nullptr if cache dir is invalid or in-memory.
  static std::unique_ptr<RuntimeDebugger> Create(
      absl::string_view preferred_cache_dir = "");

  explicit RuntimeDebugger(absl::string_view capture_dir);
  ~RuntimeDebugger() = default;

  // Returns a callback for capturing tensor buffers BEFORE each
  // computation graph execution for LiteRT compiled model executors.
  LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
  CreatePreGraphRunCallback();

  // Returns a callback for capturing tensor buffers AFTER each
  // computation graph execution for LiteRT compiled model executors.
  LlmLiteRtCompiledModelExecutorBase::GraphRunCallback
  CreatePostGraphRunCallback();

  // Registers a session-specific capture directory.
  void RegisterSession(int session_id,
                       absl::string_view preferred_cache_dir = "");

  // Unregisters a session-specific capture directory.
  void UnregisterSession(int session_id);

  // Sets the active session for subsequent computation graph runs.
  void SetActiveSession(int session_id);

  // Observes generated tokens for a session and appends them to disk in JSONL
  // format.
  void ObserveTokens(int session_id, const Responses& responses);

  const std::string& capture_dir() const { return capture_dir_; }

 private:
  struct SessionContext {
    std::string capture_dir;
    int last_step = 0;
    absl::flat_hash_map<std::string, ::litert::TensorBuffer> kv_buffers;
  };

  // Captures tensor buffers to disk in Safetensors format.
  void CaptureTensors(
      MonitorPhase phase, absl::string_view signature_name, int current_step,
      const absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
          tensors);

  void WriteTokensToFile(absl::string_view target_dir,
                         const Responses& responses);

  std::string capture_dir_;

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<int, std::unique_ptr<SessionContext>> sessions_
      ABSL_GUARDED_BY(mutex_);
  std::atomic<SessionContext*> cur_session_{nullptr};
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_RUNTIME_DEBUGGER_H_
