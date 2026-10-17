#include <chrono>
#include <tokenizers_cpp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include "kernels.cuh"
#include "config.h"
#include "runtime.h"
#include "model.h"
#include "score.h"
#include "telemetry.h"
#include "queue.h"

// Global thread-safe queue & engine state flag
RequestQueue g_request_queue;
std::atomic<bool> g_engine_running{true};

void run_engine_loop (
  const Config& cfg,
  const ModelConfig& model_cfg,
  Weights& weights,
  cublasHandle_t cublas_handle,
  Buffers& buf,
  tokenizers::Tokenizer* tokenizer
) {
  std::vector<SlotState> slots(MAX_SEQUENCES);

  auto retire_slot = [&](int s, const char* reason) {
    std::string final_text = tokenizer->Decode(slots[s].generated);

    // signals client thread that generation is done for this req
    if (slots[s].on_token_callback) {
      slots[s].on_token_callback("", true);
    }

    std::cerr << "[Engine] Slot " << s << " finished (" << reason << ")\n";
    slots[s] = SlotState{}; // clear slot state
  };


  while(g_engine_running) {
    // Ingesting incoming requests
    for(int s{0}; s < MAX_SEQUENCES; s++) {
      if(slots[s].active) continue;

      ClientRequest req;
      // Non-blocking check for new requests
      if(!g_request_queue.pop(req, false)) break;

      int prompt_len = req.tokens.size();

      prefill(req.tokens, prompt_len, s, cfg, weights, cublas_handle, buf, slots);

      if(!slots[s].active) continue;

      slots[s].id = req.id;
      slots[s].prompt_text = req.raw_text;
      slots[s].on_token_callback = req.on_token; // Store callback
      
      std::cerr << "[Engine] Slot " << s << " prefilled " << prompt_len << " tokens\n";

      const char* reason = finishReason(slots[s], req.max_tokens);
      if(reason) retire_slot(s, reason);
    }

    int num_active = 0;
    for(const SlotState& s: slots) {
      if(s.active) num_active++;
    }

    if(num_active == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }

    decode(slots, cfg, weights, cublas_handle, buf);

    for(int s{0}; s < MAX_SEQUENCES; s++) {
      if(!slots[s].active) continue;

      // Stream last token to the request callback
      if(!slots[s].generated.empty() && slots[s].on_token_callback) {
        int last_token_id = slots[s].generated.back();
        std::string token_str = tokenizer->Decode({last_token_id});
        slots[s].on_token_callback(token_str, false);
      }

      const char* reason = finishReason(slots[s], cfg.max_new_tokens);
      if(reason) retire_slot(s, reason);
    }
  }
}

// Load Hugging Face tokenizer.json
std::unique_ptr<tokenizers::Tokenizer> loadNativeTokenizer(const std::string& tokenizer_json_path) {
  std::ifstream file(tokenizer_json_path);
  if(!file.is_open()) {
    throw std::runtime_error("Failed to open tokenizer file: " + tokenizer_json_path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();

  return tokenizers::Tokenizer::FromBlobJSON(buffer.str());
}

int main(int argc, char* argv[]) {
  Config cfg;
  ParseResult parsed = parseArgs(argc, argv, cfg);
  if(parsed == ParseResult::Exit) return 0;
  if(parsed == ParseResult::Error) return 1;

  if(!cfg.seeded) cfg.seed = std::random_device{}();
  seedSampler(cfg.seed);
  std::cerr << "seed " << cfg.seed << ", top_k " << cfg.top_k
            << ", temperature " << cfg.temperature
            << ", max_new_tokens " << cfg.max_new_tokens << "\n";

  std::ofstream output_file;
  std::ostream* out = &std::cout;
  if(!cfg.output_path.empty()) {
    output_file.open(cfg.output_path);
    if(!output_file.is_open()) {
      std::cerr << "Cannot open output file " << cfg.output_path << "\n";
      return 1;
    }
    out = &output_file;
  }

  auto tokenizer = loadNativeTokenizer(cfg.tokenizer_path);

  ModelConfig model_cfg;
  if(loadModelConfig(model_cfg, cfg.model_path)) return 1;
  setStopTokens(model_cfg.stop_token_ids);
  setRmsNormEps(model_cfg.rms_norm_eps);
  
  Weights weights;
  if(loadWeights(weights, cfg.model_path)) return 1;
  
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  
  init_rope_frequencies(model_cfg.head_dim, MAX_SEQ_LEN, model_cfg.rope_theta,
    model_cfg.rope_factor, model_cfg.rope_low_freq_factor,
    model_cfg.rope_high_freq_factor,
    model_cfg.rope_original_max_position);
    
    Buffers buf;
    if(buf.allocate(cfg.score_mode)) return 1;
    
    if(cfg.score_mode) {
      std::cerr << "Running in Scoring Mode\n";
      Recorder recorder(*out);
      
      int prompt_idx = 0;
      for(const auto& prompt_text: cfg.prompts) {
        std::string id = std::to_string(prompt_idx++);

        std::vector<int> tokens = tokenizer->Encode(prompt_text);
        if(tokens.empty() || tokens[0] != model_cfg.bos_token_id) {
          tokens.insert(tokens.begin(), model_cfg.bos_token_id);
        }
        
        int len = (int)tokens.size();
        if(len > MAX_PROMPT_LEN) {
          std::cerr << cfg.prompts_path.string() << ": prompt " << id << " encodes to "
                    << len << " tokens, exceeds MAX_PROMPT_LEN of " << MAX_PROMPT_LEN << "\n";
          return 1;
        }
        if(cfg.score_mode && len < 2) {
          std::cerr << cfg.prompts_path.string() << ": prompt " << id << " encodes to "
                    << len << " tokens, scoring needs at least 2\n";
          return 1;
        }


      int num_scored = scoreSequence(tokens, id, weights, cublas_handle, buf, *out);
      if(num_scored > 0) recorder.onScore(num_scored);
    }

    cudaDeviceSynchronize();
    buf.free();
    free_rope_frequencies();
    cublasDestroy(cublas_handle);
    return 0;
  }

  // Spawn prefill/decode loop thread
  std::thread engine_thread(run_engine_loop, std::ref(cfg), std::ref(model_cfg), std::ref(weights), cublas_handle, std::ref(buf), tokenizer.get());
  
  std::cerr << "Engine started. Type a prompt and press Enter (or 'exit' to quit):\n";

  // Request handler (terminal stdin for now)
  std::string input_line;
  int req_id = 0;
  while (std::getline(std::cin, input_line)) {
    if(input_line == "exit" || input_line == "quit") break;
    if(input_line.empty()) continue;

    ClientRequest req;
    req.id = std::to_string(req_id++);
    req.raw_text = input_line;
    req.tokens = tokenizer->Encode(input_line);

    if(req.tokens.empty() || req.tokens[0] != model_cfg.bos_token_id) {
      req.tokens.insert(req.tokens.begin(), model_cfg.bos_token_id);
    }

    req.max_tokens = cfg.max_new_tokens;

    // what happens when a new token is generated for this request is below
    req.on_token = [](const std::string& token_text, bool is_finished) {
      if(!is_finished) {
        std::cout << token_text << std::flush;
      } else {
        std::cout << "\n> " << std::flush;
      }
    };

    g_request_queue.push(std::move(req));
  }

  std::cerr << "\nShutting down engine\n";
  g_engine_running = false;

  // Pushing an empty dummy request to wake up the queue if it is sleeping
  g_request_queue.push(ClientRequest{});
  engine_thread.join();

  cudaDeviceSynchronize();

  buf.free();
  free_rope_frequencies();
  cublasDestroy(cublas_handle);

  return 0;
}
