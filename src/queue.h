#ifndef QUEUE_H
#define QUEUE_H

#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <vector>
#include <functional>

struct ClientRequest {
    std::string id;
    std::vector<int> tokens;
    std::string raw_text;
    float temperature = 0.8f;
    int top_k = 40;
    int max_tokens = 512;

    // Callback to stream tokens back to the request handler thread
    std::function<void(const std::string& token_text, bool is_finished)> on_token;
};

class RequestQueue {
private:
    std::queue<ClientRequest> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;

public:
    void push(ClientRequest req) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(req));
        }
        cv_.notify_one();
    }

    bool pop(ClientRequest& req, bool block = true) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (block) {
            cv_.wait(lock, [this] { return !queue_.empty();});
        } else if (queue_.empty()) {
            return false;
        }

        req = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

#endif // QUEUE_H