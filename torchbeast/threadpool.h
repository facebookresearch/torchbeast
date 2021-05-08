/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <thread>

class ThreadPool {
 public:
  ThreadPool(uint num_threads) {
    threads_.reserve(num_threads);
    for (uint i = 0; i < num_threads; ++i) {
      threads_.emplace_back(&ThreadPool::thread_run, this, i);
    }
  }

  ~ThreadPool() {
    if (!threads_.empty()) {
      std::terminate();
    }
  }

  void submit(std::function<void()> task) {
    if (!task) return;
    {
      std::unique_lock<std::mutex> lock(mu_);
      tasks_.push_back(std::move(task));
    }
    cv_.notify_one();
  }

  void close() {
    {
      std::unique_lock<std::mutex> lock(mu_);
      tasks_.push_back(nullptr);
    }
    cv_.notify_all();
    for (std::thread& thread : threads_) {
      thread.join();
    }
    threads_.clear();
  }

 private:
  void thread_run(uint i) {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        while (tasks_.empty()) cv_.wait(lock);
        if (!tasks_.front()) return;
        task = std::move(tasks_.front());
        tasks_.pop_front();
      }
      task();
    }
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::thread> threads_;
  std::deque<std::function<void()>> tasks_;
};
