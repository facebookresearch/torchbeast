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

#include <atomic>

#include <grpc++/grpc++.h>

#include <torch/script.h>

#include "rpc.grpc.pb.h"
#include "rpc.pb.h"

#include "threadpool.h"

namespace torchbeast {
class Server {
  typedef torch::jit::script::Module ScriptModule;
  typedef std::function<CallResponse(CallRequest *)> Function;

  class ServiceImpl final : public RPC::Service {
   public:
    grpc::Status bind(const std::string &name, Function &&function);

   private:
    virtual grpc::Status Call(
        grpc::ServerContext *context,
        grpc::ServerReaderWriter<CallResponse, CallRequest> *stream) override;

    std::map<std::string, Function> functions_;
  };

 public:
  Server(const std::string &address, uint max_parallel_calls = 2)
      : address_(address), server_(nullptr), threadpool_(max_parallel_calls) {}

  ~Server() { threadpool_.close(); }

  void run();
  void wait();
  void stop();

  bool running() { return running_.load(); }

  void bind(const std::string &name, const py::object &obj,
            std::optional<int32_t> batch_size);

 private:
  const std::string address_;
  ServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;

  std::atomic_bool running_ = false;
  ThreadPool threadpool_;
};

}  // namespace torchbeast
