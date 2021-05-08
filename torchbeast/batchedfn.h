// Copyright 2019 The SEED Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copied from
 *   https://raw.githubusercontent.com/google-research/seed_rl/master/grpc/ops/grpc.cc
 * and modified. */

#include <torch/extension.h>

#include "notification.h"
#include "threadpool.h"

#include "../third_party/nest/nest/nest.h"

typedef nest::Nest<at::Tensor> TensorNest;

struct ServerClosed : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class BatchedFn {
  typedef std::function<TensorNest(TensorNest)> TensorFunction;

 public:
  BatchedFn(TensorFunction&& function, int32_t batch_size,
            std::atomic_bool* running, ThreadPool* threadpool)
      : function_(std::move(function)),
        batch_size_(batch_size),
        server_running_(running),
        threadpool_(threadpool),
        mu_(new std::mutex()) {}

  TensorNest operator()(TensorNest args) {
    int64_t index;
    std::shared_ptr<Computation> computation;
    {
      std::unique_lock lock(*mu_);
      index = next_index_++;

      if (index == 0) {
        TORCH_CHECK(current_computation_ == nullptr,
                    "Expected empty computation");
        current_computation_ = std::make_shared<Computation>();
        current_computation_->request =
            args.map([batch_size = batch_size_](const at::Tensor& t) {
              c10::IntArrayRef sizes = t.sizes();
              std::vector<int64_t> shape = {batch_size};
              shape.insert(shape.end(), sizes.begin(), sizes.end());

              return torch::empty(shape, t.dtype());
            });
      }

      computation = current_computation_;

      if (index == batch_size_ - 1) {
        next_index_ = 0;
        current_computation_.reset();
      }
    }

    // Copy input tensors to the batched input tensors.
    TensorNest::for_each(
        [index](at::Tensor& request, const at::Tensor& arg) {
          request[index] = arg;
        },
        computation->request, args);

    int num_ready = ++computation->num_ready;
    if (num_ready == batch_size_) {
      // A full batch has been filled up, so the function should be executed.

      // TODO(heiner): We should be able to cancel this if it never returns?
      threadpool_->submit([computation, this]() {
        try {
          computation->outputs = function_(computation->request);
        } catch (const std::exception&) {
          computation->error = std::current_exception();
        }
        computation->done.Notify();
      });
    }

    // Wait for the function to run/finish.
    while (!WaitForNotificationWithTimeout(&computation->done,
                                           50000 /* 50 ms */)) {
      if (!server_running_->load()) throw ServerClosed("Server closed");
    }

    // TODO(heiner): Handle cancellation?
    if (computation->error) std::rethrow_exception(computation->error);

    return computation->outputs.map(
        [index](const at::Tensor& t) { return t[index]; });
  }

 private:
  // Represents one batched computation.
  struct Computation {
    TensorNest request;
    TensorNest outputs;
    Notification done;
    std::atomic_int num_ready{0};
    std::exception_ptr error;
  };

  TensorFunction function_;
  const int32_t batch_size_;
  std::atomic_bool* server_running_;
  ThreadPool* threadpool_;

  // HACK: A shared_ptr to make type copyable for std::function.
  std::shared_ptr<std::mutex> mu_;
  int64_t next_index_ = 0;                            // GUARDED_BY(mu_)
  std::shared_ptr<Computation> current_computation_;  // GUARDED_BY(mu_);
};
