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

#include <atomic>
#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include <grpc++/grpc++.h>

// Enable including numpy via numpy_stub.h.
#define USE_NUMPY 1

#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "nest_serialize.h"
#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "nest/nest.h"
#include "nest/nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<torch::Tensor> TensorNest;

TensorNest batch(const std::vector<TensorNest>& tensors, int64_t batch_dim) {
  // TODO(heiner): Consider using accessors and writing slices ourselves.
  nest::Nest<std::vector<torch::Tensor>> zipped = TensorNest::zip(tensors);
  return zipped.map([batch_dim](const std::vector<torch::Tensor>& v) {
    return torch::cat(v, batch_dim);
  });
}

struct ClosedBatchingQueue : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Enable a few standard Python exceptions.
namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(runtime_error, PyExc_RuntimeError)
PYBIND11_RUNTIME_EXCEPTION(timeout_error, PyExc_TimeoutError)
PYBIND11_RUNTIME_EXCEPTION(connection_error, PyExc_ConnectionError)
}  // namespace pybind11

struct Empty {};

template <typename T = Empty>
class BatchingQueue {
 public:
  struct QueueItem {
    TensorNest tensors;
    T payload;
  };
  BatchingQueue(int64_t batch_dim, int64_t minimum_batch_size,
                int64_t maximum_batch_size,
                std::optional<int> timeout_ms = std::nullopt,
                bool check_inputs = true,
                std::optional<uint64_t> maximum_queue_size = std::nullopt)
      : batch_dim_(batch_dim),
        minimum_batch_size_(
            minimum_batch_size > 0
                ? minimum_batch_size
                : throw py::value_error("Min batch size must be >= 1")),
        maximum_batch_size_(
            maximum_batch_size >= minimum_batch_size
                ? maximum_batch_size
                : throw py::value_error(
                      "Max batch size must be >= min batch size")),
        timeout_(timeout_ms),
        maximum_queue_size_(maximum_queue_size),
        check_inputs_(check_inputs) {
    if (maximum_queue_size_ != std::nullopt &&
        *maximum_queue_size_ < maximum_batch_size_) {
      throw py::value_error("Max queue size must be >= max batch size");
    }
  }

  int64_t size() const {
    std::unique_lock<std::mutex> lock(mu_);
    return deque_.size();
  }

  void enqueue(QueueItem item) {
    if (check_inputs_) {
      bool is_empty = true;

      item.tensors.for_each([this, &is_empty](const torch::Tensor& tensor) {
        is_empty = false;

        if (tensor.dim() <= batch_dim_) {
          throw py::value_error(
              "Enqueued tensors must have more than batch_dim == " +
              std::to_string(batch_dim_) + " dimensions, but got " +
              std::to_string(tensor.dim()));
        }
      });

      if (is_empty) {
        throw py::value_error("Cannot enqueue empty vector of tensors");
      }
    }

    bool should_notify = false;
    {
      std::unique_lock<std::mutex> lock(mu_);
      // Block when maximum_queue_size is reached.
      while (maximum_queue_size_ != std::nullopt && !is_closed_ &&
             deque_.size() >= *maximum_queue_size_) {
        can_enqueue_.wait(lock);
      }
      if (is_closed_) {
        throw ClosedBatchingQueue("Enqueue to closed queue");
      }
      deque_.push_back(std::move(item));
      should_notify = deque_.size() >= minimum_batch_size_;
    }

    if (should_notify) {
      enough_inputs_.notify_one();
    }
  }

  std::pair<TensorNest, std::vector<T>> dequeue_many() {
    std::vector<TensorNest> tensors;
    std::vector<T> payloads;
    {
      std::unique_lock<std::mutex> lock(mu_);

      bool timed_out = false;
      while (!is_closed_ &&
             (deque_.empty() ||
              (!timed_out && deque_.size() < minimum_batch_size_))) {
        if (timeout_ == std::nullopt) {
          // If timeout_ isn't set, stop waiting when:
          // - queue is closed, or
          // - we have enough inputs inside the queue.
          enough_inputs_.wait(lock);
        } else {
          // If timeout_ is set, stop waiting when:
          // - queue is closed, or
          // - we timed out and have at least one input, or
          // - we have enough inputs in the queue.
          timed_out = (enough_inputs_.wait_for(lock, *timeout_) ==
                       std::cv_status::timeout);
        }
      }

      if (is_closed_) {
        throw py::stop_iteration("Queue is closed");
      }
      const int64_t batch_size =
          std::min<int64_t>(deque_.size(), maximum_batch_size_);
      for (auto it = deque_.begin(), end = deque_.begin() + batch_size;
           it != end; ++it) {
        tensors.push_back(std::move(it->tensors));
        payloads.push_back(std::move(it->payload));
      }
      deque_.erase(deque_.begin(), deque_.begin() + batch_size);
    }
    can_enqueue_.notify_all();
    return std::make_pair(batch(tensors, batch_dim_), std::move(payloads));
  }

  bool is_closed() const {
    std::unique_lock<std::mutex> lock(mu_);
    return is_closed_;
  }

  void close() {
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (is_closed_) {
        throw py::runtime_error("Queue was closed already");
      }
      is_closed_ = true;
      deque_.clear();
    }
    enough_inputs_.notify_all();  // Wake up dequeues.
    can_enqueue_.notify_all();
  }

 private:
  mutable std::mutex mu_;

  const int64_t batch_dim_;
  const uint64_t minimum_batch_size_;
  const uint64_t maximum_batch_size_;
  const std::optional<std::chrono::milliseconds> timeout_;
  const std::optional<uint64_t> maximum_queue_size_;

  std::condition_variable enough_inputs_;
  std::condition_variable can_enqueue_;

  bool is_closed_ = false /* GUARDED_BY(mu_) */;
  std::deque<QueueItem> deque_ /* GUARDED_BY(mu_) */;

  const bool check_inputs_;
};

class DynamicBatcher {
 public:
  typedef std::promise<std::pair<std::shared_ptr<TensorNest>, int64_t>>
      BatchPromise;
  class Batch {
   public:
    Batch(int64_t batch_dim, TensorNest&& tensors,
          std::vector<BatchPromise>&& promises, bool check_outputs)
        : batch_dim_(batch_dim),
          inputs_(std::move(tensors)),
          promises_(std::move(promises)),
          check_outputs_(check_outputs) {}

    const TensorNest& get_inputs() { return inputs_; }

    void set_outputs(TensorNest outputs) {
      if (promises_.empty()) {
        // Batch has been set before.
        throw py::runtime_error("set_outputs called twice");
      }

      if (check_outputs_) {
        const int64_t expected_batch_size = promises_.size();

        outputs.for_each([this,
                          expected_batch_size](const torch::Tensor& tensor) {
          if (tensor.dim() <= batch_dim_) {
            std::stringstream ss;
            ss << "With batch dimension " << batch_dim_
               << ", output shape must have at least " << batch_dim_ + 1
               << " dimensions, but got " << tensor.sizes();
            throw py::value_error(ss.str());
          }
          if (tensor.sizes()[batch_dim_] != expected_batch_size) {
            throw py::value_error(
                "Output shape must have the same batch "
                "dimension as the input batch size. Expected: " +
                std::to_string(expected_batch_size) +
                ". Observed: " + std::to_string(tensor.sizes()[batch_dim_]));
          }
        });
      }

      auto shared_outputs = std::make_shared<TensorNest>(std::move(outputs));

      int64_t b = 0;
      for (auto& promise : promises_) {
        promise.set_value(std::make_pair(shared_outputs, b));
        ++b;
      }
      promises_.clear();
    }

   private:
    const int64_t batch_dim_;
    const TensorNest inputs_;
    std::vector<BatchPromise> promises_;

    const bool check_outputs_;
  };

  DynamicBatcher(int64_t batch_dim, int64_t minimum_batch_size,
                 int64_t maximum_batch_size,
                 std::optional<int> timeout_ms = std::nullopt,
                 bool check_outputs = true)
      : batching_queue_(batch_dim, minimum_batch_size, maximum_batch_size,
                        timeout_ms),
        batch_dim_(batch_dim),
        check_outputs_(check_outputs) {}

  TensorNest compute(TensorNest tensors) {
    BatchPromise promise;
    auto future = promise.get_future();

    batching_queue_.enqueue({std::move(tensors), std::move(promise)});

    std::future_status status = future.wait_for(std::chrono::seconds(10 * 60));
    if (status != std::future_status::ready) {
      throw py::timeout_error("Compute timeout reached.");
    }

    const std::pair<std::shared_ptr<TensorNest>, int64_t> pair = [&] {
      try {
        return future.get();
      } catch (const std::future_error& e) {
        if (batching_queue_.is_closed() &&
            e.code() == std::future_errc::broken_promise) {
          throw ClosedBatchingQueue("Batching queue closed during compute");
        }
        throw;
      }
    }();

    return pair.first->map([batch_dim = batch_dim_,
                            batch_entry = pair.second](const torch::Tensor& t) {
      return t.slice(batch_dim, batch_entry, batch_entry + 1);
    });
  }

  std::shared_ptr<Batch> get_batch() {
    auto pair = batching_queue_.dequeue_many();
    return std::make_shared<Batch>(batch_dim_, std::move(pair.first),
                                   std::move(pair.second), check_outputs_);
  }

  int64_t size() const { return batching_queue_.size(); }

  void close() { batching_queue_.close(); }
  bool is_closed() { return batching_queue_.is_closed(); }

 private:
  BatchingQueue<std::promise<std::pair<std::shared_ptr<TensorNest>, int64_t>>>
      batching_queue_;
  int64_t batch_dim_;

  bool check_outputs_;
};

class ActorPool {
 public:
  ActorPool(int unroll_length, std::shared_ptr<BatchingQueue<>> learner_queue,
            std::shared_ptr<DynamicBatcher> inference_batcher,
            std::vector<std::string> env_server_addresses,
            TensorNest initial_agent_state)
      : unroll_length_(unroll_length),
        learner_queue_(std::move(learner_queue)),
        inference_batcher_(std::move(inference_batcher)),
        env_server_addresses_(std::move(env_server_addresses)),
        initial_agent_state_(std::move(initial_agent_state)) {}

  void loop(int64_t loop_index, const std::string& address) {
    std::shared_ptr<grpc::Channel> channel =
        grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    std::unique_ptr<rpcenv::RPCEnvServer::Stub> stub =
        rpcenv::RPCEnvServer::NewStub(channel);

    auto deadline =
        std::chrono::system_clock::now() + std::chrono::seconds(10 * 60);

    if (loop_index == 0) {
      std::cout << "First Environment waiting for connection to " << address
                << " ...";
    }
    if (!channel->WaitForConnected(deadline)) {
      throw py::timeout_error("WaitForConnected timed out.");
    }
    if (loop_index == 0) {
      std::cout << " connection established." << std::endl;
    }

    grpc::ClientContext context;
    std::shared_ptr<grpc::ClientReaderWriter<rpcenv::Action, rpcenv::Step>>
        stream(stub->StreamingEnv(&context));

    rpcenv::Step step_pb;
    if (!stream->Read(&step_pb)) {
      throw py::connection_error("Initial read failed.");
    }

    TensorNest initial_agent_state = initial_agent_state_;

    TensorNest env_outputs = ActorPool::step_pb_to_nest(&step_pb);
    TensorNest compute_inputs(std::vector({env_outputs, initial_agent_state}));
    TensorNest all_agent_outputs =
        inference_batcher_->compute(compute_inputs);  // Copy.

    // Check this once per thread.
    if (!all_agent_outputs.is_vector()) {
      throw py::value_error("Expected agent output to be tuple");
    }
    if (all_agent_outputs.get_vector().size() != 2) {
      throw py::value_error(
          "Expected agent output to be ((action, ...), new_state) but got "
          "sequence of "
          "length " +
          std::to_string(all_agent_outputs.get_vector().size()));
    }
    TensorNest agent_state = all_agent_outputs.get_vector()[1];
    TensorNest agent_outputs = all_agent_outputs.get_vector()[0];
    if (!agent_outputs.is_vector()) {
      throw py::value_error(
          "Expected first entry of agent output to be a (action, ...) tuple");
    }

    TensorNest last(std::vector({env_outputs, agent_outputs}));

    rpcenv::Action action_pb;
    std::vector<TensorNest> rollout;
    try {
      while (true) {
        rollout.push_back(std::move(last));

        for (int t = 1; t <= unroll_length_; ++t) {
          all_agent_outputs = inference_batcher_->compute(compute_inputs);

          agent_state = all_agent_outputs.get_vector()[1];
          agent_outputs = all_agent_outputs.get_vector()[0];

          // agent_outputs must be a tuple/list.
          const TensorNest& action = agent_outputs.get_vector().front();

          action_pb.Clear();

          fill_nest_pb(
              action_pb.mutable_nest_action(), action,
              [&](rpcenv::NDArray* array, const torch::Tensor& tensor) {
                return fill_ndarray_pb(array, tensor, /*start_dim=*/2);
              });

          stream->Write(action_pb);
          if (!stream->Read(&step_pb)) {
            throw py::connection_error("Read failed.");
          }
          env_outputs = ActorPool::step_pb_to_nest(&step_pb);
          compute_inputs = TensorNest(std::vector({env_outputs, agent_state}));

          last = TensorNest(std::vector({env_outputs, agent_outputs}));
          rollout.push_back(std::move(last));
        }
        last = rollout.back();
        learner_queue_->enqueue({
            TensorNest(std::vector(
                {batch(rollout, 0), std::move(initial_agent_state)})),
        });
        rollout.clear();
        initial_agent_state = agent_state;  // Copy
        count_ += unroll_length_;
      }
    } catch (const ClosedBatchingQueue& e) {
      // Thrown when inference_batcher_ and learner_queue_ are closed. Stop.
      stream->WritesDone();
      grpc::Status status = stream->Finish();
      if (!status.ok()) {
        std::cerr << "rpc failed on finish." << std::endl;
      }
    }
  }

  void run() {
    // std::async instead of plain threads as we want to raise any exceptions
    // here and not in the created threads.
    std::vector<std::future<void>> futures;
    for (int64_t i = 0, size = env_server_addresses_.size(); i != size; ++i) {
      futures.push_back(std::async(std::launch::async, &ActorPool::loop, this,
                                   i, env_server_addresses_[i]));
    }
    for (auto& future : futures) {
      // This will only catch errors in the first thread. std::when_any would be
      // good here but it's not available yet. We could also write the
      // condition_variable code ourselves, but let's not do this.
      future.get();
    }
  }

  uint64_t count() const { return count_; }

  static TensorNest array_pb_to_nest(rpcenv::NDArray* array_pb) {
    std::vector<int64_t> shape = {1, 1};  // [T=1, B=1].
    for (int i = 0, length = array_pb->shape_size(); i < length; ++i) {
      shape.push_back(array_pb->shape(i));
    }
    std::string* data = array_pb->release_data();
    at::ScalarType dtype = torch::utils::numpy_dtype_to_aten(array_pb->dtype());

    return TensorNest(torch::from_blob(
        data->data(), shape,
        /*deleter=*/[data](void*) { delete data; }, dtype));
  }

  static TensorNest step_pb_to_nest(rpcenv::Step* step_pb) {
    TensorNest done = TensorNest(
        torch::full({1, 1}, step_pb->done(), torch::dtype(torch::kBool)));
    TensorNest reward = TensorNest(torch::full({1, 1}, step_pb->reward()));
    TensorNest episode_step = TensorNest(torch::full(
        {1, 1}, step_pb->episode_step(), torch::dtype(torch::kInt32)));
    TensorNest episode_return =
        TensorNest(torch::full({1, 1}, step_pb->episode_return()));

    return TensorNest(std::vector(
        {nest_pb_to_nest(step_pb->mutable_observation(), array_pb_to_nest),
         std::move(reward), std::move(done), std::move(episode_step),
         std::move(episode_return)}));
  }

  static void fill_ndarray_pb(rpcenv::NDArray* array,
                              const torch::Tensor& tensor,
                              int64_t start_dim = 0) {
    if (!tensor.is_contiguous())
      // TODO(heiner): Fix this non-contiguous case.
      throw py::value_error("Cannot convert non-contiguous tensor.");
    array->set_dtype(aten_to_dtype(tensor.scalar_type()));

    at::IntArrayRef shape = tensor.sizes();

    for (size_t i = start_dim, ndim = shape.size(); i < ndim; ++i) {
      array->add_shape(shape[i]);
    }

    // TODO: Consider set_allocated_data.
    // TODO: Consider [ctype = STRING_VIEW] in proto file.
    array->set_data(tensor.data_ptr(), tensor.nbytes());
  }

 private:
  // Copied from the private torch/csrc/utils/tensor_numpy.cpp.
  // TODO(heiner): Expose this in PyTorch then use that function.
  static int aten_to_dtype(const at::ScalarType scalar_type) {
    switch (scalar_type) {
      case at::kDouble:
        return NPY_DOUBLE;
      case at::kFloat:
        return NPY_FLOAT;
      case at::kHalf:
        return NPY_HALF;
      case at::kLong:
        return NPY_INT64;
      case at::kInt:
        return NPY_INT32;
      case at::kShort:
        return NPY_INT16;
      case at::kChar:
        return NPY_INT8;
      case at::kByte:
        return NPY_UINT8;
      case at::kBool:
        return NPY_BOOL;
      default: {
        std::string what = "Got unsupported ScalarType ";
        throw py::value_error(what + at::toString(scalar_type));
      }
    }
  }

  std::atomic_uint64_t count_;

  const int unroll_length_;
  std::shared_ptr<BatchingQueue<>> learner_queue_;
  std::shared_ptr<DynamicBatcher> inference_batcher_;
  const std::vector<std::string> env_server_addresses_;
  TensorNest initial_agent_state_;
};

void init_actorpool(py::module& m) {
  py::register_exception<std::future_error>(m, "AsyncError");
  py::register_exception<ClosedBatchingQueue>(m, "ClosedBatchingQueue");
  py::register_exception<std::bad_variant_access>(m, "NestError");

  py::class_<ActorPool>(m, "ActorPool")
      .def(py::init<int, std::shared_ptr<BatchingQueue<>>,
                    std::shared_ptr<DynamicBatcher>, std::vector<std::string>,
                    TensorNest>(),
           py::arg("unroll_length"), py::arg("learner_queue").none(false),
           py::arg("inference_batcher").none(false),
           py::arg("env_server_addresses"), py::arg("initial_agent_state"))
      .def("run", &ActorPool::run, py::call_guard<py::gil_scoped_release>())
      .def("count", &ActorPool::count);

  py::class_<DynamicBatcher::Batch, std::shared_ptr<DynamicBatcher::Batch>>(
      m, "Batch")
      .def("get_inputs", &DynamicBatcher::Batch::get_inputs)
      .def("set_outputs", &DynamicBatcher::Batch::set_outputs,
           py::arg("outputs"), py::call_guard<py::gil_scoped_release>());

  py::class_<DynamicBatcher, std::shared_ptr<DynamicBatcher>>(m,
                                                              "DynamicBatcher")
      .def(py::init<int64_t, int64_t, int64_t, std::optional<int>, bool>(),
           py::arg("batch_dim") = 1, py::arg("minimum_batch_size") = 1,
           py::arg("maximum_batch_size") = 1024, py::arg("timeout_ms") = 100,
           py::arg("check_outputs") = true, R"docstring(
             DynamicBatcher class.
             If timeout_ms is set to None, the batcher will not allow data to be
             retrieved until at least minimum_batch_size inputs are provided.
             If timeout_ms is not None (default behaviour), the batcher will
             allow data to be retrieved when the timeout expires, even if the
             number of inputs received is smaller than minimum_batch_size.
           )docstring")
      .def("close", &DynamicBatcher::close)
      .def("is_closed", &DynamicBatcher::is_closed)
      .def("size", &DynamicBatcher::size)
      .def("compute", &DynamicBatcher::compute,
           py::call_guard<py::gil_scoped_release>())
      .def("__iter__",
           [](std::shared_ptr<DynamicBatcher> batcher) { return batcher; })
      .def("__next__", &DynamicBatcher::get_batch,
           py::call_guard<py::gil_scoped_release>());

  py::class_<BatchingQueue<>, std::shared_ptr<BatchingQueue<>>>(m,
                                                                "BatchingQueue")
      .def(py::init<int64_t, int64_t, int64_t, std::optional<int>, bool,
                    std::optional<uint64_t>>(),
           py::arg("batch_dim") = 1, py::arg("minimum_batch_size") = 1,
           py::arg("maximum_batch_size") = 1024,
           py::arg("timeout_ms") = std::nullopt, py::arg("check_inputs") = true,
           py::arg("maximum_queue_size") = std::nullopt)
      .def("enqueue",
           [](std::shared_ptr<BatchingQueue<>> queue, TensorNest tensors) {
             queue->enqueue({std::move(tensors), Empty()});
           })
      .def("close", &BatchingQueue<>::close)
      .def("is_closed", &BatchingQueue<>::is_closed)
      .def("size", &BatchingQueue<>::size)
      .def("__iter__",
           [](std::shared_ptr<BatchingQueue<>> queue) { return queue; })
      .def("__next__", [](BatchingQueue<>& queue) {
        py::gil_scoped_release release;
        std::pair<TensorNest, std::vector<Empty>> pair = queue.dequeue_many();
        return pair.first;
      });
}
