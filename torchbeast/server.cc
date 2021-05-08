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

#include <chrono>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

// Enable including numpy via numpy_stub.h.
#define USE_NUMPY 1

#include <ATen/core/ivalue.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/utils/numpy_stub.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "nest_serialize.h"
#include "server.h"

#include "batchedfn.h"

#include "../third_party/nest/nest/nest.h"
#include "../third_party/nest/nest/nest_pybind.h"

namespace py = pybind11;

// Enable a few standard Python exceptions.
namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(runtime_error, PyExc_RuntimeError)
PYBIND11_RUNTIME_EXCEPTION(timeout_error, PyExc_TimeoutError)
PYBIND11_RUNTIME_EXCEPTION(connection_error, PyExc_ConnectionError)
}  // namespace pybind11

namespace torchbeast {

namespace detail {
at::Tensor array_pb_to_tensor(torchbeast::NDArray *array_pb) {
  std::vector<int64_t> shape;
  for (int i = 0, length = array_pb->shape_size(); i < length; ++i) {
    shape.push_back(array_pb->shape(i));
  }
  std::string *data = array_pb->release_data();
  at::ScalarType dtype = torch::utils::numpy_dtype_to_aten(array_pb->dtype());

  return torch::from_blob(
      data->data(), shape,
      /*deleter=*/[data](void *) { delete data; }, dtype);
}

void fill_ndarray_pb(NDArray *array, const at::Tensor &tensor,
                     int64_t start_dim = 0) {
  if (!tensor.is_contiguous())
    // TODO(heiner): Fix this non-contiguous case.
    throw py::value_error("Cannot convert non-contiguous tensor.");
  array->set_dtype(torch::utils::aten_to_numpy_dtype(tensor.scalar_type()));

  at::IntArrayRef shape = tensor.sizes();

  for (size_t i = start_dim, ndim = shape.size(); i < ndim; ++i) {
    array->add_shape(shape[i]);
  }

  // TODO: Consider set_allocated_data.
  // TODO: Consider [ctype = STRING_VIEW] in proto file.
  array->set_data(tensor.data_ptr(), tensor.nbytes());
}

torch::IValue nest_pb_to_ivalue(torchbeast::ArrayNest *nest_pb) {
  if (nest_pb->has_array()) {
    return torch::IValue(array_pb_to_tensor(nest_pb->mutable_array()));
  }
  if (nest_pb->vector_size() > 0) {
    c10::List<torch::IValue> v =
        c10::impl::GenericList(torch::jit::AnyType::get());
    for (int i = 0, length = nest_pb->vector_size(); i < length; ++i) {
      v.push_back(nest_pb_to_ivalue(nest_pb->mutable_vector(i)));
    }
    return torch::IValue(std::move(v));
  }
  if (nest_pb->map_size() > 0) {
    c10::Dict<torch::IValue, torch::IValue> d = c10::impl::GenericDict(
        torch::jit::StringType::get(), torch::jit::AnyType::get());
    for (auto &p : *nest_pb->mutable_map()) {
      d.insert(torch::IValue(p.first), nest_pb_to_ivalue(&p.second));
    }
    return torch::IValue(std::move(d));
  }
  throw std::invalid_argument("ArrayNest proto contained no data.");
}
}  // namespace detail

grpc::Status Server::ServiceImpl::bind(const std::string &name,
                                       Function &&function) {
  functions_.insert({name, std::move(function)});
  return grpc::Status::OK;
}

grpc::Status Server::ServiceImpl::Call(
    grpc::ServerContext *context,
    grpc::ServerReaderWriter<CallResponse, CallRequest> *stream) {
  CallRequest call_req;

  while (stream->Read(&call_req)) {
    CallResponse call_resp;
    try {
      auto it = functions_.find(call_req.function());
      if (it == functions_.end())
        throw std::runtime_error("AttributeError: No such function '" +
                                 call_req.function() + "'");
      call_resp = it->second(&call_req);
    } catch (const ServerClosed &e) {
      break;
    } catch (const std::runtime_error &e) {
      std::cerr << "Error in " << call_req.function() << ": " << e.what()
                << std::endl;
      call_resp.mutable_error()->set_message(e.what());
    } catch (const std::exception &e) {
      std::cerr << "Error in " << call_req.function() << ": " << e.what()
                << std::endl;
      return grpc::Status(grpc::INTERNAL, e.what());
    }
    stream->Write(call_resp);
  }

  return grpc::Status::OK;
}

void Server::run() {
  if (server_) throw std::runtime_error("Server already running");

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address_, grpc::InsecureServerCredentials());
  builder.RegisterService(&service_);
  server_ = builder.BuildAndStart();

  running_.store(true);
}

void Server::wait() {
  if (!server_) throw std::runtime_error("Server not running");

  server_->Wait();
}

void Server::stop() {
  if (!server_) throw std::runtime_error("Server not running");

  running_.store(false);
  server_->Shutdown(std::chrono::system_clock::now());
}

void Server::bind(const std::string &name, const py::object &obj,
                  std::optional<int32_t> batch_size) {
  if (!batch_size) {
    if (py::isinstance(obj,
                       py::module::import("torch.jit").attr("ScriptModule"))) {
      // Cf. torch/csrc/jit/script/module_python.h in PyTorch.
      ScriptModule module = py::cast<ScriptModule>(obj.attr("_c"));
      service_.bind(name, [module(std::move(module))](
                              CallRequest *call_req) mutable {
        torch::IValue inputs =
            detail::nest_pb_to_ivalue(call_req->mutable_inputs());
        torch::IValue result =
            module.forward(inputs.toList().vec());
        CallResponse call_resp;
        // TODO: Write IValue-to-proto code.
        fill_nest_pb(call_resp.mutable_outputs(), TensorNest(result.toTensor()),
                     [&](NDArray *array, const at::Tensor &tensor) {
                       return detail::fill_ndarray_pb(array, tensor, 0);
                     });
        return call_resp;
      });
    } else {  //  Treat obj as callable.
      service_.bind(name, [obj](CallRequest *call_req) mutable {
        CallResponse call_resp;

        auto array_pb_to_nest = [](torchbeast::NDArray *array_pb) {
          return TensorNest(detail::array_pb_to_tensor(array_pb));
        };

        TensorNest inputs =
            nest_pb_to_nest(call_req->mutable_inputs(), array_pb_to_nest);
        py::gil_scoped_acquire acquire;

        py::object result;
        try {
          result = obj(*py::cast(inputs));
        } catch (py::error_already_set &e) {
          call_resp.mutable_error()->set_message(e.what());
          return call_resp;
        }

        fill_nest_pb(call_resp.mutable_outputs(), py::cast<TensorNest>(result),
                     [&](NDArray *array, const at::Tensor &tensor) {
                       return detail::fill_ndarray_pb(array, tensor, 0);
                     });
        return call_resp;
      });
    }
  } else {  // batched operation
    if (py::isinstance(obj,
                       py::module::import("torch.jit").attr("ScriptModule"))) {
      // TODO: Implement batching here too.
      throw std::runtime_error("batched scriptmodule not yet supported.");
    }

    auto f = [obj](TensorNest inputs) mutable {
      py::gil_scoped_acquire acquire;
      py::object result;

      try {
        result = obj(*py::cast(inputs));
      } catch (py::error_already_set &e) {
        /* We cannot let this error_already_set bubble up as it will
           require the GIL. While we're at it, we can send over a nicely
           formatted exception. */

        // In newer versions of pybind11, this is just e.trace().
        // TODO: Update PyTorch's pybind11 version.
        py::object type, value, trace;
        e.restore();
        PyErr_Fetch(&type.ptr(), &value.ptr(), &trace.ptr());

        py::object pstream = py::module::import("io").attr("StringIO")();
        PyTraceBack_Print(trace.ptr(), pstream.ptr());

        throw std::runtime_error(
            e.what() + std::string("\n") +
            py::cast<std::string>(pstream.attr("getvalue")()));
      }

      return py::cast<TensorNest>(result);
    };

    BatchedFn batched_f(std::move(f), *batch_size, &running_, &threadpool_);

    service_.bind(name, [batched_f = std::move(batched_f)](
                            CallRequest *call_req) mutable {
      CallResponse call_resp;

      auto array_pb_to_nest = [](torchbeast::NDArray *array_pb) {
        return TensorNest(detail::array_pb_to_tensor(array_pb));
      };

      TensorNest outputs = batched_f(
          nest_pb_to_nest(call_req->mutable_inputs(), array_pb_to_nest));

      fill_nest_pb(call_resp.mutable_outputs(), outputs,
                   [&](NDArray *array, const at::Tensor &tensor) {
                     return detail::fill_ndarray_pb(array, tensor, 0);
                   });
      return call_resp;
    });
  }
}
}  // namespace torchbeast
