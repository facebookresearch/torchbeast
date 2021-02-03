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

#include <iostream>
#include <tuple>

#include <grpc++/grpc++.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nest_serialize.h"
#include "rpcenv.grpc.pb.h"
#include "rpcenv.pb.h"

#include "nest/nest.h"
#include "nest/nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<py::array> PyArrayNest;

namespace rpcenv {
class EnvServer {
 private:
  class ServiceImpl final : public RPCEnvServer::Service {
   public:
    ServiceImpl(py::object env_init) : env_init_(env_init) {}

   private:
    virtual grpc::Status StreamingEnv(
        grpc::ServerContext *context,
        grpc::ServerReaderWriter<Step, Action> *stream) override {
      py::gil_scoped_acquire acquire;  // Destroy after pyenv.
      py::object pyenv;
      py::object stepfunc;
      py::object resetfunc;

      PyArrayNest observation;
      float reward = 0.0;
      bool done = true;
      int episode_step = 0;
      float episode_return = 0.0;

      auto set_observation = py::cpp_function(
          [&observation](PyArrayNest o) { observation = std::move(o); },
          py::arg("observation"));

      auto set_observation_reward_done = py::cpp_function(
          [&observation, &reward, &done](PyArrayNest o, float r, bool d,
                                         py::args) {
            observation = std::move(o);
            reward = r;
            done = d;
          },
          py::arg("observation"), py::arg("reward"), py::arg("done"));

      try {
        pyenv = env_init_();
        stepfunc = pyenv.attr("step");
        resetfunc = pyenv.attr("reset");
        set_observation(resetfunc());
      } catch (const pybind11::error_already_set &e) {
        // Needs to be caught and not re-raised, as this isn't in a Python
        // thread.
        std::cerr << e.what() << std::endl;
        return grpc::Status(grpc::INTERNAL, e.what());
      }

      Step step_pb;
      fill_nest_pb(step_pb.mutable_observation(), std::move(observation),
                   fill_ndarray_pb);

      step_pb.set_reward(reward);
      step_pb.set_done(done);
      step_pb.set_episode_step(episode_step);
      step_pb.set_episode_return(episode_return);

      Action action_pb;
      while (true) {
        {
          py::gil_scoped_release release;  // Release while doing transfer.
          stream->Write(step_pb);
          if (!stream->Read(&action_pb)) {
            break;
          }
        }
        try {
          // I'm not sure if this is fast, but it's convienient.
          set_observation_reward_done(*stepfunc(nest_pb_to_nest(
              action_pb.mutable_nest_action(), array_pb_to_nest)));

          episode_step += 1;
          episode_return += reward;

          step_pb.Clear();
          step_pb.set_reward(reward);
          step_pb.set_done(done);
          step_pb.set_episode_step(episode_step);
          step_pb.set_episode_return(episode_return);
          if (done) {
            set_observation(resetfunc());
            // Reset episode_* for the _next_ step.
            episode_step = 0;
            episode_return = 0.0;
          }
        } catch (const pybind11::error_already_set &e) {
          std::cerr << e.what() << std::endl;
          return grpc::Status(grpc::INTERNAL, e.what());
        }

        fill_nest_pb(step_pb.mutable_observation(), std::move(observation),
                     fill_ndarray_pb);
      }
      return grpc::Status::OK;
    }

    py::object env_init_;  // TODO: Make sure GIL is held when destroyed.

    // TODO: Add observation and action size functions (pre-load env)
  };

 public:
  EnvServer(py::object env_class, const std::string &server_address)
      : server_address_(server_address),
        service_(env_class),
        server_(nullptr) {}

  void run() {
    if (server_) {
      throw std::runtime_error("Server already running");
    }
    py::gil_scoped_release release;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address_,
                             grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    std::cerr << "Server listening on " << server_address_ << std::endl;

    server_->Wait();
  }

  void stop() {
    if (!server_) {
      throw std::runtime_error("Server not running");
    }
    server_->Shutdown();
  }

  static void fill_ndarray_pb(rpcenv::NDArray *array, py::array pyarray) {
    // Make sure array is C-style contiguous. If it isn't, this creates
    // another memcopy that is not strictly necessary.
    if ((pyarray.flags() & py::array::c_style) == 0) {
      pyarray = py::array::ensure(pyarray, py::array::c_style);
    }

    // This seems surprisingly involved. An alternative would be to include
    // numpy/arrayobject.h and use PyArray_TYPE.
    int type_num =
        py::detail::array_descriptor_proxy(pyarray.dtype().ptr())->type_num;

    array->set_dtype(type_num);
    for (size_t i = 0, ndim = pyarray.ndim(); i < ndim; ++i) {
      array->add_shape(pyarray.shape(i));
    }

    // TODO: Consider set_allocated_data.
    // TODO: consider [ctype = STRING_VIEW] in proto file.
    py::buffer_info info = pyarray.request();
    array->set_data(info.ptr, info.itemsize * info.size);
  }

  static PyArrayNest array_pb_to_nest(rpcenv::NDArray *array_pb) {
    std::vector<int64_t> shape;
    for (int i = 0, length = array_pb->shape_size(); i < length; ++i) {
      shape.push_back(array_pb->shape(i));
    }

    // Somewhat complex way of turning an type_num into a py::dtype.
    py::dtype dtype = py::reinterpret_borrow<py::dtype>(
        py::detail::npy_api::get().PyArray_DescrFromType_(array_pb->dtype()));

    std::string *data = array_pb->release_data();

    // Attach capsule as base in order to free data.
    return PyArrayNest(py::array(dtype, shape, {}, data->data(),
                                 py::capsule(data, [](void *ptr) {
                                   delete reinterpret_cast<std::string *>(ptr);
                                 })));
  }

 private:
  const std::string server_address_;
  ServiceImpl service_;
  std::unique_ptr<grpc::Server> server_;
};

}  // namespace rpcenv

void init_rpcenv(py::module &m) {
  py::class_<rpcenv::EnvServer>(m, "Server")
      .def(py::init<py::object, const std::string &>(), py::arg("env_class"),
           py::arg("server_address") = "unix:/tmp/polybeast")
      .def("run", &rpcenv::EnvServer::run)
      .def("stop", &rpcenv::EnvServer::stop);
}
