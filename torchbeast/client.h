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

#include <grpc++/grpc++.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nest_serialize.h"
#include "rpc.grpc.pb.h"
#include "rpc.pb.h"

#include "../third_party/nest/nest/nest.h"
#include "../third_party/nest/nest/nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<py::array> PyArrayNest;

namespace torchbeast {
class Client {
 public:
  Client(const std::string &address) : address_(address) {}

  void connect(int deadline_sec = 60);

  PyArrayNest call(const std::string &function, PyArrayNest inputs);

 private:
  static void fill_ndarray_pb(NDArray *array, py::array pyarray);
  static PyArrayNest array_pb_to_nest(NDArray *array_pb);

  const std::string address_;
  std::unique_ptr<RPC::Stub> stub_;
  grpc::ClientContext context_;
  std::shared_ptr<grpc::ClientReaderWriter<CallRequest, CallResponse>> stream_;
};

}  // namespace torchbeast
