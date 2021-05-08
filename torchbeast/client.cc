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

#include "client.h"

namespace py = pybind11;

// Enable a few standard Python exceptions.
namespace pybind11 {
PYBIND11_RUNTIME_EXCEPTION(runtime_error, PyExc_RuntimeError)
PYBIND11_RUNTIME_EXCEPTION(timeout_error, PyExc_TimeoutError)
PYBIND11_RUNTIME_EXCEPTION(connection_error, PyExc_ConnectionError)
}  // namespace pybind11

typedef nest::Nest<py::array> PyArrayNest;

namespace torchbeast {

void Client::connect(int deadline_sec) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel(address_, grpc::InsecureChannelCredentials());
  stub_ = RPC::NewStub(channel);

  auto deadline =
      std::chrono::system_clock::now() + std::chrono::seconds(deadline_sec);

  if (!channel->WaitForConnected(deadline)) {
    throw py::timeout_error("WaitForConnected timed out.");
  }
  stream_ = stub_->Call(&context_);
}

PyArrayNest Client::call(const std::string &function, PyArrayNest inputs) {
  if (!stream_) throw py::runtime_error("Client not connected");

  CallRequest call_req;
  call_req.set_function(function);
  fill_nest_pb(call_req.mutable_inputs(), std::move(inputs), fill_ndarray_pb);

  CallResponse call_resp;
  {
    py::gil_scoped_release release;
    if (!stream_->Write(call_req)) throw py::connection_error("Write failed");
    if (!stream_->Read(&call_resp)) throw py::connection_error("Read failed");
  }

  if (call_resp.has_error()) {
    std::string message = call_resp.error().message();
    std::size_t pos = message.find(": ");
    if (pos != std::string::npos) {
      std::string type = message.substr(0, pos);
      py::object builtins = py::module::import("builtins");
      if (py::hasattr(builtins, type.c_str())) {
        py::object error_type = builtins.attr(type.c_str());
        if (PyExceptionClass_Check(error_type.ptr())) {
          message = message.substr(pos + 2);
          PyErr_SetString(error_type.ptr(), message.c_str());
          throw py::error_already_set();
        }
      }
    }
    throw py::connection_error(call_resp.error().message());
  }

  return nest_pb_to_nest(call_resp.mutable_outputs(), array_pb_to_nest);
}

void Client::fill_ndarray_pb(NDArray *array, py::array pyarray) {
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

PyArrayNest Client::array_pb_to_nest(NDArray *array_pb) {
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
}  // namespace torchbeast
