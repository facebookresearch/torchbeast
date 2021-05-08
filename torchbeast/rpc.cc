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

#include "client.h"
#include "server.h"

PYBIND11_MODULE(rpc, m) {
  py::class_<torchbeast::Client>(m, "Client")
      .def(py::init<const std::string &>(), py::arg("address"))
      .def("connect", &torchbeast::Client::connect,
           py::arg("deadline_sec") = 60)
      .def("call", &torchbeast::Client::call);

  py::class_<torchbeast::Server>(m, "Server")
      .def(py::init<const std::string &, uint>(), py::arg("address"),
           py::arg("max_parallel_calls") = 2)
      .def("run", &torchbeast::Server::run)
      .def("running", &torchbeast::Server::running)
      .def("wait", &torchbeast::Server::wait,
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &torchbeast::Server::stop,
           py::call_guard<py::gil_scoped_release>())
      .def("bind", &torchbeast::Server::bind, py::keep_alive<1, 3>(),
           py::arg("name"), py::arg("function"),
           py::arg("batch_size") = std::nullopt);
}
