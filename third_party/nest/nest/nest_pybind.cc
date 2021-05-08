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

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nest.h"
#include "nest_pybind.h"

namespace py = pybind11;

typedef nest::Nest<py::object> PyNest;

PYBIND11_MODULE(_nest, m) {
  m.def("map", [](py::function f, const PyNest &n) {
    // This says const py::object, but f can actually modify it!
    std::function<py::object(const py::object &)> cppf =
        [&f](const py::object &arg) { return f(arg); };
    return n.map(cppf);
  });
  m.def("map_many",
        [](const std::function<py::object(const std::vector<py::object> &)> &f,
           py::args args) {
          std::vector<PyNest> nests = args.cast<std::vector<PyNest>>();
          return PyNest::zip(nests).map(f);
        });
  m.def("map_many2", [](const std::function<py::object(const py::object &,
                                                       const py::object &)> &f,
                        const PyNest &n1, const PyNest &n2) {
    try {
      return PyNest::map2(f, n1, n2);
    } catch (const std::invalid_argument &e) {
      // IDK why I have to do this manually.
      throw py::value_error(e.what());
    }
  });
  m.def("for_each", [](py::function f, PyNest &n) {
    n.for_each([&f](py::object &arg) { f(arg); });
  });
  m.def("for_each2", [](py::function f, PyNest &n1, PyNest &n2) {
    PyNest::for_each([&f](py::object &a, const py::object &b) { f(a, b); }, n1,
                     n2);
  });
  m.def("pack_as", [](const PyNest &n, const py::sequence &sequence) {
    try {
      return n.pack_as(sequence.begin(), sequence.end());
    } catch (const std::exception &e) {
      // PyTorch pybind11 doesn't seem to translate exceptions?
      throw py::value_error(e.what());
    }
  });
  m.def("front", [](const PyNest &n) { return n.front(); });
}
