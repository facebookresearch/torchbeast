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

#include "nest/nest.h"
#include "rpcenv.pb.h"

template <typename T, typename Function>
void fill_nest_pb(rpcenv::ArrayNest* nest_pb, nest::Nest<T> nest,
                  Function fill_ndarray_pb) {
  using Nest = nest::Nest<T>;
  std::visit(
      nest::overloaded{
          [nest_pb, &fill_ndarray_pb](const T t) {
            fill_ndarray_pb(nest_pb->mutable_array(), t);
          },
          [nest_pb, &fill_ndarray_pb](const std::vector<Nest>& v) {
            for (const Nest& n : v) {
              rpcenv::ArrayNest* subnest = nest_pb->add_vector();
              fill_nest_pb(subnest, n, fill_ndarray_pb);
            }
          },
          [nest_pb, &fill_ndarray_pb](const std::map<std::string, Nest>& m) {
            auto* map_pb = nest_pb->mutable_map();
            for (const auto& p : m) {
              rpcenv::ArrayNest& subnest_pb = (*map_pb)[p.first];
              fill_nest_pb(&subnest_pb, p.second, fill_ndarray_pb);
            }
          }},
      nest.value);
}

template <typename Function>
std::invoke_result_t<Function, rpcenv::NDArray*> nest_pb_to_nest(
    rpcenv::ArrayNest* nest_pb, Function array_to_nest) {
  using Nest = std::invoke_result_t<Function, rpcenv::NDArray*>;
  if (nest_pb->has_array()) {
    return array_to_nest(nest_pb->mutable_array());
  }
  if (nest_pb->vector_size() > 0) {
    std::vector<Nest> v;
    for (int i = 0, length = nest_pb->vector_size(); i < length; ++i) {
      v.push_back(nest_pb_to_nest(nest_pb->mutable_vector(i), array_to_nest));
    }
    return Nest(std::move(v));
  }
  if (nest_pb->map_size() > 0) {
    std::map<std::string, Nest> m;
    for (auto& p : *nest_pb->mutable_map()) {
      m[p.first] = nest_pb_to_nest(&p.second, array_to_nest);
    }
    return Nest(std::move(m));
  }
  throw std::invalid_argument("ArrayNest proto contained no data.");
}
