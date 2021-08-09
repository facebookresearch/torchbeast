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

/*
Tests for libtorchbeast/nest_serialize.h.
*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nest_serialize.h"

// Include actorpool in order to use the functions:
// - fill_ndarray_pb,
// - array_pb_to_nest.
#include "actorpool.cc"

namespace {

TensorNest make_complex_tensornest(const torch::Tensor& t) {
  // Nest composed by:
  // map<string, TensorNest> {
  //   "vec" : vector<TensorNest> [
  //     TensorNest(Tensor passed by argument)
  //   ]
  // }
  return TensorNest(std::map<std::string, TensorNest>({std::make_pair(
      "vec", TensorNest(std::vector<TensorNest>({TensorNest(t)})))}));
}

class FillNestPbTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor_size = 5;
    tensor = torch::ones(tensor_size);
  }

  int tensor_size;
  torch::Tensor tensor;
};

TEST_F(FillNestPbTest, Simple) {
  TensorNest nest = TensorNest(tensor);
  rpcenv::ArrayNest nest_pb;
  fill_nest_pb(
      &nest_pb, nest, [&](rpcenv::NDArray* array, const torch::Tensor& tensor) {
        return ActorPool::fill_ndarray_pb(array, tensor, /*start_dim=*/0);
      });

  rpcenv::NDArray ndarray = nest_pb.array();
  ASSERT_THAT(ndarray.shape(), testing::ElementsAre(tensor_size));
  ASSERT_EQ(*ndarray.mutable_data(),
            std::string(static_cast<const char*>(tensor.data_ptr()),
                        tensor.nbytes()));
  ASSERT_EQ(ndarray.dtype(), NPY_FLOAT);
}

TEST_F(FillNestPbTest, Complex) {
  TensorNest nest = make_complex_tensornest(tensor);
  rpcenv::ArrayNest nest_pb;
  fill_nest_pb(
      &nest_pb, nest, [&](rpcenv::NDArray* array, const torch::Tensor& tensor) {
        return ActorPool::fill_ndarray_pb(array, tensor, /*start_dim=*/0);
      });

  rpcenv::NDArray ndarray = nest_pb.map().at("vec").vector(0).array();
  ASSERT_THAT(ndarray.shape(), testing::ElementsAre(tensor_size));
  ASSERT_EQ(*ndarray.mutable_data(),
            std::string(static_cast<const char*>(tensor.data_ptr()),
                        tensor.nbytes()));
  ASSERT_EQ(ndarray.dtype(), NPY_FLOAT);
}

class NestPbToNestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tensor_size = 5;
    tensor = torch::ones({1, 1, tensor_size});
  }

  int tensor_size;
  torch::Tensor tensor;
};

TEST_F(NestPbToNestTest, Simple) {
  rpcenv::ArrayNest nest_pb;
  rpcenv::NDArray* ndarray = nest_pb.mutable_array();
  ndarray->add_shape(tensor_size);
  ndarray->set_dtype(NPY_FLOAT);
  ndarray->set_data(std::string(static_cast<const char*>(tensor.data_ptr()),
                                tensor.nbytes()));

  TensorNest nest = nest_pb_to_nest(&nest_pb, ActorPool::array_pb_to_nest);
  nest.for_each([this](const torch::Tensor& t) { ASSERT_TRUE(t.equal(tensor)); });
}

TEST_F(NestPbToNestTest, Complex) {
  TensorNest target_nest = make_complex_tensornest(tensor);

  // ArrayNest with a vector containing an array.
  rpcenv::ArrayNest nest_pb_vector;
  // ArrayNest containing an array.
  rpcenv::ArrayNest* nest_pb_array = nest_pb_vector.add_vector();
  rpcenv::NDArray* ndarray = nest_pb_array->mutable_array();
  ndarray->add_shape(tensor_size);
  ndarray->set_dtype(NPY_FLOAT);
  ndarray->set_data(std::string(static_cast<const char*>(tensor.data_ptr()),
                                tensor.nbytes()));
  // ArrayNest with map with that contains a vector containing an array.
  rpcenv::ArrayNest nest_pb_map;
  auto nest_pb_mutable_map = nest_pb_map.mutable_map();
  nest_pb_mutable_map->insert(
      ::google::protobuf::MapPair(std::string("vec"), nest_pb_vector));

  TensorNest nest = nest_pb_to_nest(&nest_pb_map, ActorPool::array_pb_to_nest);

  // Compare nest is equal to target_nest.
  nest::Nest<bool> equal_nest = TensorNest::map2(
      [](const torch::Tensor& actual, const torch::Tensor& target) {
        return actual.equal(target);
      },
      nest, target_nest);
  equal_nest.for_each([](const bool e) { ASSERT_TRUE(e); });
}

}  // namespace