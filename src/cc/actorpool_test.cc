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
Tests for libtorchbeast/actorpool.cc.
Build this test by running scripts/build_actorpool_test.sh.
*/

#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Including .cc file as actorpool does not have a header file.
#include "actorpool.cc"

namespace {

TEST(BatchingQueue, TestConstructDestroy) {
  {
    BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                                 /*maximum_batch_size=*/128);
    batching_queue.close();
  }
  {
    BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                                 /*maximum_batch_size=*/128,
                                 /*timeout=*/60 * 10, /*check_inputs=*/true);
    batching_queue.close();
  }
}

TEST(BatchingQueue, TestBadConstruct) {
  // Invalid minimum_batch_size = 0.
  ASSERT_THROW(
      {
        BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/0,
                                     /*maximum_batch_size=*/1);
      },
      py::value_error);
  // Invalid maximum_batch_size = 0.
  ASSERT_THROW(
      {
        BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/1,
                                     /*maximum_batch_size=*/0);
      },
      py::value_error);
  // Invalid minimum_batch_size > maximum_batch_size.
  ASSERT_THROW(
      {
        BatchingQueue batching_queue(/*batch_dim=*/3,
                                     /*minimum_batch_size=*/128,
                                     /*maximum_batch_size=*/64);
      },
      py::value_error);
}

TEST(BatchingQueue, TestMultipleCloseCalls) {
  BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                               /*maximum_batch_size=*/128);
  batching_queue.close();
  ASSERT_THROW({ batching_queue.close(); }, py::runtime_error);
}

TEST(BatchingQueue, TestEnqueue) {
  BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                               /*maximum_batch_size=*/128);
  TensorNest tensor_nest(torch::ones({2, 2, 2, 2}));
  batching_queue.enqueue({tensor_nest, Empty()});
  batching_queue.close();
}

TEST(BatchingQueue, TestBadEnqueue) {
  // Tensor with only one dimension, but more than 3 are expected.
  ASSERT_THROW(
      {
        BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                                     /*maximum_batch_size=*/128);
        TensorNest tensor_nest(torch::ones(5));
        batching_queue.enqueue({tensor_nest, Empty()});
      },
      py::value_error);
  // Empty tensor.
  ASSERT_THROW(
      {
        BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                                     /*maximum_batch_size=*/128);
        TensorNest tensor_nest(torch::ones({}));
        batching_queue.enqueue({tensor_nest, Empty()});
      },
      py::value_error);
  // Pushing tensors when the queue is closed.
  ASSERT_THROW(
      {
        BatchingQueue batching_queue(/*batch_dim=*/3, /*minimum_batch_size=*/64,
                                     /*maximum_batch_size=*/128);
        TensorNest tensor_nest(torch::ones({2, 2, 2, 2}));
        batching_queue.close();
        batching_queue.enqueue({tensor_nest, Empty()});
      },
      ClosedBatchingQueue);
}

TEST(BatchingQueue, TestDequeueMany) {
  int64_t minimum_batch_size = 32;
  int64_t maximum_batch_size = 64;

  BatchingQueue batching_queue(/*batch_dim=*/3, minimum_batch_size,
                               maximum_batch_size);

  // Push enough tensors in order to trigger enough_inputs_.notify_one() in
  // enqueue.
  for (int i = 0; i < minimum_batch_size; i++) {
    TensorNest tensor_nest(torch::ones({2, 2, 2, 2}));
    batching_queue.enqueue({tensor_nest, Empty()});
  }
  ASSERT_EQ(batching_queue.size(), minimum_batch_size);

  // Dequeue all the tensors.
  std::pair<TensorNest, std::vector<Empty>> tensors =
      batching_queue.dequeue_many();
  ASSERT_EQ(batching_queue.size(), 0);
  ASSERT_FALSE(tensors.first.empty());
  ASSERT_EQ(tensors.second.size(), uint64_t(minimum_batch_size));

  // Push more tensors than the maximum batch size.
  for (int i = 0; i < maximum_batch_size + 1; i++) {
    TensorNest tensor_nest(torch::ones({2, 2, 2, 2}));
    batching_queue.enqueue({tensor_nest, Empty()});
  }
  ASSERT_EQ(batching_queue.size(), 65ll);

  // Dequeue tensors.
  tensors = batching_queue.dequeue_many();
  ASSERT_EQ(batching_queue.size(), 1);  // One tensor left in the queue.
  ASSERT_FALSE(tensors.first.empty());
  ASSERT_EQ(tensors.second.size(), uint64_t(maximum_batch_size));

  batching_queue.close();
  ASSERT_EQ(batching_queue.size(), 0);  // Closed queue should be empty.
}

TEST(BatchingQueue, TestBadDequeueMany) {
  int64_t minimum_batch_size = 32;
  BatchingQueue batching_queue(/*batch_dim=*/3, minimum_batch_size,
                               /*maximum_batch_size=*/64);

  // Push enough tensors in order to trigger enough_inputs_.notify_one() in
  // enqueue.
  for (int i = 0; i < minimum_batch_size; i++) {
    TensorNest tensor_nest(torch::ones({2, 2, 2, 2}));
    batching_queue.enqueue({tensor_nest, Empty()});
  }

  batching_queue.close();

  // Cannot retrieve data from closed queue.
  ASSERT_THROW({ batching_queue.dequeue_many(); }, py::stop_iteration);
}

TEST(ActorPool, TestArrayPbToNest1) {
  // The first two dimensions are T and B.
  torch::Tensor target = torch::zeros({1, 1, 1, 2});

  rpcenv::NDArray array;
  array.set_dtype(NPY_FLOAT);
  array.add_shape(1);
  array.add_shape(2);
  // 8 bytes of data (1 * 2 * 4), all full of zeros:
  // 1 -> shape of first dimension.
  // 2 -> shape of second dimension.
  // 4 -> number of bytes in an int32.
  const char* data = "\x00\x00\x00\x00\x00\x00\x00\x00";
  array.set_data(data);
  TensorNest actual = ActorPool::array_pb_to_nest(&array);
  ASSERT_TRUE(actual.front().equal(target));
}

TEST(ActorPool, TestArrayPbToNest2) {
  // The first two dimensions are T and B.
  torch::Tensor target = torch::ones({1, 1, 2, 2});

  rpcenv::NDArray array;
  // Fill up NDArray.
  ActorPool::fill_ndarray_pb(&array, torch::ones({2, 2}));
  TensorNest actual = ActorPool::array_pb_to_nest(&array);
  ASSERT_TRUE(actual.front().equal(target));
}

TEST(ActorPool, TestStepPbToNest) {
  // The following values are arbitrary.
  int observation_size = 5;
  bool done = true;
  float reward = 3.5f;
  int32_t episode_step = 42;
  float episode_return = 15.3f;

  TensorNest target = TensorNest(std::vector(
      {torch::ones({1, 1, observation_size}), torch::full({1, 1}, reward),
       torch::full({1, 1}, done, torch::dtype(torch::kUInt8)),
       torch::full({1, 1}, episode_step, torch::dtype(torch::kInt32)),
       torch::full({1, 1}, episode_return)}));

  rpcenv::Step step_pb;
  step_pb.set_reward(reward);
  step_pb.set_done(done);
  step_pb.set_episode_step(episode_step);
  step_pb.set_episode_return(episode_return);
  rpcenv::ArrayNest* obs = step_pb.mutable_observation();
  rpcenv::NDArray* array = obs->mutable_array();
  // Fill up observation.
  ActorPool::fill_ndarray_pb(array, torch::ones(observation_size));

  TensorNest actual = ActorPool::step_pb_to_nest(&step_pb);

  nest::Nest<bool> equal_nest = TensorNest::map2(
      [](const torch::Tensor& actual, const torch::Tensor& target) {
        return actual.equal(target);
      },
      actual, target);
  equal_nest.for_each([](const bool e) { ASSERT_TRUE(e); });
}

TEST(ActorPool, TestFillNdarrayPb) {
  rpcenv::NDArray array;
  torch::Tensor tensor = torch::ones({5, 3});
  int64_t start_dim = 0;

  ActorPool::fill_ndarray_pb(&array, tensor, start_dim);
  ASSERT_EQ(*array.mutable_data(),
            std::string(static_cast<const char*>(tensor.data_ptr()),
                        tensor.nbytes()));
  ASSERT_THAT(array.shape(), testing::ElementsAre(5, 3));
  ASSERT_EQ(array.dtype(), NPY_FLOAT);
}

}  // namespace
