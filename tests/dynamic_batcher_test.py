# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for actorpool.DynamicBatcher."""

import threading
import time
import unittest

import numpy as np
import torch
import libtorchbeast


_BROKEN_PROMISE_MESSAGE = (
    "The associated promise has been destructed prior"
    " to the associated state becoming ready."
)


class DynamicBatcherTest(unittest.TestCase):
    def test_simple_run(self):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=0, minimum_batch_size=1, maximum_batch_size=1
        )

        inputs = torch.zeros(1, 2, 3)
        outputs = torch.ones(1, 42, 3)

        def target():
            np.testing.assert_array_equal(batcher.compute(inputs), outputs)

        t = threading.Thread(target=target, name="compute-thread")
        t.start()

        batch = next(batcher)
        np.testing.assert_array_equal(batch.get_inputs(), inputs)
        batch.set_outputs(outputs)

        t.join()

    def test_timeout(self):
        timeout_ms = 300
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=0,
            minimum_batch_size=5,
            maximum_batch_size=5,
            timeout_ms=timeout_ms,
        )

        inputs = torch.zeros(1, 2, 3)
        outputs = torch.ones(1, 42, 3)

        def compute_target():
            batcher.compute(inputs)

        compute_thread = threading.Thread(target=compute_target, name="compute-thread")
        compute_thread.start()

        start_waiting_time = time.time()
        # Wait until approximately timeout_ms.
        batch = next(batcher)
        waiting_time_ms = (time.time() - start_waiting_time) * 1000
        # Timeout has expired and the batch of size 1 (< minimum_batch_size)
        # has been consumed.
        batch.set_outputs(outputs)

        compute_thread.join()

        self.assertTrue(timeout_ms <= waiting_time_ms <= timeout_ms + timeout_ms / 10)

    def test_batched_run(self, batch_size=10):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=0, minimum_batch_size=batch_size, maximum_batch_size=batch_size
        )

        inputs = [torch.full((1, 2, 3), i) for i in range(batch_size)]
        outputs = torch.ones(batch_size, 42, 3)

        def target(i):
            while batcher.size() < i:
                # Make sure thread i calls compute before thread i + 1.
                time.sleep(0.05)

            np.testing.assert_array_equal(
                batcher.compute(inputs[i]), outputs[i : i + 1]
            )

        threads = []
        for i in range(batch_size):
            threads.append(
                threading.Thread(target=target, name=f"compute-thread-{i}", args=(i,))
            )

        for t in threads:
            t.start()

        batch = next(batcher)

        batched_inputs = batch.get_inputs()
        np.testing.assert_array_equal(batched_inputs, torch.cat(inputs))
        batch.set_outputs(outputs)

        for t in threads:
            t.join()

    def test_dropped_batch(self):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=0, minimum_batch_size=1, maximum_batch_size=1
        )

        inputs = torch.zeros(1, 2, 3)

        def target():
            with self.assertRaisesRegex(
                libtorchbeast.AsyncError, _BROKEN_PROMISE_MESSAGE
            ):
                batcher.compute(inputs)

        t = threading.Thread(target=target, name="compute-thread")
        t.start()

        next(batcher)  # Retrieves but doesn't keep the batch object.
        t.join()

    def test_check_outputs1(self):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=2, minimum_batch_size=1, maximum_batch_size=1
        )

        inputs = torch.zeros(1, 2, 3)

        def target():
            batcher.compute(inputs)

        t = threading.Thread(target=target, name="compute-thread")
        t.start()

        batch = next(batcher)

        with self.assertRaisesRegex(ValueError, "output shape must have at least"):
            outputs = torch.ones(1)
            batch.set_outputs(outputs)

        # Set correct outputs so the thread can join.
        batch.set_outputs(torch.ones(1, 1, 1))
        t.join()

    def test_check_outputs2(self):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=2, minimum_batch_size=1, maximum_batch_size=1
        )

        inputs = torch.zeros(1, 2, 3)

        def target():
            batcher.compute(inputs)

        t = threading.Thread(target=target, name="compute-thread")
        t.start()

        batch = next(batcher)

        with self.assertRaisesRegex(
            ValueError,
            "Output shape must have the same batch dimension as the input batch size.",
        ):
            # Dimenstion two of the outputs is != from the size of the batch (3 != 1).
            batch.set_outputs(torch.ones(1, 42, 3))

        # Set correct outputs so the thread can join.
        batch.set_outputs(torch.ones(1, 1, 1))
        t.join()

    def test_multiple_set_outputs_calls(self):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=0, minimum_batch_size=1, maximum_batch_size=1
        )

        inputs = torch.zeros(1, 2, 3)
        outputs = torch.ones(1, 42, 3)

        def target():
            batcher.compute(inputs)

        t = threading.Thread(target=target, name="compute-thread")
        t.start()

        batch = next(batcher)
        batch.set_outputs(outputs)
        with self.assertRaisesRegex(RuntimeError, "set_outputs called twice"):
            batch.set_outputs(outputs)

        t.join()


class DynamicBatcherProducerConsumerTest(unittest.TestCase):
    def test_many_consumers(
        self,
        minimum_batch_size=1,
        compute_thread_number=64,
        repeats=100,
        consume_thread_number=16,
    ):
        batcher = libtorchbeast.DynamicBatcher(
            batch_dim=0, minimum_batch_size=minimum_batch_size
        )

        lock = threading.Lock()
        total_batches_consumed = 0

        def compute_thread_target(i):
            for _ in range(repeats):
                inputs = torch.full((1, 2, 3), i)
                batcher.compute(inputs)

        def consume_thread_target():
            nonlocal total_batches_consumed
            for batch in batcher:
                inputs = batch.get_inputs()
                batch_size, *_ = inputs.shape
                batch.set_outputs(torch.ones_like(inputs))
                with lock:
                    total_batches_consumed += batch_size

        compute_threads = []
        for i in range(compute_thread_number):
            compute_threads.append(
                threading.Thread(
                    target=compute_thread_target, name=f"compute-thread-{i}", args=(i,)
                )
            )

        consume_threads = []
        for i in range(consume_thread_number):
            consume_threads.append(
                threading.Thread(
                    target=consume_thread_target, name=f"consume-thread-{i}"
                )
            )

        for t in compute_threads + consume_threads:
            t.start()

        for t in compute_threads:
            t.join()

        # Stop iteration in all consume_threads.
        batcher.close()

        for t in consume_threads:
            t.join()

        self.assertEqual(total_batches_consumed, compute_thread_number * repeats)


if __name__ == "__main__":
    unittest.main()
