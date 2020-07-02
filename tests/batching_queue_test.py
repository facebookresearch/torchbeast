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
"""Tests for actorpool.BatchingQueue.
Basic functionalities actorpool.BatchingQueue are tested
in libtorchbeast/actorpool_test.cc.
"""

import threading
import time
import unittest

import numpy as np
import torch
import libtorchbeast


class BatchingQueueTest(unittest.TestCase):
    def test_bad_construct(self):
        with self.assertRaisesRegex(ValueError, "Min batch size must be >= 1"):
            libtorchbeast.BatchingQueue(
                batch_dim=3, minimum_batch_size=0, maximum_batch_size=1
            )

        with self.assertRaisesRegex(
            ValueError, "Max batch size must be >= min batch size"
        ):
            libtorchbeast.BatchingQueue(
                batch_dim=3, minimum_batch_size=1, maximum_batch_size=0
            )

    def test_multiple_close_calls(self):
        queue = libtorchbeast.BatchingQueue()
        queue.close()
        with self.assertRaisesRegex(RuntimeError, "Queue was closed already"):
            queue.close()

    def test_check_inputs(self):
        queue = libtorchbeast.BatchingQueue(batch_dim=2)
        with self.assertRaisesRegex(
            ValueError, "Enqueued tensors must have more than batch_dim =="
        ):
            queue.enqueue(torch.ones(5))
        with self.assertRaisesRegex(
            ValueError, "Cannot enqueue empty vector of tensors"
        ):
            queue.enqueue([])
        with self.assertRaisesRegex(
            libtorchbeast.ClosedBatchingQueue, "Enqueue to closed queue"
        ):
            queue.close()
            queue.enqueue(torch.ones(1, 1, 1))

    def test_simple_run(self):
        queue = libtorchbeast.BatchingQueue(
            batch_dim=0, minimum_batch_size=1, maximum_batch_size=1
        )

        inputs = torch.zeros(1, 2, 3)
        queue.enqueue(inputs)
        batch = next(queue)
        np.testing.assert_array_equal(batch, inputs)

    def test_batched_run(self, batch_size=2):
        queue = libtorchbeast.BatchingQueue(
            batch_dim=0, minimum_batch_size=batch_size, maximum_batch_size=batch_size
        )

        inputs = [torch.full((1, 2, 3), i) for i in range(batch_size)]

        def enqueue_target(i):
            while queue.size() < i:
                # Make sure thread i calls enqueue before thread i + 1.
                time.sleep(0.05)
            queue.enqueue(inputs[i])

        enqueue_threads = []
        for i in range(batch_size):
            enqueue_threads.append(
                threading.Thread(
                    target=enqueue_target, name=f"enqueue-thread-{i}", args=(i,)
                )
            )

        for t in enqueue_threads:
            t.start()

        batch = next(queue)
        np.testing.assert_array_equal(batch, torch.cat(inputs))

        for t in enqueue_threads:
            t.join()


class BatchingQueueProducerConsumerTest(unittest.TestCase):
    def test_many_consumers(
        self, enqueue_threads_number=16, repeats=100, dequeue_threads_number=64
    ):
        queue = libtorchbeast.BatchingQueue(batch_dim=0)

        lock = threading.Lock()
        total_batches_consumed = 0

        def enqueue_target(i):
            for _ in range(repeats):
                queue.enqueue(torch.full((1, 2, 3), i))

        def dequeue_target():
            nonlocal total_batches_consumed
            for batch in queue:
                batch_size, *_ = batch.shape
                with lock:
                    total_batches_consumed += batch_size

        enqueue_threads = []
        for i in range(enqueue_threads_number):
            enqueue_threads.append(
                threading.Thread(
                    target=enqueue_target, name=f"enqueue-thread-{i}", args=(i,)
                )
            )

        dequeue_threads = []
        for i in range(dequeue_threads_number):
            dequeue_threads.append(
                threading.Thread(target=dequeue_target, name=f"dequeue-thread-{i}")
            )

        for t in enqueue_threads + dequeue_threads:
            t.start()

        for t in enqueue_threads:
            t.join()

        queue.close()

        for t in dequeue_threads:
            t.join()

        self.assertEqual(total_batches_consumed, repeats * enqueue_threads_number)


if __name__ == "__main__":
    unittest.main()
