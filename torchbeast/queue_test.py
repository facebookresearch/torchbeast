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

import unittest
import threading

from torchbeast import queue


class QueueTest(unittest.TestCase):
    def test_simple(self):
        q = queue.Queue()
        q.put(1)
        q.put(2)
        self.assertEqual(q.get(), 1)

        q.close()
        with self.assertRaises(queue.Closed):
            q.get()

    def test_thread(self):
        q = queue.Queue(maxsize=1)

        def target():
            with self.assertRaises(queue.Closed):
                q.get(timeout=5.0)

        thread = threading.Thread(target=target)
        thread.start()

        q.close()
        thread.join()  # Should take way less than 5 sec.


if __name__ == "__main__":
    unittest.main()
