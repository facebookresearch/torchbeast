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

import queue

Full = queue.Full
Empty = queue.Empty


class Closed(Exception):
    pass


class Queue(queue.Queue):
    """A closable queue."""

    def _init(self, maxsize):
        super()._init(maxsize)
        self._closed = False

    def closed(self):
        with self.mutex:
            return self._closed

    def close(self):
        with self.mutex:
            self._closed = True
            self.not_full.notify_all()
            self.not_empty.notify_all()
            self.all_tasks_done.notify_all()

    def _put(self, item):  # Called under self.mutex
        if self._closed:
            raise Closed
        super()._put(item)

    def _get(self):  # Called under self.mutex
        if self._closed:
            raise Closed
        return super()._get()

    def _qsize(self):  # Called under self.mutex
        if self._closed:
            raise Closed
        return super()._qsize()
