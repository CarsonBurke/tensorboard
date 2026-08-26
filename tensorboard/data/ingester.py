# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Abstraction for data ingestion logic."""

import abc


class ReloadError(RuntimeError):
    """Raised when an ingester supports on-demand reload but it fails."""


class DataIngester(metaclass=abc.ABCMeta):
    """Link between a data source and a data provider.

    A data ingester starts a reload operation in the background and
    provides a data provider as a view.
    """

    @property
    @abc.abstractmethod
    def data_provider(self):
        """Returns a `DataProvider`.

        It may be an error to dereference this before `start` is called.
        """
        pass

    @abc.abstractmethod
    def start(self):
        """Starts ingesting data.

        This may start a background thread or process, and will return
        once communication with that task is established. It won't block
        forever as data is reloaded.

        Must only be called once.
        """

    def request_reload(self, timeout=None):
        """Request an immediate logdir rescan and wait until it finishes.

        The default implementation is a no-op. Implementations that poll
        for new runs should override this so a UI refresh can discover
        runs created after the last background cycle.

        Args:
          timeout: Optional seconds to wait. ``None`` waits until the
            reload finishes.

        Returns:
          True if a reload ran (or was already up to date), False if this
          ingester cannot honor the request.
        """
        del timeout
        return False
