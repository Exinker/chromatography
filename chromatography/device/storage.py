import time
from typing import Callable

import numpy as np

from vmk_spectrum2_wrapper.typing import Array
from vmk_spectrum2_wrapper.storage import BufferDeviceStorage


class DeviceStorage(BufferDeviceStorage):

    def __init__(self, buffer_size: int = 1, buffer_handler: Callable[[Array], Array] | None = None, signal_handler: Callable[[Array], Array] | None = None) -> None:
        super().__init__(buffer_size, buffer_handler)

        self._signal_handler = signal_handler
        self._signal = []

    # --------        data        --------
    @property
    def data(self) -> Array[int]:
        return np.array(self._data)

    @property
    def signal(self) -> Array[int]:
        return np.array(self._signal)

    def put(self, frame: Array[int]) -> None:
        """Добавить новый кадр `frame`."""

        # time
        time_at = time.perf_counter()

        if self._started_at is None:
            self._started_at = time_at

        self._finished_at = time_at

        # data
        if self.scale:
            frame = self.scale * frame

        if self.buffer_size == 1:  # если буфер размера `1`, то данные отправляюится сразу в `data`

            # buffer
            buffer = np.array(frame)
            if self.buffer_handler:
                buffer = self.buffer_handler(buffer)

            # data
            self._data.append(buffer)
            self._signal.append(
                self._signal_handler(buffer)
            )

        else:
            self._buffer.append(frame)

            if len(self.buffer) == self.buffer_size:  # если буфер заполнен, то ранные обрабатываются `handler`, передаются в `data` и буфер очищается

                try:
                    # buffer
                    buffer = np.array(self.buffer)
                    if self.buffer_handler:
                        buffer = self.buffer_handler(buffer)

                    # data
                    self._data.append(buffer)
                    self._signal.append(
                        self._signal_handler(buffer)
                    )

                finally:
                    self._buffer.clear()
