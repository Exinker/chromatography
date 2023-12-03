import os
import time
from decimal import Decimal
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from IPython import display
from tqdm import tqdm

from libspectrum2_wrapper.alias import Array, Hz, MilliSecond, Number, Second
from libspectrum2_wrapper.device import Device, DeviceEthernetConfig
from libspectrum2_wrapper.storage import DeviceStorage, BufferDeviceStorage

from .analyte import Channels, Channel, Ratio, handle_analyte_signal
from .handlers import handle_dark_data, handle_base_data, handle_absorbance_signal
from .wavelength import WavelengthCalibration


CMAP = plt.get_cmap("tab10")


class ChromDeviceStorage(BufferDeviceStorage):

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


class ChromConfig:

    def __init__(self, omega: Hz, tau: MilliSecond = .4, factor: int = 8) -> None:
        assert 0.1 <= omega <= 100, 'Частота регистрации `omega` должна лежать в диапазоне [0.1; 100] Гц!'
        assert int(np.round(1000 * tau).astype(int)) % 100 == 0, 'Базовое время экспозиции `tau` должно задано с точностью до 0.1 мс!'
        assert .4 <= tau <= .6, 'Базовое время экспозиции `tau` должно лежать в диапазоне [0.4; 0.6] мс!'
        assert isinstance(factor, int), '`factor` должен быть целым числом!'
        assert 2048 % factor == 0, '`factor` должен быть кратен количеству фотоячеек!'  # FIXME: remove 2048!

        self._omega = omega
        self._tau = tau
        self._buffer_size = self.calculate_buffer_size(omega=omega, tau=tau)
        self._factor = factor

    @property
    def omega(self) -> float:
        """Частота регистрации (Гц)."""
        return self._omega

    @property
    def tau(self) -> MilliSecond:
        """Базовое время экспозиции (мс)."""
        return self._tau

    @property
    def buffer_size(self) -> int:
        """Количество накоплений во времени."""
        return self._buffer_size

    @property
    def factor(self) -> Number:
        """Количество накоплений в пространстве."""
        return self._factor

    @staticmethod
    def calculate_buffer_size(omega: Hz, tau: MilliSecond) -> int:
        """Рассчитать размер буфера."""
        assert Decimal(1e+6) / Decimal(omega) % Decimal(int(1000*tau)) == 0, 'Частота регистрации `omega` должна быть кратна базовому времени экспозиции `tau`!'

        return int(Decimal(1e+6) / Decimal(omega) / Decimal(int(1000*tau)))


class Chrom:
    """Интерфейс для работы с устройством."""

    def __init__(self, config: ChromConfig) -> None:

        # config
        self.config = config

        # device
        self.device = Device(
            storage=BufferDeviceStorage(),
        )
        self.device.create(
            config=DeviceEthernetConfig(
                ip='10.116.220.2',
            ),
        )
        self.device.connect()

        #
        self._wavelength = None
        self._dark_data = None
        self._base_data = None

    # --------        dark_data        --------
    @property
    def dark_data(self) -> Array[float]:
        """Темновой сигнал."""
        assert self._dark_data is not None, 'Calibrate device to dark data!'

        return self._dark_data

    def calibrate_dark_data(self, n_frames: int, filename: str = 'dark.data', verbose: bool = False, show: bool = False) -> None:
        """Калибровать устройство на темновой сигнал по `n_frames` накоплений."""

        self.device.set_storage(
            storage=BufferDeviceStorage(
                buffer_size=n_frames,
                buffer_handler=partial(handle_dark_data, factor=self.config.factor),
            ),
        )

        data = self._read(n_frames, verbose=verbose)
        data = data.reshape(-1,)

        # verbose
        if verbose:
            pass

        # show
        if show:
            figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

            x = np.arange(1, len(data)+1)
            y = data.reshape(-1,)
            plt.plot(
                x, y,
                color='black', linestyle='-',
            )

            plt.xlabel('number')
            plt.ylabel('$I_{d}$, %')

            plt.grid(color='grey', linestyle=':')

            plt.show()

        # save
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as file:
            np.savetxt(file, data)

        #
        self._dark_data = data

    def load_dark_data(self, filename: str = 'dark.data') -> None:
        """Калибровать устройство на темновой сигнал из файла `filename`."""

        # load
        filepath = os.path.join('data', filename)
        with open(filepath, 'r') as file:
            data = np.genfromtxt(file)

        #
        self._dark_data = data

    # --------        base_data        --------
    @property
    def base_data(self) -> Array[float]:
        assert self._base_data is not None, 'Calibrate to base data!'

        return self._base_data

    def read_base_data(self, n_frames: int, recycle: bool = False, verbose: bool = False, show: bool = False) -> Array[float]:
        """Читать спектр источника излучения"""

        self.device.set_storage(
            storage=BufferDeviceStorage(
                buffer_size=n_frames,
                buffer_handler=partial(handle_base_data, factor=self.config.factor, dark_data=self.dark_data),
            ),
        )

        #
        try:
            while True:
                # read
                data = self._read(n_frames, verbose=verbose)
                data = data.reshape(-1,)

                # show
                if show:
                    display.clear_output(wait=True)

                    #
                    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

                    x = np.arange(1, len(data)+1) if self.wavelength is None else self.wavelength
                    y = data.reshape(-1,)
                    plt.plot(
                        x, y,
                        color='black', linestyle='-',
                    )

                    plt.xlabel('number' if self.wavelength is None else '$\lambda$, nm')
                    plt.ylabel('$I_{0}$, %')

                    plt.grid(color='grey', linestyle=':')

                    plt.pause(0.001)

                # break
                if not recycle:
                    break

        except KeyboardInterrupt as error:
            pass

        finally:
            return data

    def calibrate_base_data(self, n_frames: int, filename: str = 'base.data', verbose: bool = False, show: bool = False) -> None:
        """Калибровать устройство на источник излучения."""

        # read
        data = self.read_base_data(
            n_frames=n_frames,
            verbose=verbose,
            show=show,
        )

        # save
        filepath = os.path.join('data', filename)
        with open(filepath, 'w') as file:
            np.savetxt(file, data)

        #
        self._base_data = data

    def load_base_data(self, filename: str = 'base.data') -> None:
        """Калибровать устройство на источник излучения из файла `filename`."""

        # load
        filepath = os.path.join('data', filename)
        with open(filepath, 'r') as file:
            data = np.genfromtxt(file)

        #
        self._base_data = data

    # --------        wavelength        --------
    @property
    def wavelength(self) -> Array[float]:

        if self._wavelength is None:
            self._wavelength = WavelengthCalibration(
                factor=self.config.factor,
            ).wavelength

        return self._wavelength

    # --------        read        --------
    def read(self, n_frames: int | None = None, duration: Second | None = None, channels: Channels | None = None, show: bool = False) -> Array[float]:
        """Читать `n_frames` кадров (или в течение `duration` секунд)."""
        config = self.config

        # n_frames
        n_frames = n_frames or self.calculate_n_frames(duration)
        assert n_frames or duration, 'Укажите время измерения `duration` или количество накоплений `n_frames`'

        n_frames = (n_frames//config.buffer_size) * config.buffer_size  # FIXME: 
        assert n_frames < 2**24, 'Общее количество накполений `n_frames` должно быть менее 2**24!'

        # setup
        self.device.set_storage(
            storage=ChromDeviceStorage(
                buffer_size=config.buffer_size,
                buffer_handler=partial(handle_absorbance_signal, factor=config.factor, dark_data=self.dark_data, base_data=self._base_data),
                signal_handler=partial(handle_analyte_signal, channels=channels),
            ),
        )

        data = self._read(
            n_frames=n_frames,
            channels=channels,
            show=show,
        )

        # save
        filepath = os.path.join('data', 'data.data')
        with open(filepath, 'w') as file:
            np.savetxt(file, data)

        #
        return data

    # --------        functions        --------
    def calculate_signal(self, absorbance: Array[float], channels: Channels, save: bool = True) -> Array[float]:
        """"""
        config = self.config
        
        n_counts, n_numbers = absorbance.shape
        n_channels = len(channels)

        # signal
        signal = np.zeros((n_counts, n_channels))
        for t in range(n_counts):
            signal[t,:] = handle_analyte_signal(absorbance[t, :], channels=channels)

        # save
        if save:
            filepath = os.path.join('data', 'signal.csv')
            pd.DataFrame(
                signal,
                index=config.buffer_size * 1e-3*config.tau * np.arange(n_counts),
                columns=[
                    f'{channel}'
                    for i, channel in enumerate(channels)
                ],
            ).to_csv(
                filepath,
            )

        #
        return signal

    def calculate_n_frames(self, duration: Second) -> int:
        """Расчитать количество накоплений для регистрации в течение `duration` секунд."""
        return int(duration * 1000 / self.config.tau)

    # --------                --------
    def _read(self, n_frames: int, channels: Channels | None = None, handler: Callable[[Array[float]], Array[float]] | None = None, verbose: bool = False, show: bool = False) -> Array[float]:
        device = self.device
        storage = self.device.storage
        config = self.config

        total_duration: Second = n_frames * 1e-3*config.tau  # duration of reading [sec]

        # setup
        if verbose:
            progress = tqdm(
                total=n_frames,
                unit='frame',
                ncols=60,
                bar_format='{l_bar}{bar}'
            )
            progress.update(0)
            progress.refresh()

        # update
        try:
            device.set_exposure(config.tau)
            device.read(n_frames)

            while True:
                buffer_capacity = len(storage.buffer)
                storage_capacity = len(storage.data)

                if verbose:
                    progress.n = storage_capacity * storage.buffer_size + buffer_capacity
                    progress.refresh()

                if show:
                    display.clear_output(wait=True)

                    #
                    figure, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))

                    # ax_left
                    plt.sca(ax_left)

                    if storage_capacity > 0:
                        datum = storage.data[-1]
                        datum = handler(datum) if handler else datum

                        number = np.arange(1, len(datum)+1) if self._wavelength is None else self._wavelength

                        x = number
                        y = datum
                        plt.plot(
                            x, y,
                            color='black', linestyle='-',
                        )

                        for i, channel in enumerate(channels):

                            if isinstance(channel, Channel):
                                lb, ub = channel.bounds
                                index = np.arange(lb, ub)

                                plt.fill_between(
                                    number[index], datum[index], y2=0,
                                    step='mid',
                                    facecolor=CMAP(i), edgecolor='none',
                                    alpha=.2,
                                )

                    content = [
                        fr'$\omega$: {config.omega} [Hz]',
                        fr'',
                        fr'$\delta{{t}}$: {storage.buffer_size}',
                        fr'$\delta{{n}}$: {config.factor}',
                    ]
                    plt.text(
                        .95, .95,
                        '\n'.join(content),
                        fontsize=12,
                        ha='right', va='top',
                        transform=plt.gca().transAxes,
                    )

                    plt.xlabel('number' if self.wavelength is None else '$\lambda$, nm')
                    plt.ylabel('$A$')

                    plt.grid(color='grey', linestyle=':')

                    # ax_right
                    plt.sca(ax_right)

                    if storage_capacity > 0:
                        signal = np.array(storage._signal)

                        for i, channel in enumerate(channels):

                            if isinstance(channel, Channel):
                                x = np.arange(signal.shape[0]) * (storage.buffer_size * 1e-3*config.tau)
                                y = signal[:, i]

                                plt.plot(
                                    x, y,
                                    color=CMAP(i), linestyle='-',
                                    alpha=.75,
                                    label=f'{channel}',
                                )

                    left = total_duration - storage.duration
                    content = '\n'.join([
                        f'completed: {100 * storage_capacity / (n_frames//storage.buffer_size):.2f}%',
                        'time left: {hours}:{minutes}:{seconds}'.format(
                            hours=f'{str(int(left // 3600)):0>2s}',
                            minutes=f'{str(int(left // 60)):0>2s}',
                            seconds=f'{str(int(left % 60)):0>2s}',
                        ) if left > 0 else '',
                    ])
                    plt.text(
                        .95, .95,
                        content,
                        ha='right', va='top',
                        transform=plt.gca().transAxes,
                    )

                    plt.xlabel('time, s')
                    plt.ylabel('$A$')

                    plt.grid(color='grey', linestyle=':')
                    plt.legend(loc='lower left')

                    # ax_right_ratio
                    plt.sca(ax_right.twinx())

                    if storage_capacity > 0:
                        signal = np.array(storage._signal)

                        for i, channel in enumerate(channels):

                            if isinstance(channel, Ratio):
                                x = np.arange(signal.shape[0]) * (storage.buffer_size * 1e-3*config.tau)
                                y = signal[:, i]

                                plt.plot(
                                    x, y,
                                    color=CMAP(i), linestyle=':',
                                    alpha=.75,
                                    label=f'{channel}',
                                )

                    plt.legend(loc='lower right')

                    #
                    plt.pause(.001)

                else:
                    time.sleep(0.001)

                if storage_capacity == n_frames//storage.buffer_size:
                    break

            else:
                if verbose:
                    progress.close()

                if show:
                    figure.clear()

            #
            data = np.array(storage.data)
            data = handler(data) if handler else data

            return data

        finally:
            storage._data.clear()
            # storage._signal.clear()
