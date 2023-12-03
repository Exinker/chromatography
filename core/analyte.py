
import numpy as np

from libspectrum2_wrapper.alias import Array, NanoMeter, Number

from .wavelength import WavelengthCalibration



class Channel:

    def __init__(self, wavelength: NanoMeter, position: Number, interval: Number = 3) -> None:
        self.wavelength = wavelength
        self.position = position
        self.interval = interval

        self._bounds = None

    @property
    def bounds(self) -> tuple[Number, Number]:
        if self._bounds is None:
            interval = (self.interval-1) // 2
            lb = self.position - interval
            ub = self.position + interval

            self._bounds = lb, ub

        return self._bounds

    def __repr__(self) -> str:
        cls = self.__class__

        return f'{cls.__name__}(w={self.wavelength})'

    def __str__(self) -> str:
        return fr'$\alpha_{{{self.wavelength}}}$'


class Ratio:

    def __init__(self, items: tuple[int, int], threshold: float):
        self.items = items
        self.threshold = threshold

    def __str__(self) -> str:
        left, right = self.items

        return fr'$\beta({{{left}}} / {{{right}}})$'


class Channels:

    def __init__(self, __items: list[tuple[NanoMeter, Number]], factor: int):

        # wavelength calibration
        calibration = WavelengthCalibration(
            factor=factor,
        )

        # channels
        self._items = []
        for wavelength, interval in __items:
            item = Channel(
                wavelength=wavelength,
                position=calibration.transform(wavelength),
                interval=interval,
            )

            self._items.append(item)

    # --------        ratios        --------
    @property
    def ratios(self) -> list[Ratio]:
        return self._ratios

    def add_ratio(self, __items: tuple[int, int], threshold: float = 0.001):
        """Добавить отношение каналов."""

        ratio = Ratio(
            __items,
            threshold=threshold,
        )

        self._items.append(ratio)

        return self

    # --------                --------
    def __getitem__(self, i: int) -> Channel:
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)


def truncate(data: float | Array[float], threshold: float) -> Array[float]:
    """Обрезать сигнал, выходящий за уровень `threshold`."""
    if data < threshold:
        return np.nan

    return data


def handle_analyte_signal(absorbance: Array[float], channels: Channels | None) -> Array[float]:
    """Расчитать аналитический сигнал для каждого канала."""
    if channels is None:
        return absorbance

    n_channels = len(channels)

    signal = np.zeros(n_channels, )
    for i, channel in enumerate(channels):

        if isinstance(channel, Channel):
            lb, ub = channel.bounds
            signal[i] = np.mean(absorbance[lb:ub])

        if isinstance(channel, Ratio):
            left, right = channel.items
            threshold = channel.threshold

            signal[i] = truncate(signal[left-1], threshold=threshold) / truncate(signal[right-1], threshold=threshold)

    return signal


def calulate_analyte_signal(absorbance: Array[float], channels: Channels | None) -> Array[float]:
    """Расчитать аналитический сигнал для каждого канала."""
    if channels is None:
        return absorbance

    n_times, n_numbers = absorbance.shape
    n_channels = len(channels)

    signal = np.zeros(n_times, n_channels)
    for i, channel in enumerate(channels):

        if isinstance(channel, Channel):
            lb, ub = channel.bounds
            signal[:, i] = np.mean(absorbance[:, lb:ub], axis=1)

        if isinstance(channel, Ratio):
            left, right = channel.items
            threshold = channel.threshold

            signal[i] = truncate(signal[left], threshold=threshold) / truncate(signal[right], threshold=threshold)

    return signal
