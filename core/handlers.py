
import numpy as np

from libspectrum2_wrapper.alias import Array


def none(data: Array[int]) -> Array[int]:
    return data


def average(data: Array[int], factor: int) -> Array[int]:
    """Усреднить `data` по времени и пространству в `factor` раз."""

    # average by times
    match data.ndim:
        case 1:
            pass
        case 2:
            data = np.mean(data, axis=0)

    # average by numbers
    if factor > 1:
        data = reduce_resolution(data, factor=factor)

    #
    return data


def reduce_resolution(data: Array[int], factor: int) -> Array[int]:
    """Снизить разрешение `data` в `factor` раз."""
    assert data.ndim == 1
    assert len(data) % factor == 0

    return data.reshape(-1, factor).mean(axis=1)


def handle_dark_data(data: Array[int], factor: int) -> Array[float]:
    """Обработчик темнового сигнала."""

    data = average(data, factor=factor)

    return data


def handle_base_data(data: Array[int], factor: int, dark_data: Array[float]) -> Array[float]:
    """Обработчик сигнала источника излучения."""

    data = average(data, factor=factor)
    data = data - dark_data

    return data


def handle_absorbance_signal(data: Array[float], factor: int, dark_data: Array[float], base_data: Array[float]) -> Array[float]:
    """Обработчик сигнала поглощения."""

    data = average(data, factor=factor)
    data = data - dark_data
    data = np.log10(base_data / data)

    return data

