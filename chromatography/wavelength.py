
import numpy as np

from vmk_spectrum2_wrapper.typing import Array, NanoMeter, Number

from .handlers import reduce_resolution


class WavelengthCalibration:

    def __init__(self, filename: str = '.wl', factor: int = 1) -> None:
        self.filename = filename
        self.factor = factor

        self._wavelength = None

    @property
    def wavelength(self) -> Array[float]:
        if self._wavelength is None:
            with open(self.filename, 'r') as file:
                data = np.genfromtxt(file, delimiter='\t')

            wavelength = data[:, 0]
            wavelength = reduce_resolution(wavelength, factor=self.factor)

            self._wavelength = wavelength

        return self._wavelength

    def transform(self, value: NanoMeter) -> Number:
        """Transform wavelength to number."""
        assert min(self.wavelength) <= value <= max(self.wavelength), f'Wavelength {value:.2f} [nm] is out of the range!'

        return np.argmin(abs(self.wavelength - value))
