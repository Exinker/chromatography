from decimal import Decimal

import numpy as np

from vmk_spectrum2_wrapper.typing import Hz, MilliSecond, Number


class DeviceConfig:

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
