# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tuning a qudit's frequency and turning a coupling on.

A qudit's ``tuning`` (a ``FluxTuning``) says how an amplitude, an offset
from its idle point in the tuning's ``unit``, moves its frequencies. A
coupling's ``control`` says what turns its interaction on: a
``ChargeCoupling`` is a pair coupled through charge whose interaction is
made by tuning the two qudits, one side each; an ``ExchangeControl`` is the
pair's own control, its exchange tabulated against its amplitude.
``PhysicalQudit.frequency_at`` and ``Coupling.at`` read them without
compiling anything.

The module loads only when a quantum system declares one of them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidValueError, Problem
from .base import CouplingControl, Tuning, _Field, _finite

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from .base import Coupling

__all__ = ["ChargeCoupling", "CouplingPoint", "ExchangeControl", "FluxTuning"]

# Charge states -25..25, as transmon_model.TransmonCircuit diagonalizes them.
_CHARGE_CUTOFF = 25
# Operating points diagonalized at once: a batch of 51 x 51 matrices.
_BATCH = 512


class FluxTuning(Tuning):
    """A qudit tuned by the flux through its SQUID: an asymmetric-SQUID
    transmon of charging energy ``ec_hz``, largest Josephson energy
    ``ej_max_hz`` and SQUID asymmetry ``asymmetry`` (d = (E_J2 - E_J1) /
    (E_J1 + E_J2)), idling at the flux ``idle_flux`` in flux quanta (0 and
    1/2 are its sweet spots).

    An amplitude is a flux offset from the idle point, in flux quanta
    (``unit`` ``"flux_quantum"``), and signed: the frequency is even about a
    sweet spot, so ``a`` and ``-a`` give the same one there, which is why a
    net-zero pulse returns the qudit to its frequency. The levels are the
    Cooper-pair box's, H = 4 E_C n^2 - E_J(Phi) cos(phi) with E_J(Phi) =
    E_J,max sqrt(cos^2(pi Phi) + d^2 sin^2(pi Phi)), diagonalized in the
    charge basis.

    ``source`` says where the values come from: a citation, or the
    calibration entries they were read from (``Calibration.entry`` gives each
    with its kind, source and method).
    """

    __slots__ = ()
    _fields = ("ec_hz", "ej_max_hz", "asymmetry", "idle_flux", "source")
    unit = "flux_quantum"

    def __init__(
        self,
        *,
        ec_hz: float,
        ej_max_hz: float,
        asymmetry: float,
        idle_flux: float,
        source: str = "",
    ) -> None:
        object.__setattr__(self, "_values", (ec_hz, ej_max_hz, asymmetry, idle_flux, source))

    if TYPE_CHECKING:

        @property
        def ec_hz(self) -> float:
            """The charging energy E_C / h."""
            ...

        @property
        def ej_max_hz(self) -> float:
            """The Josephson energy E_J,max / h of the SQUID at zero flux."""
            ...

        @property
        def asymmetry(self) -> float:
            """The SQUID asymmetry d."""
            ...

        @property
        def idle_flux(self) -> float:
            """The flux (flux quanta) the qudit idles at."""
            ...

        @property
        def source(self) -> str:
            """Where the values come from."""
            ...

    else:
        ec_hz, ej_max_hz, asymmetry, idle_flux, source = map(_Field, range(5))

    def frequency_at(self, amplitude: Any) -> Any:
        """The 0-1 frequency (Hz) at the flux offset *amplitude* from the
        idle point: a float, or an array for an array of amplitudes."""
        energies, _ = self._spectrum(amplitude, 2)
        return _scalar(energies[..., 1], amplitude)

    def anharmonicity_at(self, amplitude: Any) -> Any:
        """The anharmonicity E_12 - E_01 (Hz) at *amplitude*."""
        energies, _ = self._spectrum(amplitude, 3)
        return _scalar(energies[..., 2] - 2.0 * energies[..., 1], amplitude)

    def amplitude_for(self, frequency_hz: Any) -> Any:
        """The flux offset from the idle point that puts the 0-1 frequency at
        *frequency_hz*, on the branch between the two sweet spots (fluxes 0
        and 1/2) the idle point is on; its negative gives the same frequency
        when the qudit idles at a sweet spot. A frequency the qudit cannot
        reach is an ``InvalidValueError``."""
        import numpy as np

        from .transmon_model import TransmonCircuit

        circuit = TransmonCircuit(self._values[0], self._values[1], self._values[2])
        idle = float(self._values[3])
        given = np.asarray(frequency_hz, dtype=float)
        found = np.array([circuit.flux_for_f01(float(f)) - idle for f in given.reshape(-1)])
        return _scalar(found.reshape(given.shape), frequency_hz)

    def _spectrum(
        self, amplitude: Any, count: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The first *count* energies above the ground state (Hz) and the
        charge matrix ``<i|n|j>`` at each amplitude, with the signs that make
        ``<i|n|i+1>`` positive, as ``TransmonCircuit.eigensystem`` gives them."""
        import numpy as np

        ec, ej_max, d, idle = (float(v) for v in self._values[:4])
        flux = idle + np.asarray(amplitude, dtype=float)
        flat = flux.reshape(-1)
        n = np.arange(-_CHARGE_CUTOFF, _CHARGE_CUTOFF + 1, dtype=float)
        size = n.size
        rows = np.arange(size)
        energies = np.empty((flat.size, count))
        charge = np.empty((flat.size, count, count))
        for start in range(0, flat.size, _BATCH):
            x = math.pi * flat[start : start + _BATCH]
            ej = ej_max * np.sqrt(np.cos(x) ** 2 + d**2 * np.sin(x) ** 2)
            h = np.zeros((x.size, size, size))
            h[:, rows, rows] = 4.0 * ec * n**2
            h[:, rows[:-1], rows[1:]] = -0.5 * ej[:, None]
            h[:, rows[1:], rows[:-1]] = -0.5 * ej[:, None]
            values, vectors = np.linalg.eigh(h)
            vectors = vectors[:, :, :count]
            for i in range(1, count):
                overlap = np.einsum("pk,k,pk->p", vectors[:, :, i - 1], n, vectors[:, :, i])
                vectors[:, :, i] *= np.where(overlap < 0, -1.0, 1.0)[:, None]
            energies[start : start + x.size] = values[:, :count] - values[:, :1]
            charge[start : start + x.size] = np.einsum("pki,k,pkj->pij", vectors, n, vectors)
        return (
            energies.reshape(flux.shape + (count,)),
            charge.reshape(flux.shape + (count, count)),
        )

    def _problems(self, where: str) -> list[Problem]:
        problems = []
        for name, value, ok in (
            ("ec_hz", self._values[0], _finite(self._values[0]) and self._values[0] > 0),
            ("ej_max_hz", self._values[1], _finite(self._values[1]) and self._values[1] > 0),
            ("asymmetry", self._values[2], _finite(self._values[2]) and 0 <= self._values[2] <= 1),
            ("idle_flux", self._values[3], _finite(self._values[3])),
        ):
            if not ok:
                what = "a finite asymmetry in [0, 1]" if name == "asymmetry" else None
                problems.append(
                    Problem(
                        f"{value!r} is not {what or _WHAT[name]}",
                        path=f"{where}.{name}",
                        value=value,
                    )
                )
        return problems


_WHAT = {
    "ec_hz": "a finite energy above 0 Hz",
    "ej_max_hz": "a finite energy above 0 Hz",
    "idle_flux": "a finite flux in flux quanta",
}


class ChargeCoupling(CouplingControl):
    """A pair coupled through charge, g_c n_a n_b, whose interaction is made
    by tuning the two qudits: the coupling's control is the two qudits' own
    tunings, one side each, so an amplitude on a side is that qudit's
    tuning's (flux quanta for a ``FluxTuning``).

    ``charge_coupling_hz`` is g_c; ``tunings`` are the qudits' tunings in
    the order of the coupling's ``qudits``, each the one its
    ``PhysicalQudit.tuning`` gives (building the quantum system checks they
    agree). In the rotating-wave approximation the levels i, i + 1 of
    qudit a exchange with j + 1, j of qudit b at g_c ``<i|n_a|i+1>
    <j+1|n_b|j>``, the charge matrix elements at the qudits' operating
    points; the 0-1 exchange J_qq = g_c ``<0|n_a|1> <0|n_b|1>`` at the idle
    points is the coupling's ``strength_hz``. ``source`` says where g_c
    comes from.
    """

    __slots__ = ()
    _fields = ("charge_coupling_hz", "tunings", "source")
    tunes_qudits = True

    def __init__(
        self, *, charge_coupling_hz: float, tunings: Sequence[Tuning], source: str = ""
    ) -> None:
        object.__setattr__(self, "_values", (charge_coupling_hz, tuple(tunings), source))

    if TYPE_CHECKING:

        @property
        def charge_coupling_hz(self) -> float:
            """The charge coupling g_c / h."""
            ...

        @property
        def tunings(self) -> tuple[Tuning, ...]:
            """The coupled qudits' tunings, in the coupling's order."""
            ...

        @property
        def source(self) -> str:
            """Where g_c comes from."""
            ...

    else:
        charge_coupling_hz, tunings, source = map(_Field, range(3))

    @property
    def unit(self) -> str:
        """The unit of an amplitude on either side: its tuning's."""
        tunings = self._values[1]
        return tunings[0].unit if tunings else ""

    def _problems(
        self, qudits: tuple[int, ...], known: Mapping[int, Any], where: str
    ) -> list[Problem]:
        g = self._values[0]
        problems = []
        if not (_finite(g) and g > 0):
            problems.append(
                Problem(
                    f"{g!r} is not a finite coupling above 0 Hz",
                    path=f"{where}.charge_coupling_hz",
                    value=g,
                )
            )
        tunings = self._values[1]
        if len(tunings) != len(qudits):
            problems.append(
                Problem(
                    f"gives {len(tunings)} tunings for a coupling of {len(qudits)} qudits",
                    path=f"{where}.tunings",
                    value=tunings,
                    fix="give the two qudits' tunings, in the order of the coupling's qudits",
                )
            )
            return problems
        for j, (q, tuning) in enumerate(zip(qudits, tunings, strict=True)):
            path = f"{where}.tunings[{j}]"
            if not isinstance(tuning, FluxTuning):
                problems.append(
                    Problem(
                        f"must be a FluxTuning, got {type(tuning).__name__}",
                        path=path,
                        value=tuning,
                    )
                )
                continue
            problems += tuning._problems(path)
            own = known[q].tuning
            if own != tuning:
                problems.append(
                    Problem(
                        f"is not the tuning of qudit {q}"
                        + (", which has none" if own is None else ""),
                        path=path,
                        value=tuning,
                        fix=f"give qudit {q} tuning= this tuning, or this the tuning of qudit {q}",
                    )
                )
        return problems

    def _point(self, a: int, b: int, amplitude_a: Any, amplitude_b: Any) -> CouplingPoint:
        import numpy as np

        tuning_a, tuning_b = self._values[1]
        at_a, at_b = np.broadcast_arrays(
            np.asarray(amplitude_a, dtype=float), np.asarray(amplitude_b, dtype=float)
        )
        energies_a, n_a = tuning_a._spectrum(at_a, 3)
        energies_b, n_b = tuning_b._spectrum(at_b, 3)
        g = float(self._values[0])
        scalar = at_a.ndim == 0

        def value(x: Any) -> Any:
            return float(x) if scalar else x

        return CouplingPoint(
            frequencies_hz={a: value(energies_a[..., 1]), b: value(energies_b[..., 1])},
            anharmonicities_hz={
                a: value(energies_a[..., 2] - 2.0 * energies_a[..., 1]),
                b: value(energies_b[..., 2] - 2.0 * energies_b[..., 1]),
            },
            exchange_hz=value(g * n_a[..., 0, 1] * n_b[..., 0, 1]),
            exchanges_hz={
                ((0, 1), (1, 0)): value(g * n_a[..., 0, 1] * n_b[..., 0, 1]),
                ((1, 1), (2, 0)): value(g * n_a[..., 1, 2] * n_b[..., 0, 1]),
                ((1, 1), (0, 2)): value(g * n_a[..., 0, 1] * n_b[..., 1, 2]),
            },
        )


class ExchangeControl(CouplingControl):
    """The pair's own control, such as a tunable coupler's flux or the
    voltage of a gate between two qudits: the 0-1 exchange J / 2 pi (Hz) of
    the coupled qudits at each of ``amplitudes`` (in ``unit``, offsets from
    the idle point, increasing, and spanning 0) is ``exchange_hz``, and
    linear between them. It moves no qudit's frequency. The exchange at 0 is
    the idle coupling, the coupling's ``strength_hz``. ``source`` says where
    the table comes from.
    """

    __slots__ = ()
    _fields = ("unit", "amplitudes", "exchange_hz", "source")

    def __init__(
        self,
        *,
        unit: str,
        amplitudes: Sequence[float],
        exchange_hz: Sequence[float],
        source: str = "",
    ) -> None:
        object.__setattr__(self, "_values", (unit, tuple(amplitudes), tuple(exchange_hz), source))

    if TYPE_CHECKING:

        @property
        def unit(self) -> str:
            """The unit of an amplitude, as ``"volt"`` or ``"flux_quantum"``."""
            ...

        @property
        def amplitudes(self) -> tuple[float, ...]:
            """The amplitudes the exchange is given at, increasing."""
            ...

        @property
        def exchange_hz(self) -> tuple[float, ...]:
            """The exchange J / 2 pi at each amplitude."""
            ...

        @property
        def source(self) -> str:
            """Where the table comes from."""
            ...

    else:
        unit, amplitudes, exchange_hz, source = map(_Field, range(4))

    def _problems(
        self, qudits: tuple[int, ...], known: Mapping[int, Any], where: str
    ) -> list[Problem]:
        unit, amplitudes, exchange, _ = self._values
        problems = []
        if not (isinstance(unit, str) and unit):
            problems.append(
                Problem(
                    f"{unit!r} is not the name of a unit",
                    path=f"{where}.unit",
                    value=unit,
                    fix='name the unit of an amplitude, as "volt"',
                )
            )
        if len(amplitudes) < 2 or len(amplitudes) != len(exchange):
            problems.append(
                Problem(
                    f"gives {len(amplitudes)} amplitudes and {len(exchange)} exchanges",
                    path=f"{where}.exchange_hz",
                    fix="give the exchange at each amplitude, at two amplitudes or more",
                )
            )
            return problems
        if not all(map(_finite, amplitudes)) or any(
            lo >= hi for lo, hi in zip(amplitudes, amplitudes[1:], strict=False)
        ):
            problems.append(
                Problem(
                    "must be finite and increasing",
                    path=f"{where}.amplitudes",
                    value=amplitudes,
                )
            )
        elif not amplitudes[0] <= 0.0 <= amplitudes[-1]:
            problems.append(
                Problem(
                    "does not span the idle point, amplitude 0",
                    path=f"{where}.amplitudes",
                    value=amplitudes,
                    fix="give the exchange at amplitudes on both sides of 0, or at 0",
                )
            )
        if not all(map(_finite, exchange)):
            problems.append(
                Problem("must be finite (Hz)", path=f"{where}.exchange_hz", value=exchange)
            )
        return problems

    def _point(self, a: int, b: int, amplitude: Any) -> CouplingPoint:
        import numpy as np

        _, amplitudes, exchange, _ = self._values
        at = np.asarray(amplitude, dtype=float)
        if at.size and (at.min() < amplitudes[0] or at.max() > amplitudes[-1]):
            raise InvalidValueError(
                f"the control of the coupling of qudits {a} and {b} is tabulated from "
                f"{amplitudes[0]} to {amplitudes[-1]} {self._values[0]}; {amplitude!r} is "
                "outside it"
            )
        j = np.interp(at, amplitudes, exchange)
        j = float(j) if at.ndim == 0 else j
        return CouplingPoint(
            frequencies_hz={},
            anharmonicities_hz={},
            exchange_hz=j,
            exchanges_hz={((0, 1), (1, 0)): j},
        )


@dataclass(frozen=True)
class CouplingPoint:
    """A coupling at an amplitude of its control, as ``Coupling.at`` reads
    it: each tuned qudit's 0-1 frequency and anharmonicity (Hz) by qudit
    index, the 0-1 exchange J / 2 pi (Hz), and the exchange of each pair of
    levels the control gives, keyed ``((i_a, i_b), (j_a, j_b))`` by the
    coupled qudits' levels in the coupling's order: ``((1, 1), (2, 0))`` is
    ``|11> <-> |20>``. Floats, or arrays the shape of the amplitude. An
    ``ExchangeControl`` moves no qudit, so its point has no frequencies."""

    frequencies_hz: Mapping[int, Any]
    anharmonicities_hz: Mapping[int, Any]
    exchange_hz: Any
    exchanges_hz: Mapping[tuple[tuple[int, int], tuple[int, int]], Any]


def _scalar(values: Any, given: Any) -> Any:
    """*values*, a float where the amplitude *given* is a number."""
    import numpy as np

    return float(values) if np.ndim(given) == 0 else values


def _side(side: Any) -> Any:
    """A side, as a qudit index: a ``Qudit`` handle names its carrier."""
    from .._device import Qudit

    return side.index if side.__class__ is Qudit else side


def _coupling_at(
    coupling: Coupling, control: CouplingControl, amplitude: Any, side: Any
) -> CouplingPoint:
    """``Coupling.at``: *coupling* with its *control* at *amplitude*."""
    a, b = coupling.qudits
    if isinstance(control, ChargeCoupling):
        if side is None:
            if not isinstance(amplitude, Mapping):
                raise InvalidValueError(
                    f"the coupling of qudits {a} and {b} is made by tuning both: give side={a} "
                    f"or side={b}, or each side's amplitude as {{{a}: ..., {b}: ...}}"
                )
            given = {_side(k): v for k, v in amplitude.items()}
            other = sorted(set(given) - {a, b}, key=repr)
            if other:
                raise InvalidValueError(
                    f"{other[0]!r} is not a side of the coupling of qudits {a} and {b}"
                )
            return control._point(a, b, given.get(a, 0.0), given.get(b, 0.0))
        side = _side(side)
        if side not in (a, b):
            raise InvalidValueError(
                f"qudit {side!r} is not a side of the coupling of qudits {a} and {b}; its sides "
                f"are {a} and {b}"
            )
        return control._point(
            a, b, amplitude if side == a else 0.0, amplitude if side == b else 0.0
        )
    if isinstance(control, ExchangeControl):
        if side is not None:
            raise InvalidValueError(
                f"the coupling of qudits {a} and {b} is turned on by its own control, which "
                "has no sides: give side=None"
            )
        return control._point(a, b, amplitude)
    raise InvalidValueError(
        f"{type(control).__name__} gives no operating point; Coupling.at reads a "
        "ChargeCoupling or an ExchangeControl"
    )
