# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantum systems: their qudits, couplings, decoherence and calibrations."""

from typing import TYPE_CHECKING

from ._registry import UnknownQuantumSystemError, available_quantum_systems, register_quantum_system
from .base import (
    Coupling,
    CouplingControl,
    DispersiveReadout,
    DragPulse,
    Drive,
    FluxCZ,
    MicrowaveDrive,
    PhysicalQudit,
    Readout,
    ReadoutPulse,
    ResidualZZ,
    Tuning,
    VirtualZ,
)
from .calibration import Calibration
from .models import SimulationModel, TransmonModel
from .transmon import transmon_lattice, transmon_surface17

if TYPE_CHECKING:
    from .tuning import ChargeCoupling, CouplingPoint, ExchangeControl, FluxTuning

__all__ = [
    "PhysicalQudit",
    "Readout",
    "DispersiveReadout",
    "Drive",
    "MicrowaveDrive",
    "Coupling",
    "ResidualZZ",
    "Tuning",
    "FluxTuning",
    "CouplingControl",
    "ChargeCoupling",
    "ExchangeControl",
    "CouplingPoint",
    "DragPulse",
    "VirtualZ",
    "ReadoutPulse",
    "FluxCZ",
    "SimulationModel",
    "TransmonModel",
    "Calibration",
    "transmon_surface17",
    "transmon_lattice",
    "register_quantum_system",
    "available_quantum_systems",
    "UnknownQuantumSystemError",
]


# The components of tuning lines and coupling controls load with the first
# quantum system that declares one, or when named here.
_TUNING = frozenset({"FluxTuning", "ChargeCoupling", "ExchangeControl", "CouplingPoint"})


def __getattr__(name: str) -> object:
    if name in _TUNING:
        from . import tuning

        return getattr(tuning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
