# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qudit tunings and coupling controls: the components a quantum system
declares for its coupling lines, their read-only maps, and that a system
without them is what it was."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import (
    ChargeCoupling,
    Coupling,
    CouplingControl,
    ExchangeControl,
    FluxTuning,
    PhysicalQudit,
    Tuning,
    transmon_surface17,
)
from cudaq_pulse.quantum_systems.calibration import shipped_calibration
from cudaq_pulse.quantum_systems.transmon_model import TransmonCircuit

D4, X2 = 3, 10


@pytest.fixture(scope="module")
def qs():
    return pulse.quantum_system("transmon_surface17")


def _pair(control=None, tunings=(None, None)):
    return pulse.quantum_system(
        name="pair",
        qudits={
            i: PhysicalQudit(i, f, 50.0, 30.0, anharmonicity_hz=-2.0e8, tuning=t)
            for i, (f, t) in enumerate(zip([4.8e9, 6.9e9], tunings, strict=True))
        },
        couplings=[Coupling((0, 1), strength_hz=5.7e6, control=control)],
    )


def test_every_shipped_qudit_is_tuned_by_its_calibrated_circuit(qs):
    cal = shipped_calibration("transmon_surface17")
    for q in qs.physical_qudits.values():
        tuning = q.tuning
        assert isinstance(tuning, FluxTuning) and isinstance(tuning, Tuning)
        assert tuning.unit == "flux_quantum"
        assert tuning.source.startswith(f"transmon_surface17 calibration, qudits.{q.label}:")
        assert cal.value("qudits", q.label, "ec") == tuning.ec_hz
        assert tuning.idle_flux == cal.idle_flux(q.label)
        # The circuit is fitted to the published idle frequency.
        assert q.frequency_at(0.0) == pytest.approx(q.frequency_hz, abs=1e3)
        assert q.anharmonicity_at(0.0) == pytest.approx(q.anharmonicity_hz, abs=1e3)


def test_frequency_at_is_the_circuit_and_the_calibration_conversion(qs):
    cal = shipped_calibration("transmon_surface17")
    d4 = qs.physical_qudits[D4]
    circuit = TransmonCircuit(d4.tuning.ec_hz, d4.tuning.ej_max_hz, d4.tuning.asymmetry)
    for amplitude in (-0.3, -0.1, 0.0):
        flux = 0.5 + amplitude
        assert d4.frequency_at(amplitude) == pytest.approx(circuit.f01_hz(flux), rel=1e-12)
        assert d4.anharmonicity_at(amplitude) == pytest.approx(
            circuit.anharmonicity_hz(flux), rel=1e-9
        )
        assert d4.frequency_at(amplitude) - d4.frequency_at(0.0) == pytest.approx(
            cal.detuning_for_flux("D4", flux), rel=1e-9
        )


def test_amplitudes_are_signed_and_even_about_the_sweet_spot(qs):
    d4 = qs.physical_qudits[D4]
    amplitudes = np.linspace(-0.3, 0.3, 13)
    frequencies = d4.frequency_at(amplitudes)
    assert frequencies.shape == (13,)
    np.testing.assert_allclose(frequencies, frequencies[::-1], rtol=1e-12)
    # A data qudit idles at its lower sweet spot: any flux tunes it up.
    assert np.all(frequencies >= d4.frequency_at(0.0) - 1.0)
    # amplitude_for inverts on the branch of the idle point.
    found = d4.amplitude_for(frequencies[:7])
    np.testing.assert_allclose(found, amplitudes[:7], atol=1e-7)
    assert d4.amplitude_for(5.276e9) == pytest.approx(-0.30148, abs=1e-4)
    with pytest.raises(pulse.InvalidValueError, match="outside the tuning range"):
        d4.amplitude_for(9e9)


def test_the_coupling_at_its_idle_point_is_its_strength(qs):
    for c in qs.couplings:
        assert isinstance(c.control, ChargeCoupling) and isinstance(c.control, CouplingControl)
        assert c.control.tunes_qudits and c.control.unit == "flux_quantum"
        assert c.control.tunings == tuple(qs.physical_qudits[q].tuning for q in c.qudits)
        point = c.at(0.0, side=c.qudits[0])
        assert point.exchange_hz == pytest.approx(c.strength_hz, rel=1e-9)
        assert point.frequencies_hz == pytest.approx(
            {q: qs.physical_qudits[q].frequency_hz for q in c.qudits}, abs=1e3
        )


def test_the_cz_operating_point_of_x2_d4(qs):
    # Both qudits tuned: the data qudit to the interaction frequency, the
    # auxiliary's 1-2 transition there, so |11> meets |20>.
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    d4, x2 = qs.physical_qudits[D4], qs.physical_qudits[X2]
    interaction = c.get_calibration(pulse.gates.CZ).interaction_frequency_hz
    amplitudes = np.linspace(0.0, 0.3, 3001)
    f12 = x2.frequency_at(amplitudes) + x2.anharmonicity_at(amplitudes)
    at_aux = float(np.interp(interaction, f12[::-1], amplitudes[::-1]))
    point = c.at({qs.qudits[X2]: at_aux, D4: d4.amplitude_for(interaction)})
    assert point.frequencies_hz[D4] == pytest.approx(interaction, abs=1e3)
    assert point.frequencies_hz[X2] + point.anharmonicities_hz[X2] == pytest.approx(
        interaction, abs=2e5
    )
    assert point.exchange_hz == pytest.approx(6.38e6, abs=0.01e6)
    assert point.exchanges_hz[(1, 1), (2, 0)] == pytest.approx(8.876e6, abs=0.01e6)
    # The same from the calibration's charge-coupled pair (data first there).
    pair = shipped_calibration("transmon_surface17").pair("X2-D4", 3)
    n_d = pair.a.charge_matrix(0.5 + d4.amplitude_for(interaction), 3)
    n_a = pair.b.charge_matrix(at_aux, 3)
    assert point.exchanges_hz[(1, 1), (2, 0)] == pytest.approx(
        pair.g_c_hz * n_a[1, 2] * n_d[0, 1], rel=1e-9
    )
    assert point.exchanges_hz[(1, 1), (0, 2)] == pytest.approx(
        pair.g_c_hz * n_a[0, 1] * n_d[1, 2], rel=1e-9
    )


def test_one_side_moves_alone_and_arrays_broadcast(qs):
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    amplitudes = np.array([-0.2, 0.0, 0.2])
    point = c.at(amplitudes, side=qs.qudits[D4])
    assert point.frequencies_hz[D4].shape == (3,)
    np.testing.assert_allclose(
        point.frequencies_hz[X2], qs.physical_qudits[X2].frequency_at(0.0), rtol=1e-12
    )
    np.testing.assert_allclose(
        point.frequencies_hz[D4], qs.physical_qudits[D4].frequency_at(amplitudes), rtol=1e-12
    )
    assert point.exchange_hz[0] == pytest.approx(point.exchange_hz[2], rel=1e-12)


def test_coupling_at_names_what_is_wrong(qs):
    c = qs.coupling(qs.qudits[X2], qs.qudits[D4])
    with pytest.raises(pulse.InvalidValueError, match=r"made by tuning both: give side=10"):
        c.at(0.1)
    with pytest.raises(pulse.InvalidValueError, match="qudit 4 is not a side"):
        c.at(0.1, side=4)
    with pytest.raises(pulse.InvalidValueError, match="4 is not a side"):
        c.at({4: 0.1})
    with pytest.raises(pulse.NotFoundError, match="has no control: it is always on"):
        pulse.quantum_system("transmon_lattice:2").couplings[0].at(0.0)
    with pytest.raises(pulse.NotFoundError, match=r"qudit 0 \(Q0\) has no tuning"):
        pulse.quantum_system("transmon_lattice:2").physical_qudits[0].frequency_at(0.1)


def test_an_exchange_control_is_the_pairs_own():
    control = ExchangeControl(
        unit="volt", amplitudes=(-0.1, 0.0, 0.2), exchange_hz=(0.0, 5.7e6, 25.7e6)
    )
    pair = _pair(control)
    c = pair.couplings[0]
    assert c.at(0.0).exchange_hz == pytest.approx(5.7e6)
    assert c.at(0.1).exchange_hz == pytest.approx(15.7e6)
    np.testing.assert_allclose(c.at(np.array([-0.05, 0.2])).exchange_hz, [2.85e6, 25.7e6])
    assert c.at(0.1).frequencies_hz == {}
    with pytest.raises(pulse.InvalidValueError, match="has no sides"):
        c.at(0.1, side=0)
    with pytest.raises(pulse.InvalidValueError, match="tabulated from -0.1 to 0.2 volt"):
        c.at(0.3)


@pytest.mark.parametrize(
    ("control", "path", "match"),
    [
        (
            ExchangeControl(unit="volt", amplitudes=(0.0, 0.0), exchange_hz=(1.0, 2.0)),
            "couplings[0].control.amplitudes",
            "finite and increasing",
        ),
        (
            ExchangeControl(unit="volt", amplitudes=(0.1, 0.2), exchange_hz=(1.0, 2.0)),
            "couplings[0].control.amplitudes",
            "does not span the idle point",
        ),
        (
            ExchangeControl(unit="", amplitudes=(0.0, 0.2), exchange_hz=(1.0,)),
            "couplings[0].control.unit",
            "not the name of a unit",
        ),
        ("flux", "couplings[0].control", "must be a CouplingControl"),
    ],
)
def test_a_control_with_problems_is_refused_at_its_path(control, path, match):
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(control)
    assert any(p.path == path and match in p.message for p in raised.value.problems), str(
        raised.value
    )


def test_a_charge_coupling_must_tune_by_its_qudits_own_tunings(qs):
    d4, x2 = qs.physical_qudits[D4].tuning, qs.physical_qudits[X2].tuning
    control = ChargeCoupling(charge_coupling_hz=3.1e6, tunings=(d4, x2))
    assert _pair(control, (d4, x2)).couplings[0].control is control
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(control, (d4, None))
    (problem,) = raised.value.problems
    assert problem.path == "couplings[0].control.tunings[1]"
    assert "is not the tuning of qudit 1, which has none" in problem.message
    bad = ChargeCoupling(charge_coupling_hz=-1.0, tunings=(d4.replace(ec_hz=0.0), x2))
    with pytest.raises(pulse.QuantumSystemDefinitionError) as raised:
        _pair(bad, (d4.replace(ec_hz=0.0), x2))
    paths = sorted(p.path for p in raised.value.problems)
    assert paths == [
        "couplings[0].control.charge_coupling_hz",
        "couplings[0].control.tunings[0].ec_hz",
    ]


def test_the_published_system_and_untuned_systems_are_unchanged():
    published = transmon_surface17(calibration=None)
    assert all(q.tuning is None for q in published.physical_qudits.values())
    assert all(c.control is None for c in published.couplings)
    # Fingerprints of systems without tunings, as before tunings existed.
    assert pulse.quantum_system("transmon_lattice:3").fingerprint == (
        "9814de8af8ac7d098baf74228631cfd4"
    )
    assert published.fingerprint == "acdf23ba20dd32bca283e0e63696d9af"
    assert _pair().fingerprint == "0cc289b040d470cdc776a0bc3d7526ca"
    q = _pair().physical_qudits[0]
    assert "tuning" not in repr(q) and "control" not in repr(_pair().couplings[0])
    data = _pair().as_dict()
    assert "tuning" not in data["qudits"][0] and "control" not in data["couplings"][0]


def test_tunings_change_the_fingerprint():
    tuning = FluxTuning(ec_hz=1.65e8, ej_max_hz=2.6e10, asymmetry=0.47, idle_flux=0.5)
    tuned = _pair(None, (tuning, None))
    assert tuned.fingerprint != _pair().fingerprint
    assert tuned.fingerprint != _pair(None, (tuning.replace(idle_flux=0.0), None)).fingerprint


def test_tunings_and_controls_pickle_and_round_trip_through_a_system_file(qs):
    variant = qs.replace(layout=[3, 10])
    data = json.loads(json.dumps(variant.replace(qudits=dict(qs.physical_qudits)).as_dict()))
    assert data["qudits"][D4]["tuning"]["kind"] == "flux"
    control = data["couplings"][0]["control"]
    assert control["kind"] == "charge" and "tunings" not in control
    loaded = pulse.QuantumSystem.from_dict(data)
    assert loaded.physical_qudits == qs.physical_qudits
    assert tuple(loaded.couplings) == tuple(qs.couplings)
    exchange = ExchangeControl(unit="volt", amplitudes=(0.0, 1.0), exchange_hz=(1e6, 2e6))
    data = json.loads(json.dumps(_pair(exchange).as_dict()))
    assert data["couplings"][0]["control"] == {
        "kind": "exchange",
        "unit": "volt",
        "amplitudes": [0.0, 1.0],
        "exchange_hz": [1e6, 2e6],
        "source": "",
    }
    assert pulse.QuantumSystem.from_dict(data).couplings[0].control == exchange
    for value in (qs.physical_qudits[D4], qs.couplings[0], exchange):
        assert pickle.loads(pickle.dumps(value)) == value


def test_the_tuning_module_loads_only_with_a_system_that_declares_one():
    code = (
        "import sys, cudaq_pulse as pulse\n"
        "qs = pulse.quantum_system('transmon_lattice:4')\n"
        "@pulse.kernel\n"
        "def k(q):\n"
        "    drive(get_drive_line(q), square(20, 0.1))\n"
        "k.lower(qs.qudits[0]).mlir()\n"
        "print('cudaq_pulse.quantum_systems.tuning' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False"]
