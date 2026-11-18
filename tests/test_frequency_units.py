# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frequencies with units: ``5 * pulse.ghz`` wherever a frequency goes, the
same value bit for bit as the bare number of hertz ``5e9``."""

from __future__ import annotations

import json
import pickle
from fractions import Fraction

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, Frequency
from cudaq_pulse.classical import calibration
from cudaq_pulse.ops import (
    custom_samples,
    drive,
    get_drive_line,
    set_frequency,
    shift_frequency,
    square,
    wait,
)
from cudaq_pulse.quantum_systems import (
    Coupling,
    DispersiveReadout,
    DragPulse,
    FluxCZ,
    MicrowaveDrive,
    PhysicalQudit,
    ReadoutPulse,
    ResidualZZ,
    transmon_lattice,
)
from cudaq_pulse.quantum_systems.transmon_model import ChargeCoupledPair, TransmonCircuit
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property_settings import examples
from tests.test_units import _events

ghz, mhz, khz, hz = pulse.ghz, pulse.mhz, pulse.khz, pulse.hz
UNITS = ((hz, 0), (khz, 3), (mhz, 6), (ghz, 9))


# ── The arithmetic ───────────────────────────────────────────────────────────


def test_a_frequency_is_a_float_of_hertz():
    assert isinstance(5 * ghz, Frequency) and isinstance(5 * ghz, float)
    assert float(5 * ghz) == 5e9 and 5 * ghz == 5e9
    assert (ghz * 5).__class__ is Frequency
    assert (2.5 * mhz).exact_hz == 2_500_000
    assert [repr(f) for f in (5.1234 * ghz, 250 * mhz, 40 * khz, 12 * hz, -200 * mhz)] == [
        "5.1234 GHz",
        "250 MHz",
        "40 kHz",
        "12 Hz",
        "-200 MHz",
    ]


def test_a_bare_number_is_hertz():
    assert 5 * ghz + 2e6 == 5.002e9
    assert 2e6 + 5 * ghz == 5.002e9
    assert 5 * ghz - 1 * mhz == 4.999e9
    assert 1e10 - 5 * ghz == 5e9
    assert 5 * ghz > 4.9e9 and 5 * ghz == 5e9
    assert hash(5 * ghz) == hash(5e9)
    assert {5e9: "q0"}[5 * ghz] == "q0"


def test_arithmetic_keeps_the_kind_where_it_means_one():
    assert ((5 * ghz) * 2).__class__ is Frequency
    assert ((5 * ghz) / 2).__class__ is Frequency
    assert -(5 * mhz) == -5e6 and abs(-5 * mhz) == 5e6
    assert (10 * mhz) / (2 * mhz) == 5.0 and ((10 * mhz) / (2 * mhz)).__class__ is float
    # A time times a frequency is its number of cycles, exactly.
    assert (250 * pulse.ns) * (4 * mhz) == 1.0
    assert (4 * mhz) * (250 * pulse.ns) == 1.0
    assert ((1 * pulse.us) * (5 * ghz)).__class__ is float


@pytest.mark.parametrize(
    "make",
    [
        lambda: mhz * mhz,
        lambda: mhz + pulse.ns,
        lambda: pulse.ns + mhz,
        lambda: mhz - pulse.ns,
        lambda: mhz == pulse.ns,
        lambda: mhz < pulse.ns,
        lambda: pulse.ns / mhz,
        lambda: mhz / pulse.ns,
        lambda: 1 / mhz,
        lambda: mhz // 2,
        lambda: mhz % 2,
        lambda: mhz**2,
        lambda: Frequency(20 * pulse.ns),
    ],
)
def test_mixing_kinds_is_a_type_error(make):
    with pytest.raises(TypeError):
        make()


def test_a_frequency_compares_with_a_float_as_floats_do():
    # A third of a gigahertz has no exact float: it equals its float of hertz,
    # so it is neither above nor below it.
    third = ghz / 3
    hertz = float(third)
    assert third == hertz and third <= hertz and third >= hertz
    assert not third < hertz and not third > hertz and not third != hertz
    # With a frequency or an int it compares exactly.
    assert third != Frequency(hertz) and third > Frequency(hertz)
    assert 5 * ghz == 5_000_000_000 and 5 * ghz < 5_000_000_001


def test_a_time_still_compares_only_with_times():
    with pytest.raises(TypeError):
        assert 20 * pulse.ns == 20
    assert repr(20 * pulse.ns) == "20 ns"


def test_a_frequency_pickles_and_serializes_as_its_hertz():
    f = 5.1234 * ghz
    assert (
        pickle.loads(pickle.dumps(f)) == f and pickle.loads(pickle.dumps(f)).exact_hz == f.exact_hz
    )
    assert json.dumps(f) == json.dumps(5.1234e9)
    assert f"{f:.3e}" == "5.123e+09" and f"{f}" == "5.1234 GHz"


# ── Bit for bit with the bare number ─────────────────────────────────────────


@given(
    digits=st.integers(min_value=-(10**12), max_value=10**12),
    places=st.integers(min_value=0, max_value=9),
    unit=st.sampled_from(UNITS),
)
@settings(max_examples=examples(500))
def test_a_unit_scales_the_digits_written(digits, places, unit):
    scale, exponent = unit
    written = repr(digits / 10**places)
    # The literal with the unit's exponent, as Python reads it (correctly rounded).
    literal = float(Fraction(written) * 10**exponent)
    assert float(float(written) * scale) == literal
    assert (float(written) * scale).exact_hz == Fraction(written) * 10**exponent


def test_the_tutorial_frequencies_are_their_literals():
    # Plain float arithmetic gets some of these wrong by one ulp; units never do.
    for written in ("4.7", "5.1234", "6.9", "0.1", "5.15", "4.8", "5.102", "0.3", "1.15"):
        for scale, exponent in UNITS:
            assert float(float(written) * scale) == float(f"{written}e{exponent}"), (written, scale)


# ── Kernels ──────────────────────────────────────────────────────────────────

_QS = pulse.quantum_system(
    name="frequency-units",
    qudits={
        0: PhysicalQudit(0, 5.0e9, 0.0, 0.0, drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0))
    },
)


def _export(kernel, *args, qs=_QS):
    lowered = pulse.compile(kernel, quantum_system=qs).lower(qs.qudits[0], *args)
    return lowered, json.dumps(lowered.export_to_controller(), sort_keys=True)


@pulse.kernel
def _bare(q):
    d = get_drive_line(q)
    set_frequency(d.tone, 5.1e9)
    drive(d, square(8, 0.1))
    shift_frequency(d.tone, 2e6)
    drive(d, square(8, 0.1))


@pulse.kernel
def _with_units(q):
    d = get_drive_line(q)
    set_frequency(d.tone, 5.1 * pulse.ghz)
    drive(d, square(8, 0.1))
    shift_frequency(d.tone, 2 * pulse.mhz)
    drive(d, square(8, 0.1))


def test_a_kernel_with_units_compiles_to_the_same_program():
    bare, bare_export = _export(_bare)
    units, units_export = _export(_with_units)
    assert units_export == bare_export
    assert units.mlir().replace("_with_units", "K") == bare.mlir().replace("_bare", "K")
    assert _events(units) == _events(bare)


@pulse.kernel
def _shift_by(q, delta: float):
    d = get_drive_line(q)
    shift_frequency(d.tone, delta * pulse.mhz)
    drive(d, square(8, 0.1))


@pulse.kernel
def _shift_by_bare(q, delta: float):
    d = get_drive_line(q)
    shift_frequency(d.tone, delta * 1e6)
    drive(d, square(8, 0.1))


def test_a_float_argument_times_a_unit_is_the_same_expression():
    assert _export(_shift_by, 2.5)[1] == _export(_shift_by_bare, 2.5)[1]
    generic = pulse.compile(_shift_by, quantum_system=_QS).lower(_QS.qudits[0], 2.5).generic
    bare = pulse.compile(_shift_by_bare, quantum_system=_QS).lower(_QS.qudits[0], 2.5).generic
    assert generic.mlir().replace("_shift_by", "K") == bare.mlir().replace("_shift_by_bare", "K")


@pulse.kernel
def _time_as_frequency(q):
    d = get_drive_line(q)
    shift_frequency(d.tone, 20 * pulse.ns)


@pulse.kernel
def _parameter_time_as_frequency(q, tau: float):
    d = get_drive_line(q)
    set_frequency(d.tone, tau * pulse.ns)
    drive(d, square(8, 0.1))


@pulse.kernel
def _frequency_as_wait(q):
    d = get_drive_line(q)
    wait(d, 5 * pulse.mhz)


@pulse.kernel
def _frequency_as_duration(q):
    d = get_drive_line(q)
    drive(d, square(5 * pulse.mhz, 0.1))


@pulse.kernel
def _parameter_frequency_as_wait(q, f: float):
    d = get_drive_line(q)
    wait(d, f * pulse.mhz)


@pytest.mark.parametrize(
    ("kernel", "args", "says"),
    [
        (_time_as_frequency, (), "shift_frequency(): the frequency 20 ns is a time"),
        (
            _parameter_time_as_frequency,
            (3.0,),
            "set_frequency(): the frequency tau * 1 ns is a time",
        ),
        (_frequency_as_wait, (), "wait(): duration 5 MHz is a frequency"),
        (_frequency_as_duration, (), "square(): duration 5 MHz is a frequency"),
        (_parameter_frequency_as_wait, (3.0,), "is a frequency"),
    ],
)
def test_a_kernel_refuses_the_wrong_kind_and_shows_the_fix(kernel, args, says):
    with pytest.raises(TypeError) as caught:
        _export(kernel, *args)
    assert says in str(caught.value)
    assert "pulse.ghz" in str(caught.value) or "pulse.ns" in str(caught.value)


@pulse.kernel
def _samples_at(q, rate):
    d = get_drive_line(q)
    drive(d, custom_samples([0.1] * 8, sample_rate_ghz=rate))


def test_custom_samples_takes_its_rate_with_units():
    assert _export(_samples_at, 2 * pulse.ghz)[1] == _export(_samples_at, 2.0)[1]


def test_a_shift_with_units_does_not_warn_about_units():
    import warnings

    @pulse.kernel
    def small_shift(q):
        d = get_drive_line(q)
        shift_frequency(d.tone, 2 * pulse.mhz)
        drive(d, square(8 * pulse.ns, 0.1))

    with warnings.catch_warnings():
        warnings.simplefilter("error", pulse.UnitWarning)
        pulse.compile(small_shift, quantum_system=_QS, unit_warnings=True).lower(_QS.qudits[0])


# ── The host ─────────────────────────────────────────────────────────────────


def test_a_quantum_system_with_units_is_the_same_quantum_system():
    readout = DispersiveReadout(frequency_hz=7.2e9, dispersive_shift_hz=-1e6, linewidth_hz=2e6)
    with_units = DispersiveReadout(
        frequency_hz=7.2 * ghz, dispersive_shift_hz=-1 * mhz, linewidth_hz=2 * mhz
    )
    assert with_units == readout and pickle.dumps(with_units) == pickle.dumps(readout)
    bare = PhysicalQudit(0, 5.1e9, 30.0, 20.0, anharmonicity_hz=-2.2e8, readout=readout)
    units = PhysicalQudit(0, 5.1 * ghz, 30.0, 20.0, anharmonicity_hz=-220 * mhz, readout=with_units)
    assert units == bare and repr(units) == repr(bare)
    assert pickle.dumps(units) == pickle.dumps(bare)
    assert units.frequency_hz.__class__ is float and units.anharmonicity_hz.__class__ is float
    assert Coupling((0, 1), strength_hz=3 * mhz).strength_hz.__class__ is float
    assert ResidualZZ((0, 1), rate_hz=-20 * khz).rate_hz.__class__ is float
    cz = FluxCZ(duration_ns=98.0, interaction_frequency_hz=5.9 * ghz)
    assert cz.interaction_frequency_hz.__class__ is float
    systems = [
        pulse.quantum_system(name="p", qudits={0: q}, clock_ghz=c)
        for q, c in ((bare, 2.0), (units, 2 * ghz))
    ]
    assert pickle.dumps(systems[0]) == pickle.dumps(systems[1])
    assert systems[1].clock_ghz.__class__ is float


def test_a_clock_takes_a_bare_number_in_ghz_or_any_unit():
    plain = Controller(clock_ghz=0.5, sample_rate_ghz=2.0)
    assert Controller(clock_ghz=500 * mhz, sample_rate_ghz=2 * ghz) == plain
    assert Controller(clock_ghz=500 * mhz).clock_ghz.__class__ is float
    system = ControlSystem(controllers=[plain], reference_clock_ghz=250 * mhz)
    assert system.reference_clock_ghz == 0.25
    assert pulse.Processor(clock_ghz=1 * ghz).clock_ghz == 1.0


def test_an_export_takes_its_sample_rate_with_units():
    lowered, _ = _export(_bare)
    assert lowered.export_to_controller(2 * ghz) == lowered.export_to_controller(2.0)


@pytest.mark.parametrize(
    "make",
    [
        lambda: PhysicalQudit(0, 5 * pulse.ns, 30.0, 20.0),
        lambda: Coupling((0, 1), strength_hz=3 * pulse.ns),
        lambda: Controller(clock_ghz=2 * pulse.ns),
        lambda: ControlSystem(reference_clock_ghz=1 * pulse.ns),
        lambda: pulse.quantum_system(
            name="p", qudits={0: PhysicalQudit(0, 5e9, 1, 1)}, clock_ghz=pulse.ns
        ),
        lambda: calibration.ramsey_detuning(1 * pulse.us, 0.0, 100.0),
        lambda: Controller(max_fine_delay_ns=5 * mhz),
        lambda: calibration.ramsey_probability(5 * mhz, 0.0, 1e6),
        lambda: calibration.t1_update(1.0, 1.0, True, 5 * mhz),
        lambda: pulse.Link(fixed_ns=5 * mhz),
        lambda: pulse.Link(feedback_ns=5 * mhz),
        lambda: pulse.Time(5 * mhz),
    ],
)
def test_the_host_refuses_the_wrong_kind(make):
    with pytest.raises(TypeError, match=r"pulse\.(ghz|ns)"):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda: DragPulse(duration_ns=5 * mhz, sigma_ns=10.0),
        lambda: ReadoutPulse(duration_ns=300.0, integration_ns=1 * mhz),
        lambda: FluxCZ(duration_ns=98.0, buffer_ns=1 * khz),
        lambda: DispersiveReadout(depletion_ns=1 * mhz),
        lambda: PhysicalQudit(0, 5e9, 50 * khz, 20.0),
        lambda: PhysicalQudit(0, 5e9, 50.0, 20.0, t2_echo_us=1 * khz),
    ],
)
def test_a_time_as_a_bare_number_refuses_a_frequency(make):
    # Times in a quantum system's components are bare numbers of ns or us.
    with pytest.raises(TypeError, match=r"is a time, a number of (ns|us), and .* is a frequency"):
        make()


def test_the_calibration_helpers_take_units_and_return_hertz():
    bare = calibration.frequency_update(1e5, 1e6, 1, 200.0, 0.3)
    units = calibration.frequency_update(100 * khz, 1 * mhz, 1, 200 * pulse.ns, 0.3)
    assert units == bare and all(v.__class__ is float for v in units)
    assert calibration.ramsey_setting(100 * khz, 1 * mhz) == calibration.ramsey_setting(1e5, 1e6)
    assert calibration.ramsey_detuning(100 * khz, 3.0, 200.0).__class__ is float
    assert calibration.binary_search_wait(1 * mhz) == calibration.binary_search_wait(1e6)
    assert calibration.binary_search_phase(100 * khz, 200.0) == calibration.binary_search_phase(
        1e5, 200.0
    )
    assert calibration.binary_search_update(1e5, 1e6, 1, 200.0) == calibration.binary_search_update(
        100 * khz, 1 * mhz, 1, 200.0
    )
    assert calibration.ramsey_probability(200.0, 0.3, 1 * mhz) == calibration.ramsey_probability(
        200.0, 0.3, 1e6
    )


def test_the_transmon_models_take_units():
    bare = transmon_lattice(3, base_frequency_hz=5.0e9, anharmonicity_hz=-3.3e8)
    units = transmon_lattice(3, base_frequency_hz=5 * ghz, anharmonicity_hz=-330 * mhz)
    assert units.physical_qudits == bare.physical_qudits
    circuit = TransmonCircuit(ec_hz=250 * mhz, ej_max_hz=20 * ghz, asymmetry=0.3)
    assert circuit == TransmonCircuit(ec_hz=2.5e8, ej_max_hz=2e10, asymmetry=0.3)
    assert circuit.flux_for_f01(5 * ghz) == circuit.flux_for_f01(5e9)
    pair = ChargeCoupledPair(circuit, circuit, 20 * mhz)
    assert pair.g_c_hz.__class__ is float


def test_a_calibration_takes_its_frequencies_with_units():
    from cudaq_pulse.quantum_systems.calibration import Calibration

    cal = Calibration.load("transmon_surface17")
    assert cal.flux_for_detuning("X2", -738 * mhz) == cal.flux_for_detuning("X2", -738e6)
    assert cal.exchange_hz("X2-D4", data_hz=4.4 * ghz) == cal.exchange_hz("X2-D4", data_hz=4.4e9)
    assert cal.exchange_hz("X2-D4", auxiliary_hz=5.5 * ghz) == cal.exchange_hz(
        "X2-D4", auxiliary_hz=5.5e9
    )
    with pytest.raises(TypeError, match=r"pulse\.ghz"):
        cal.flux_for_detuning("X2", 5 * pulse.ns)
