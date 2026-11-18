# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rules docs/user_guide/conventions.rst states: units, time bases and
signs, one test each, which the page names; and the page's snippets, which
run and print what their output blocks or comments say, as a tutorial page's do."""

from __future__ import annotations

import contextlib
import io
import math
import re
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.quantum_systems import MicrowaveDrive, PhysicalQudit

from tests.test_tutorials import _matches, code_blocks, expected_output

_PAGE = Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "conventions.rst"
F = 5.0e9
# 2 ns at pi/4 rad/ns: a quarter turn, short enough that a 1 MHz detuning
# barely acts during it.
A90 = math.pi / 4


def _system(**qudit):
    """One qudit at 5 GHz, without decay, whose amplitude is its Rabi rate in rad/ns."""
    drive = MicrowaveDrive(amplitude_scale_rad_per_ns=1.0)
    return pulse.quantum_system(
        name="conventions", qudits={0: PhysicalQudit(0, F, 0.0, 0.0, drive=drive, **qudit)}
    )


QS = _system()


def _rho(kernel, *args, qs=QS, levels=2):
    result = pulse.simulate(
        pulse.compile(kernel, quantum_system=qs), qs.qudits[0], *args, levels=levels, dt_ns=0.01
    )
    return np.asarray(result.density_matrix)


def _bloch(rho):
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real])


def _export(kernel, *args, qs=QS):
    lowered = pulse.compile(kernel, quantum_system=qs).lower(qs.qudits[0], *args)
    return lowered.export_to_controller()


def _drives(program):
    return [e for e in program["events"] if e["kind"] == "drive"]


@pytest.mark.skipif(not _PAGE.is_file(), reason="needs the source tree's docs/")
def test_the_page_names_every_test_here_and_only_those():
    named = re.findall(r"^\.\. test: (\w+)$", _PAGE.read_text(), flags=re.M)
    here = sorted(n for n in globals() if n.startswith("test_") and n not in _OF_THE_PAGE)
    assert sorted(named) == here
    assert len(named) == len(set(named))


@pytest.mark.skipif(not _PAGE.is_file(), reason="needs the source tree's docs/")
def test_the_page_runs_and_prints_its_comments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "conventions_page"}
    blocks = code_blocks(_PAGE)
    assert len(blocks) > 10
    checked = 0
    for number, (_, code, shown) in enumerate(blocks, 1):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, f"{_PAGE.name}, block {number}", "exec"), namespace)
        printed = [line.rstrip() for line in out.getvalue().splitlines()]
        if shown is not None:
            assert printed == shown, (
                f"block {number}: printed {printed!r}, its output block shows {shown!r}"
            )
            checked += len(shown)
            continue
        for k, want in enumerate(expected_output(code)):
            got = printed[k] if k < len(printed) else "<nothing printed>"
            assert _matches(want, got), f"block {number}: printed {got!r}, says {want!r}"
            checked += 1
    assert checked > 10


# The tests of the page itself, which it does not name.
_OF_THE_PAGE = {
    "test_the_page_names_every_test_here_and_only_those",
    "test_the_page_runs_and_prints_its_comments",
}


# ── Units ───────────────────────────────────────────────────────────────────


@pulse.kernel
def _in_cycles(q):
    d = get_drive_line(q)
    drive(d, square(20, 0.1))
    drive(d, gaussian(40, 0.1, 10))
    wait(d, 8)


@pulse.kernel
def _in_ns(q):
    d = get_drive_line(q)
    drive(d, square(10 * pulse.ns, 0.1))
    drive(d, gaussian(20 * pulse.ns, 0.1, 5 * pulse.ns))
    wait(d, 4 * pulse.ns)


def test_a_bare_duration_counts_clock_cycles():
    assert QS.clock_ghz == 2.0
    cycles, ns = _export(_in_cycles), _export(_in_ns)
    assert [e["duration_cycles"] for e in _drives(cycles)] == [20, 40]
    assert cycles["events"] == ns["events"]
    slow = _system().replace(clock_ghz=1.0)
    assert [e["duration_cycles"] for e in _drives(_export(_in_ns, qs=slow))] == [10, 20]
    assert [e["duration_cycles"] for e in _drives(_export(_in_cycles, qs=slow))] == [20, 40]


def test_frequencies_are_in_hz():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        set_frequency(d.tone, 5.1e9)
        drive(d, square(8, 0.1))
        shift_frequency(d.tone, 2e6)
        drive(d, square(8, 0.1))

    assert [e["frequency_hz"] for e in _drives(_export(k))] == [5.1e9, 5.102e9]

    @pulse.kernel
    def with_units(q):
        d = get_drive_line(q)
        set_frequency(d.tone, 5.1 * pulse.ghz)
        drive(d, square(8, 0.1))
        shift_frequency(d.tone, 2e6)
        drive(d, square(8, 0.1))

    assert _export(with_units)["events"] == _export(k)["events"]

    # A transmon's 1-2 transition is at frequency_hz + anharmonicity_hz (< 0),
    # and turns sqrt(2) faster.
    qs = _system(anharmonicity_hz=-200e6)

    @pulse.kernel
    def to_level_2(q):
        d = get_drive_line(q)
        drive(d, square(100 * pulse.ns, math.pi / 100))
        set_frequency(d.tone, 5.0e9 - 200e6)
        drive(d, square(100 * pulse.ns, math.pi / 100 / math.sqrt(2)))

    rho = _rho(to_level_2, qs=qs, levels=3)
    assert rho[2, 2].real == pytest.approx(1.0, abs=5e-3)


def test_amplitude_times_duration_is_the_rotation_angle():
    @pulse.kernel
    def k(q, amplitude: float):
        drive(get_drive_line(q), square(20 * pulse.ns, amplitude))

    assert _rho(k, math.pi / 20)[1, 1].real == pytest.approx(1.0, abs=1e-6)
    assert _rho(k, math.pi / 40)[1, 1].real == pytest.approx(0.5, abs=1e-6)


# ── Time bases ──────────────────────────────────────────────────────────────


def test_the_program_clock_is_2_ghz_unless_the_system_says_otherwise():
    assert QS.clock_ghz == 2.0
    assert pulse.quantum_system("transmon_surface17").clock_ghz == 2.0
    assert pulse.quantum_system("transmon_surface17", clock_ghz=1.0).clock_ghz == 1.0
    assert _export(_in_ns)["clock_ghz"] == 2.0


def test_the_schedule_and_the_export_count_cycles_from_the_program_start():
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        drive(d, square(10 * pulse.ns, 0.1))
        wait(d, 5 * pulse.ns)
        drive(d, square(10 * pulse.ns, 0.1))

    lowered = pulse.compile(k, quantum_system=QS).lower(QS.qudits[0])
    played = _drives(lowered.export_to_controller())
    assert [(e["start_cycles"], e["duration_cycles"], e["end_cycles"]) for e in played] == [
        (0, 20, 20),
        (30, 20, 50),
    ]
    assert sorted(e.start_cycles for e in lowered.schedule if e.duration_cycles == 20) == [0, 30]


# ── Frames and signs ────────────────────────────────────────────────────────


@pulse.kernel
def _x90(q):
    drive(get_drive_line(q), square(2 * pulse.ns, A90))


@pulse.kernel
def _imaginary_x90(q):
    drive(get_drive_line(q), square(2 * pulse.ns, A90 * 1j))


@pulse.kernel
def _phase_then_x90(q, phi: float):
    d = get_drive_line(q)
    shift_phase(d.tone, phi)
    drive(d, square(2 * pulse.ns, A90))


def test_a_real_amplitude_turns_about_x_and_an_imaginary_one_about_y():
    assert _bloch(_rho(_x90)) == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)
    assert _bloch(_rho(_imaginary_x90)) == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


@pulse.kernel
def _x90_phase_pulse(q, phi: float):
    d = get_drive_line(q)
    drive(d, square(2 * pulse.ns, A90))
    shift_phase(d.tone, phi)
    drive(d, square(3 * pulse.ns, 0.3))


@pulse.kernel
def _x90_turned_pulse(q, re, im):
    d = get_drive_line(q)
    drive(d, square(2 * pulse.ns, A90))
    drive(d, square(3 * pulse.ns, [re, im]))


def test_shift_phase_is_a_virtual_z_that_turns_later_pulses_by_minus_phi():
    assert _bloch(_rho(_phase_then_x90, math.pi / 2)) == pytest.approx([-1.0, 0, 0], abs=1e-6)
    phi = 0.7
    shifted = _rho(_x90_phase_pulse, phi)
    turned = 0.3 * np.exp(-1j * phi)
    assert shifted == pytest.approx(_rho(_x90_turned_pulse, turned.real, turned.imag), abs=1e-9)
    other = 0.3 * np.exp(1j * phi)
    assert np.abs(shifted - _rho(_x90_turned_pulse, other.real, other.imag)).max() > 0.1


def test_the_simulated_state_is_in_the_qudit_frame():
    @pulse.kernel
    def x90_then_phase(q):
        d = get_drive_line(q)
        drive(d, square(2 * pulse.ns, A90))
        shift_phase(d.tone, math.pi / 2)

    assert _rho(x90_then_phase) == pytest.approx(_rho(_x90), abs=1e-12)

    @pulse.kernel
    def x90_detuned_wait(q):
        d = get_drive_line(q)
        set_frequency(d.tone, F - 1e6)
        drive(d, square(2 * pulse.ns, A90))
        wait(d, 248 * pulse.ns)

    assert _bloch(_rho(x90_detuned_wait)) == pytest.approx([0.0, -1.0, 0.0], abs=1e-2)


@pulse.kernel
def _ramsey(q, offset_hz: float, virtual_hz: float):
    # The tone offset_hz below the qudit; pulses 250 ns apart, centre to centre.
    d = get_drive_line(q)
    set_frequency(d.tone, F - offset_hz)
    drive(d, square(2 * pulse.ns, A90))
    wait(d, 248 * pulse.ns)
    shift_phase(d.tone, 2 * math.pi * virtual_hz * 250e-9)
    drive(d, square(2 * pulse.ns, A90))


def test_a_virtual_detuning_cancels_an_offset_of_the_same_sign():
    # 1 MHz for 250 ns is a quarter turn: cancelled, the two X90s make an X;
    # doubled, a half turn between them undoes the first.
    for offset in (1e6, -1e6):
        assert _rho(_ramsey, offset, offset)[1, 1].real == pytest.approx(1.0, abs=1e-3)
        assert _rho(_ramsey, offset, -offset)[1, 1].real == pytest.approx(0.0, abs=1e-3)


def test_shift_frequency_raises_the_tone():
    @pulse.kernel
    def shifted_ramsey(q, virtual_hz: float):
        d = get_drive_line(q)
        set_frequency(d.tone, F)
        shift_frequency(d.tone, 1e6)
        drive(d, square(2 * pulse.ns, A90))
        wait(d, 248 * pulse.ns)
        shift_phase(d.tone, 2 * math.pi * virtual_hz * 250e-9)
        drive(d, square(2 * pulse.ns, A90))

    assert {e["frequency_hz"] for e in _drives(_export(shifted_ramsey, 0.0))} == {F + 1e6}
    # The qudit is now 1 MHz below the tone: D = -1 MHz, which v = -1 MHz cancels.
    assert _rho(shifted_ramsey, -1e6)[1, 1].real == pytest.approx(1.0, abs=1e-3)
    assert _rho(shifted_ramsey, 1e6)[1, 1].real == pytest.approx(0.0, abs=1e-3)


@pulse.kernel
def _excursion_over_a_wait(q, delta_hz: float):
    d = get_drive_line(q)
    drive(d, square(2 * pulse.ns, A90))
    shift_frequency(d.tone, delta_hz)
    wait(d, 250 * pulse.ns)
    shift_frequency(d.tone, -delta_hz)
    drive(d, square(2 * pulse.ns, A90))


@pulse.kernel
def _excursion_over_a_pulse(q, delta_hz: float):
    d = get_drive_line(q)
    drive(d, square(2 * pulse.ns, A90))
    shift_frequency(d.tone, delta_hz)
    drive(d, square(250 * pulse.ns, 0.0))
    shift_frequency(d.tone, -delta_hz)
    drive(d, square(2 * pulse.ns, A90))


@pulse.kernel
def _shifted_after_the_wait(q, delta_hz: float):
    d = get_drive_line(q)
    drive(d, square(2 * pulse.ns, A90))
    wait(d, 250 * pulse.ns)
    shift_frequency(d.tone, delta_hz)
    drive(d, square(2 * pulse.ns, A90))


def test_a_frequency_shifted_over_a_wait_and_back_leaves_the_phase_it_gained():
    over_wait, over_pulse = _excursion_over_a_wait, _excursion_over_a_pulse
    for delta in (1e6, -1e6):
        gained = 2 * math.pi * delta * 250e-9
        last = _drives(_export(over_wait, delta))[-1]
        assert (last["frequency_hz"], last["phase_rad"]) == (F, pytest.approx(gained))
        assert _drives(_export(over_pulse, delta))[-1]["phase_rad"] == pytest.approx(gained)
        # A quarter turn either way: the second X90 turns about -Y or +Y.
        assert _bloch(_rho(over_wait, delta)) == pytest.approx([0.0, -1.0, 0.0], abs=1e-2)
        assert _rho(over_wait, delta) == pytest.approx(_rho(over_pulse, delta), abs=1e-6)
        # Made after the wait, the shift does not act during it.
        [_, after] = _drives(_export(_shifted_after_the_wait, delta))
        assert (after["frequency_hz"], after["phase_rad"]) == (F + delta, pytest.approx(0.0))
    assert _bloch(_rho(over_wait, 0.0)) == pytest.approx([0.0, 0.0, -1.0], abs=1e-2)


def test_an_exported_frame_phase_multiplies_the_samples_by_exp_minus_i_phase():
    phi = 0.7
    first, second = _drives(_export(_x90_phase_pulse, phi))
    assert (first["phase_rad"], second["phase_rad"]) == (0.0, pytest.approx(phi))
    assert second["amplitude"] == {"re": 0.3, "im": 0.0}
    assert second["frequency_hz"] == F
    # A square's envelope is 1: its samples are amplitude * exp(-i phase_rad).
    sample = complex(second["amplitude"]["re"], second["amplitude"]["im"])
    rendered = sample * np.exp(-1j * second["phase_rad"])
    played = _rho(_x90_turned_pulse, rendered.real, rendered.imag)
    assert played == pytest.approx(_rho(_x90_phase_pulse, phi), abs=1e-9)


def test_a_readout_result_is_the_level_and_if_takes_level_1():
    qs = pulse.quantum_system("transmon_surface17")

    @pulse.kernel
    def reset(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, d.get_calibration(pulse.gates.X))
        sync(d, r)
        m = readout(r, r.get_calibration(pulse.gates.READOUT))
        sync(d, r)
        if m:
            drive(d, d.get_calibration(pulse.gates.X))

    result = pulse.simulate(reset, qs.qudits[0], levels=2)
    excited = sum(b.probability for b in result.branches if b.levels == (1,))
    assert excited > 0.9
    assert np.asarray(result.density_matrix)[1, 1].real < 0.05
