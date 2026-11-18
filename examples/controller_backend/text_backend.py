# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small controller backend, which plays a program as text.

It plays each drive and readout of an exported program as one line of text,
skips waits and syncs, and refuses anything else, such as a pulse inside a
branch, with ``IncompatibleBackendError``. Registered in the running process,
it emits a program and passes ``backends.validate``. The "From Kernel to
Controller" tutorial shows it a piece at a time.
"""

# [start imports]
from collections.abc import Mapping
from typing import Any

from cudaq_pulse import Controller, backends
from cudaq_pulse.backends import IncompatibleBackendError
from cudaq_pulse.controller import FORMAT

Shots = list[list[str]]

# [end imports]


# [start describe]
def describe(event: Mapping[str, Any]) -> str:
    """One line of text for a drive or readout."""
    start = event["start_cycles"]
    kind = event["kind"]
    line = event["line"]
    shape = event["waveform"]["shape"]
    frequency = f"{event['frequency_hz']:.6g} Hz"
    return f"{start} {kind} line {line} {shape} at {frequency}"


# [end describe]


# [start class]
class TextBackend:
    name = "text"
    formats = (FORMAT,)

    def control_system(self) -> Controller:
        return Controller(
            clock_ghz=1.0,
            instruction_cycles=2,
            min_idle_cycles=2,
        )

    # [end class]
    # [start emit]
    def emit(self, program: Mapping[str, Any]) -> list[str]:
        if program["format"] != FORMAT:
            message = f"cannot read {program['format']}"
            raise IncompatibleBackendError(message)

        played = []
        for event in program["events"]:
            if event["kind"] in ("wait", "sync"):
                continue
            playable = event["kind"] in ("drive", "readout")
            if not playable or event["conditions"]:
                message = f"cannot play event {event['index']}"
                raise IncompatibleBackendError(message)
            played.append(describe(event))
        return played

    # [end emit]
    # [start run]
    def run(self, artifact: Any, *, shots_count: int = 1) -> Shots:
        shots = []
        for _ in range(shots_count):
            shots.append(list(artifact))
        return shots

    # [end run]


if __name__ == "__main__":
    import cudaq_pulse as pulse
    from cudaq_pulse.quantum_systems import PhysicalQudit

    qudit = PhysicalQudit(
        index=0,
        frequency_hz=5 * pulse.ghz,
        t1_us=50.0,
        t2_star_us=30.0,
        anharmonicity_hz=-200 * pulse.mhz,
    )
    system = pulse.quantum_system(
        name="one-qubit",
        qudits={0: qudit},
        clock_ghz=1.0,
    )

    @pulse.kernel
    def drive_and_measure(qudit):
        drive_line = get_drive_line(qudit)
        readout_line = get_readout_line(qudit)
        pulse_shape = gaussian(
            duration=40 * pulse.ns,
            amplitude=0.5,
            sigma=10 * pulse.ns,
        )
        drive(drive_line, pulse_shape)
        sync(drive_line, readout_line)
        readout(readout_line, square(200 * pulse.ns, 0.05))

    backends.register("text", TextBackend)
    backend = backends.get("text")
    compiled = pulse.compile(
        drive_and_measure,
        quantum_system=system,
        control_system=backend.control_system(),
    )
    program = compiled.lower(system.qudits[0]).export_to_controller()
    for line in backend.run(backend.emit(program))[0]:
        print(line)

    report = backends.validate("text", controller=Controller(clock_ghz=1.0))
    print(str(report).splitlines()[-1])
    backends.unregister("text")
