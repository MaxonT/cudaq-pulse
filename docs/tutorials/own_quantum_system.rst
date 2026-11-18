..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Describe Your Own Quantum System
================================

The tutorials so far ran on ``transmon_surface17``, a model of a published
17-qubit processor. Yours has its own frequencies, coherence times and
couplings.

What you'll learn:

- how to describe qudits and a coupling from measured numbers;
- how calibrated π pulses get their amplitude;
- how one compile can serve every qudit;
- how to follow a recalibration and save the description to a file.

.. themed-figure:: figures/transmon_surface17.svg
   :alt: The 17 qudits of transmon_surface17 and its 24 couplings: nine data qubits, D1 to D9, in a 3 by 3 grid, and the ancillas X1, X2, X3, X4, Z1, Z2, Z3, Z4, each coupled to the four data qubits around it or, at the edge, to two. Each qudit is labelled with its 0-1 frequency: the data qubits between 3.74 and 4.14 GHz, the ancillas between 5.88 and 6.33 GHz.
   :width: 90%

   The shipped ``transmon_surface17`` quantum system, after Krinner et al.,
   Nature 605, 669 (2022), drawn from its qudits and couplings.

A quantum system of your own is described the same way, qudit by qudit and
coupling by coupling.

Qudits and a coupling
---------------------

A quantum system is built from its qudits and the couplings between them.
Start with the imports:

.. code-block:: python

   import json
   from pathlib import Path

   import cudaq_pulse as pulse
   from cudaq_pulse.quantum_systems import (
       Coupling,
       DragPulse,
       PhysicalQudit,
   )

Each qudit takes its index, its 0-1 frequency in Hz, its T1 and T2* in
microseconds, and the rest by keyword. Both share the shape of their π
pulse:

.. code-block:: python

   x_pulse = {pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0)}

   qudit_0 = PhysicalQudit(
       index=0,
       frequency_hz=5.1 * pulse.ghz,
       t1_us=60.0,
       t2_star_us=40.0,
       anharmonicity_hz=-220 * pulse.mhz,
       calibrations=x_pulse,
   )

The second qudit sits 200 MHz higher, with shorter coherence times:

.. code-block:: python

   qudit_1 = PhysicalQudit(
       index=1,
       frequency_hz=5.3 * pulse.ghz,
       t1_us=45.0,
       t2_star_us=30.0,
       anharmonicity_hz=-210 * pulse.mhz,
       calibrations=x_pulse,
   )

Build the pair with a 3 MHz coupling between them:

.. code-block:: python

   pair = pulse.quantum_system(
       name="my_pair",
       qudits={0: qudit_0, 1: qudit_1},
       couplings=[Coupling((0, 1), strength_hz=3 * pulse.mhz)],
   )
   print(pair)

.. code-block:: text
   :class: output

   <QuantumSystem my_pair (2 qudits)>

What the numbers mean:

- the anharmonicity is the shift of the 1-2 transition below the 0-1 one.
  It makes a transmon a usable qubit, and simulating its third level needs
  it;
- the coupling's ``strength_hz`` is the exchange rate g / 2π.

.. themed-figure:: figures/pair_levels.svg
   :alt: The three lowest levels of each transmon of my_pair, qudit 0: 0-1 at 5.10 GHz and 1-2 at 4.88 GHz; qudit 1: 0-1 at 5.30 GHz and 1-2 at 5.09 GHz. The 1-2 transition sits below the 0-1 by the anharmonicity. The two are coupled with g / 2 pi = 3 MHz.
   :width: 80%

   The three lowest levels of each qudit of ``pair``.

What to notice: a drive at the 0-1 frequency stays off the 1-2 line,
hundreds of megahertz away.

Calibrated pulses
-----------------

The calibration gives each π pulse a shape, a 40 ns DRAG pulse with a
10 ns width, but no amplitude. The quantum system works the amplitude out
from the shape and the qudit's drive strength:

.. code-block:: python

   x_cal = pair.qudits[0].get_calibration(pulse.gates.X)
   print(f"{x_cal.duration_ns} ns DRAG, amplitude {x_cal.amplitude:.4f}")

.. code-block:: text
   :class: output

   40.0 ns DRAG, amplitude 0.1313

.. themed-figure:: figures/calibrated_x.svg
   :alt: The calibrated pi pulse of qudit 0 on its drive line: a 40 ns DRAG pulse of amplitude 0.1313, one smooth bump that starts and ends at zero.
   :width: 100%

   Qudit 0's calibrated π pulse, as its drive line plays it.

What to notice: the pulse is one smooth bump. Its height is the amplitude
the quantum system worked out.

That amplitude turns the qubit by π. Its π/2 pulse, ``pulse.gates.X90``, is
the same pulse at half the amplitude.

Run a kernel on it
------------------

Kernels run on whichever quantum system their qudits come from, with no
other change:

.. code-block:: python

   @pulse.kernel
   def excite(qudit):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = drive_line.get_calibration(pulse.gates.X)

       drive(drive_line, flip)
       sync(drive_line, readout_line)
       return readout(readout_line, square(300 * pulse.ns, 0.05))

Run it on each qudit of the pair:

.. code-block:: python

   for qudit in pair.qudits:
       result = pulse.simulate(excite, qudit)
       excited = result.level_populations(qudit)[1]
       print(f"qudit {qudit.index} excited: {excited:.3f}")

   print("compilations:", excite.cache_info().compiled)

.. code-block:: text
   :class: output

   qudit 0 excited: 0.993
   qudit 1 excited: 0.991
   compilations: 2

What to notice:

- both π pulses land within 1% of the excited state; what's missing is the
  decay during the 300 ns readout;
- the kernel compiled twice, once per qudit, because it reads each line's
  calibration while it compiles.

One compile for every qudit
---------------------------

Annotate the qudit parameter with ``pulse.Qudit``, and the compiled program
takes the qudit as an argument. Then one compile serves every qudit.

Such a kernel can't read a calibration, since the qudit is known only when
it is called. So the host reads it and passes the amplitude in:

.. code-block:: python

   @pulse.kernel
   def excite_any(qudit: pulse.Qudit, amplitude: float):
       drive_line = get_drive_line(qudit)
       readout_line = get_readout_line(qudit)
       flip = gaussian(
           duration=40 * pulse.ns,
           amplitude=amplitude,
           sigma=10 * pulse.ns,
       )
       drive(drive_line, flip)
       sync(drive_line, readout_line)
       return readout(readout_line, square(300 * pulse.ns, 0.05))

Run it on both qudits, each with its own calibrated amplitude:

.. code-block:: python

   for qudit in pair.qudits:
       amplitude = qudit.get_calibration(pulse.gates.X).amplitude
       result = pulse.simulate(excite_any, qudit, amplitude)
       excited = result.level_populations(qudit)[1]
       print(f"qudit {qudit.index} excited: {excited:.3f}")

   print("compilations:", excite_any.cache_info().compiled)

.. code-block:: text
   :class: output

   qudit 0 excited: 0.993
   qudit 1 excited: 0.991
   compilations: 1

.. themed-figure:: figures/one_compile.svg
   :alt: The kernel excite, run on both qudits of the pair, compiles 2 programs, one per qudit. excite_any, which takes the qudit as an argument, compiles 1, which serves both.
   :width: 100%

   The programs each kernel compiled for the two qudits.

What to notice: the drive's frequency still follows each qudit, because a
drive line plays at its own qudit's frequency unless the kernel says
otherwise.

Follow a recalibration
----------------------

Measured values drift. ``replace`` makes a new quantum system with some
values changed and the rest kept, the way ``dataclasses.replace`` does.

Move qudit 0 up by 2 MHz:

.. code-block:: python

   old_qudit = pair.physical_qudits[0]
   new_qudit = old_qudit.replace(frequency_hz=5.102e9)
   qudits = {**pair.physical_qudits, 0: new_qudit}
   recalibrated = pair.replace(qudits=qudits)

The same kernel follows it. Export it for both systems and compare the
drive's frequency:

.. code-block:: python

   systems = {"before": pair, "after": recalibrated}
   for label, quantum_system in systems.items():
       scheduled = excite.lower(quantum_system.qudits[0])
       program = scheduled.export_to_controller()
       for event in program["events"]:
           if event["kind"] == "drive":
               frequency_ghz = event["frequency_hz"] / 1e9
               print(f"{label:6} drive at {frequency_ghz:.3f} GHz")

.. code-block:: text
   :class: output

   before drive at 5.100 GHz
   after  drive at 5.102 GHz

The recalibrated program plays its drive 2 MHz higher, with no change to
the kernel.

.. themed-figure:: figures/recalibration.svg
   :alt: The frequency of excite's drive in its export for qudit 0: 5100 MHz before the recalibration and 5102 MHz after, 2 MHz higher, with the same kernel.
   :width: 80%

   The drive's frequency in each export.

What to notice: quantum systems are read-only, so the original ``pair`` is
unchanged, and programs compiled for it stay valid.

Save it to a file
-----------------

A quantum system is plain data. ``as_dict`` gives it as a dictionary that
JSON can hold, and ``QuantumSystem.load`` reads the file back:

.. code-block:: python

   path = Path("my_pair.json")
   text = json.dumps(pair.as_dict(), indent=2)
   path.write_text(text)

   loaded = pulse.QuantumSystem.load(path)
   print("same system:", loaded == pair)
   print("format:", json.loads(text)["format"])

.. code-block:: text
   :class: output

   same system: True
   format: cudaq-pulse.system/1

Mistakes, all at once
---------------------

When a description has mistakes, building it reports all of them at once,
each with where it is.

A ``layout`` picks which of a quantum system's qudits a program numbers 0,
1, 2 and so on. Ask for one that doesn't exist:

.. code-block:: python

   try:
       pulse.quantum_system("transmon_surface17", layout=[0, 99, 99])
   except pulse.QuantumSystemDefinitionError as error:
       for problem in error.problems:
           print(f"{problem.path}: {problem.message}")

.. code-block:: text
   :class: output

   layout[1]: carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)
   layout[2]: carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)

A layout is how a program written for a few qubits is placed on a larger
quantum system. A file with problems reports the line and column of each.

Where to go next
----------------

- :example:`23_define_a_qpu.py` adds the control system: two controllers
  with their wiring, a feedback link between them, and the whole QPU saved
  as one file.
- :example:`22_runtime_qudits.py` compiles one kernel for every qudit and
  passes the extra frequencies it needs as arguments.
- :example:`transmon/transmon_surface17_rabi.py` uses the shipped system's
  published π pulse on qubit D5.
- :example:`26_drift_following.py` recompiles when a drifting quantum
  system changes.

:doc:`../user_guide/quantum_systems` covers every field of a quantum
system, residual ZZ, layouts and the shipped calibration with its sources.
