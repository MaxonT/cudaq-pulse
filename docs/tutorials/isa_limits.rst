..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Getting More out of Your Controller's Instruction Set
=====================================================

.. note::

   Experimental, as is :doc:`isa_lowering`, which this page builds on.

Every controller has limits. This tutorial takes six common ones. For
each, it shows what the compiler does, so that a small instruction set is
enough.

What you'll learn:

- how the compiler meets feedback latency, within and across controllers
- how two controllers in different boxes act as one QPU
- how loops, decoder deadlines and waveform memory reach your ISA
- which limits are checked when a program compiles, not on the hardware

Set up
------

This page uses the toy controllers and the lowering pass of
:doc:`isa_lowering`. The full code is in
:example:`isa_lowering/toy_isa.py`. Save it beside your own code.

.. code-block:: python

   import dataclasses
   import textwrap

   import cudaq_pulse as pulse
   from cudaq_pulse.controller import ControllerLimitError
   from toy_isa import lower, toy_control_system, toy_qpu

   qpu = toy_qpu()
   first, second = toy_control_system().controllers
   q0, q1, q2 = qpu.qudits
   print("model:           ", first.model)
   print("feedback latency:", first.feedback_latency_cycles, "cycles")

.. code-block:: text
   :class: output

   model:            toy.sequencer
   feedback latency: 80 cycles

The QPU has three qudits on two controllers:

- The first controller drives qudits 0 and 1, and reads both on one
  feedline.
- The second drives and reads qudit 2.
- Outcomes reach the other controller 40 cycles later.
- Both count from a 250 MHz reference clock, and a decoder sits on a
  link.

A refused program says why in one long message. This helper prints it
wrapped to the page:

.. code-block:: python

   def show(error):
       message = str(error.violations[0])
       for line in textwrap.wrap(message, width=64):
           print(line)

Feedback latency
----------------

A common constraint: a branch can use a measurement only some time after
the acquisition ends. It takes longer still when another controller made
the measurement.

Your ISA needn't count those cycles. The compiler does.

A branch on another controller's readout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here the first controller reads qudit 0. The second flips qudit 2 when
it reads 1:

.. code-block:: python

   @pulse.kernel
   def flip_other(measured, other):
       readout_line = get_readout_line(measured)
       drive_line = get_drive_line(other)
       probe = square(duration=200, amplitude=0.05)
       flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)

       outcome = readout(readout_line, probe)
       sync(readout_line, drive_line)
       if outcome:
           drive(drive_line, flip)

The default schedule refuses it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default schedule starts the flip at cycle 200, before the bit can
have arrived. The compile refuses, and names the cycle and the fix:

.. code-block:: python

   compiled = pulse.compile(flip_other, quantum_system=qpu)
   try:
       compiled.lower(q0, q2)
   except ControllerLimitError as error:
       show(error)

.. code-block:: text
   :class: output

   controllers[1] (toy.sequencer): the drive on line 1 at 200
   branches on the readout of controllers[0] (toy.sequencer) ending
   at 200, whose outcome reaches it 120 cycles later, at 320 (120
   of feedback and peer latency, and 0 of the two controllers'
   start uncertainty); start it at 320 or later: schedule with rcp,
   which waits for it, or wait 120 cycles before it on its line

The rcp schedule places it
~~~~~~~~~~~~~~~~~~~~~~~~~~

The resource-constrained schedule, ``rcp``, keeps every branch that far
from its readout. Each controller's part then lowers on its own:

.. code-block:: python

   compiled = pulse.compile(
       flip_other,
       quantum_system=qpu,
       schedule="rcp",
   )
   split = compiled.lower(q0, q2)

   part = split.export_to_controller(controller=0)
   for line in str(lower(part)).splitlines():
       print(line)

.. code-block:: text
   :class: output

   .chan ch0.0 readout q0
   .wave w0 200
   set_freq ch0.0 5000000000
   acquire ch0.0 w0 0.05 m0

The first controller only acquires. The second waits for the bit, then
branches on it:

.. code-block:: python

   part = split.export_to_controller(controller=1)
   for line in str(lower(part)).splitlines():
       print(line)

.. code-block:: text
   :class: output

   .chan ch0 drive q2
   .wave w0 40
   wait 80
   branch !m0 end0
   set_freq ch0 5600000000
   play ch0 w0 0.5
   end0:

.. toy-flow:: figures/toy_flow.json
   :scene: flip_other
   :fallback: figures/isa_feedback.svg
   :alt: Timeline of flip_other on two toy controllers, in nanoseconds of the 1 GHz clock. Controller 0 acquires q0 from 0 to 200 ns into bit m0. The bit is ready on controller 0 80 ns later, after its feedback latency, and on controller 1 another 40 ns later, after the peer latency. Controller 1 plays the flip of q2 from 320 ns, only when the bit is 1.

   The two listings in time, on the 1 GHz clock, so a nanosecond is a
   cycle. Bit m0 is ready for the second controller at cycle 320, where
   its ``wait 80`` ends.

What to notice:

- The flip waits 120 cycles: 80 of feedback latency, then 40 from one
  controller to the other.
- Both parts use the program's event indices, so the bit is ``m0`` on
  both.

A controller that can't forward outcomes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sending a bit to another controller is a capability, ``forward_outcome``.
A controller without it is refused this program:

.. code-block:: python

   local_only = dataclasses.replace(
       first,
       supports=["play", "readout", "wait", "sync", "branch"],
   )
   system = pulse.ControlSystem(
       controllers=[local_only, second],
       reference_clock_ghz=0.25,
   )
   compiled = pulse.compile(
       flip_other,
       quantum_system=qpu,
       schedule="rcp",
       control_system=system,
   )

The refusal names the readout that needs it:

.. code-block:: python

   try:
       compiled.lower(q0, q2)
   except ControllerLimitError as error:
       show(error)

.. code-block:: text
   :class: output

   controllers[0] (toy.sequencer): the readout on line 0 (qudit 0)
   at cycle 0 needs 'forward_outcome', which the controller does
   not support; run that part of the program on a controller that
   supports it, or write it without 'forward_outcome'

Takeaway: describe the latencies once, and every branch is placed where
its bit has arrived.

One QPU from two controllers
----------------------------

A common constraint: controllers in different boxes have no shared
notion of "now". Nothing in the ISA lines them up.

The compiler checks that they can be lined up, before anything runs.

A pulse on each controller
~~~~~~~~~~~~~~~~~~~~~~~~~~

This kernel syncs a pulse on each controller:

.. code-block:: python

   @pulse.kernel
   def joint(a, b):
       first_line = get_drive_line(a)
       second_line = get_drive_line(b)
       x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
       flat = square(duration=40, amplitude=0.1)

       drive(first_line, x90)
       sync(first_line, second_line)
       drive(first_line, flat)
       drive(second_line, flat)

Without a shared clock
~~~~~~~~~~~~~~~~~~~~~~

Without a shared reference clock, each controller's start is known only
to within one of its cycles. The sync is refused, with both ways out:

.. code-block:: python

   unclocked = pulse.ControlSystem(controllers=[first, second])
   compiled = pulse.compile(
       joint,
       quantum_system=qpu,
       control_system=unclocked,
   )
   try:
       compiled.lower(q0, q2)
   except ControllerLimitError as error:
       show(error)

.. code-block:: text
   :class: output

   controllers[0] (toy.sequencer): the sync at 40 lines up the
   drive on line 0 with the pulses of controllers[1]
   (toy.sequencer), but they land within 1 of its cycles of where
   they are meant to (1 without a shared reference_clock_ghz), more
   than its alignment_tolerance_cycles of 0; share a
   reference_clock_ghz between them, or raise its
   alignment_tolerance_cycles to 1

With the QPU's clock
~~~~~~~~~~~~~~~~~~~~

With the QPU's reference clock, the sync is in both parts, under one
index, at one cycle:

.. code-block:: python

   together = pulse.compile(joint, quantum_system=qpu).lower(q0, q2)

   for number in range(2):
       part = together.export_to_controller(controller=number)
       print(f"controller {number}:")
       for event in part["events"]:
           print(f"  {event['index']}  {event['kind']:<5} "
                 f"at {event['start_cycles']}")

.. code-block:: text
   :class: output

   controller 0:
     0  drive at 0
     1  sync  at 40
     2  drive at 40
   controller 1:
     1  sync  at 40
     3  drive at 40

Takeaway: both sequencers just wait until cycle 40. The ISA needs no
handshake instruction.

Rolled loops and instruction memory
-----------------------------------

A common constraint: a sequencer holds a few hundred or a few thousand
instructions. A long pulse train has to stay a loop.

A pulse train
~~~~~~~~~~~~~

Here is a train of 1,000 pulses:

.. code-block:: python

   @pulse.kernel
   def train(qudit):
       drive_line = get_drive_line(qudit)
       x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
       for _ in range(1000):
           drive(drive_line, x90)
           wait(drive_line, 8)

On a controller without loops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The toy sequencer has no loop instruction, and room for 256 pulses. So
the train is refused, with the reason:

.. code-block:: python

   try:
       pulse.compile(train, quantum_system=qpu).lower(q0)
   except ControllerLimitError as error:
       show(error)

.. code-block:: text
   :class: output

   controllers[0] (toy.sequencer): 1000 drives and readouts; the
   controller plays at most 256. It does not support 'loop', so
   rolled loops are unrolled for it: the loop of 1000 iterations
   from cycle 0 plays 999 more

On a controller with loops
~~~~~~~~~~~~~~~~~~~~~~~~~~

Give the controller the ``loop`` capability, and the train compiles:

.. code-block:: python

   capabilities = list(first.supports or [])
   capabilities.append("loop")
   looping = dataclasses.replace(first, supports=capabilities)
   compiled = pulse.compile(
       train,
       quantum_system=qpu,
       control_system=looping,
   )
   rolled = compiled.lower(q0)

   flat = rolled.export_to_controller()
   print(len(flat["events"]), "events")

.. code-block:: text
   :class: output

   2000 events

The default format, ``/1``, lists all 2,000 executions. The structured
format keeps the train as one loop:

.. code-block:: python

   structured = rolled.export_to_controller(version=2)
   loop = structured["events"][0]

   print(loop["kind"], loop["trips"], "trips")
   print(loop["period_cycles"], "cycles each")

   for event in loop["body"]:
       print(" ", event["kind"])

.. code-block:: text
   :class: output

   loop 1000 trips
   48 cycles each
     play
     wait

Against the controller's instruction memory:

.. memory-gauge:: figures/controller_memory.json
   :rows: train train_loop
   :columns: pulses

Takeaway: a sequencer with a loop counter lowers the body once.

A decoder with a deadline
-------------------------

A common constraint in QEC: syndromes go to a decoder off the sequencer.
Its answer comes back after a latency the sequencer can't predict.

The compiler places the call, and the moment to give up on it.

The decoder
~~~~~~~~~~~

The decoder here is a parity check. Its budget is 100 ns:

.. code-block:: python

   def parity(bits: pulse.Bits) -> pulse.Bits:
       return [bits[0] ^ bits[1]]


   decode = pulse.classical(
       parity,
       link="decoder",
       max_latency_ns=100,
   )

A round with a deadline
~~~~~~~~~~~~~~~~~~~~~~~

A QEC round reads two syndromes and asks the decoder. It waits at most
60 ns for the answer:

.. code-block:: python
   :class: continues

   @pulse.kernel
   def qec_round(data, ancilla):
       probe = square(duration=200, amplitude=0.05)
       flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)

       syndrome_0 = readout(get_readout_line(data), probe)
       syndrome_1 = readout(get_readout_line(ancilla), probe)
       request = decode.submit([syndrome_0, syndrome_1])
       in_time, fix = request.result(timeout=60 * pulse.ns)

Then it applies the correction, or the fallback when the answer is late:

.. code-block:: python

       if fix[0]:
           drive(get_drive_line(data), flip)
       if not in_time:
           drive(get_drive_line(ancilla), flip)

The result starts with a flag, ``in_time``. It is true when the answer
came by the deadline.

- When it did, ``fix`` is the decoder's correction.
- When it didn't, ``fix`` is all zeros, and the round takes its
  fallback. Here the fallback flips the ancilla, so its next readout
  marks the round as undecoded.

The listing
~~~~~~~~~~~

Lower the round:

.. code-block:: python

   compiled = pulse.compile(
       qec_round,
       quantum_system=qpu,
       schedule="rcp",
   )
   qec = compiled.lower(q0, q1)
   listing = lower(qec.export_to_controller())

Both syndromes are read at once, on the two tones of the feedline:

.. code-block:: python

   for line in listing.code:
       print(line)

.. code-block:: text
   :class: output

   set_freq ch0.0 5000000000
   acquire ch0.0 w0 0.05 m0
   set_freq ch0.1 5300000000
   acquire ch0.1 w0 0.05 m1
   wait 70
   submit L0 m0 m1
   wait 18
   poll L0 a3
   branch !a3 end0
   set_freq ch1 5000000000
   play ch1 w1 0.5
   end0:
   branch a3.ok end1
   set_freq ch2 5300000000
   play ch2 w1 0.5
   end1:

What to notice:

- The request leaves at cycle 280, when both bits are on the bus.
- The ``poll`` is at cycle 352: the submit's 4 cycles, the 60 ns
  deadline, and the link's 8 ns back.
- The branch on ``a3`` is the correction. The branch on ``a3.ok`` is the
  fallback.

.. themed-figure:: figures/isa_deadline.svg
   :alt: Timeline of one QEC round on the toy QPU: both syndromes are read at once on the feedline's two tones, and the request leaves for the decoder at 280 ns. The kernel waits for the answer with a 60 ns timeout, plus the link's 8 ns back, so it is due at 352 ns. Two flips that play only on the shots that need them start there: the correction and the fallback.
   :width: 100%

   The round in time, from the compiled program. The request leaves when
   both bits are on the bus, and the answer is due after the 60 ns
   deadline and the link's way back.

What to notice: the 80 ns between the readouts and the request is the
feedback latency, before the bits are on the bus.

Each branch reads its own result of the answer. In the export, each
condition says which one:

.. code-block:: python

   for event in qec.export_to_controller()["events"]:
       for condition in event["conditions"]:
           line = event["line"]
           answer = condition["answers"][0]
           result = condition["result"]
           print(f"line {line}: answer {answer}, result {result}")

.. code-block:: text
   :class: output

   line 2: answer 3, result 1
   line 3: answer 3, result 0

Result 0 is the in-time flag. Result 1 is the decoder's first value.

A decoder between two controllers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both controllers reach the decoder's link. So the answer can steer a
pulse on either one.

In this round, the first controller reads the syndromes. The data qudit
it corrects is on the second controller:

.. code-block:: python

   @pulse.kernel
   def correct_across(ancilla_0, ancilla_1, data):
       probe = square(duration=200, amplitude=0.05)
       flip = gaussian(duration=40, amplitude=0.5, sigma=10.0)

       syndrome_0 = readout(get_readout_line(ancilla_0), probe)
       syndrome_1 = readout(get_readout_line(ancilla_1), probe)
       request = decode.submit([syndrome_0, syndrome_1])
       in_time, fix = request.result(timeout=60 * pulse.ns)

       if fix[0]:
           drive(get_drive_line(data), flip)
       if not in_time:
           drive(get_drive_line(ancilla_1), flip)

Lower it with qudits 0 and 1 as the ancillas and qudit 2 as the data,
then print each controller's listing:

.. code-block:: python

   compiled = pulse.compile(
       correct_across,
       quantum_system=qpu,
       schedule="rcp",
   )
   across = compiled.lower(q0, q1, q2)

   for number in range(2):
       part = across.export_to_controller(controller=number)
       print(f"; controller {number}")
       for line in lower(part).code:
           print(line)

.. code-block:: text
   :class: output

   ; controller 0
   set_freq ch0.0 5000000000
   acquire ch0.0 w0 0.05 m0
   set_freq ch0.1 5300000000
   acquire ch0.1 w0 0.05 m1
   wait 70
   submit L0 m0 m1
   wait 18
   poll L0 a3
   branch a3.ok end1
   set_freq ch1 5300000000
   play ch1 w1 0.5
   end1:
   ; controller 1
   wait 88
   poll L0 a3
   branch !a3 end0
   set_freq ch0 5600000000
   play ch0 w0 0.5
   end0:

What to notice:

- Only the first controller submits. The request leaves at cycle 280,
  as before.
- Both controllers poll at cycle 352. Each branches on the result it
  needs: the second on the fix, the first on the in-time flag.
- The second controller never branches on ``m0`` or ``m1``. It uses only
  the decoder's answer.

Here is the round playing, once with the answer in time and once late:

.. toy-flow:: figures/toy_flow.json
   :scene: correct_across

The shaded stretch on the link runs from the request's departure to its
deadline. The decoder may answer at any point in it. The polls, 8 cycles
after the deadline, take the answer if it came.

Takeaway: the deadline is a number the compiler schedules around. Your
sequencer only polls once, where the compiler put the poll.

Waveform memory and oscillators
-------------------------------

A common constraint: envelopes live in a small memory next to the
converters. A new shape costs memory, but a new gain costs nothing.

A sweep is one table entry
~~~~~~~~~~~~~~~~~~~~~~~~~~

An amplitude sweep compiles once, with a ``float`` argument:

.. code-block:: python

   @pulse.kernel
   def rabi(qudit, amplitude: float):
       shape = gaussian(
           duration=40,
           amplitude=amplitude,
           sigma=10.0,
       )
       drive(get_drive_line(qudit), shape)


   sweep = pulse.compile(rabi, quantum_system=qpu)
   for amplitude in (0.2, 0.4):
       program = sweep.lower(q0, amplitude).export_to_controller()
       print(lower(program).code[-1])

.. code-block:: text
   :class: output

   play ch0 w0 0.2
   play ch0 w0 0.4

Every value plays the same entry, ``w0``, at its own gain.

New shapes and new frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thirty pulse widths are thirty entries. And a channel holds only so many
oscillator frequencies:

.. code-block:: python

   @pulse.kernel
   def widths(qudit):
       drive_line = get_drive_line(qudit)
       for k in range(30):
           sigma = 4.0 + 0.5 * k
           shape = gaussian(duration=40, amplitude=0.25, sigma=sigma)
           drive(drive_line, shape)

And three frequencies on one line:

.. code-block:: python

   @pulse.kernel
   def hops(qudit):
       drive_line = get_drive_line(qudit)
       x90 = gaussian(duration=40, amplitude=0.25, sigma=10.0)
       for _ in range(3):
           drive(drive_line, x90)
           shift_frequency(drive_line.tone, 1 * pulse.mhz)

Both are checked against the description when they compile:

.. code-block:: python

   for kernel in (widths, hops):
       try:
           pulse.compile(kernel, quantum_system=qpu).lower(q0)
       except ControllerLimitError as error:
           show(error)

.. code-block:: text
   :class: output

   controllers[0] (toy.sequencer): line 0: its waveforms take 1200
   samples; the controller's waveform memory holds 1024
   controllers[0] (toy.sequencer): line 0: 3 tone frequencies; the
   controller holds 2 per line

What to notice:

- Thirty widths overflow the channel's 1,024 samples.
- Three frequencies overflow its two oscillator settings.
- Both are compile errors, not a failed upload.

Here are the three programs against the channel's memories:

.. memory-gauge:: figures/controller_memory.json
   :rows: rabi widths hops
   :columns: samples frequencies

Phase is cheaper still. The compiler folds constant phase shifts into
the next pulse's frame, as it did with the Ramsey sequence's virtual Z.

Takeaway: your description's memory sizes are checked for you, before
anything reaches the hardware.

Where to go next
----------------

- :doc:`isa_lowering` builds the toy controllers and the pass this page
  uses.
- :doc:`realtime_decoding` runs a QEC round with a decoder deadline and a
  fallback table.
- :example:`23_define_a_qpu.py` loads controllers from an installed
  description and saves a QPU as one file.
- :doc:`../user_guide/controller_backends` documents each limit a
  controller states.
