// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// An integrated readout acquires an IQ point and no level: nothing reads its
// level, on any path it may take, and a call takes it only as an IQ argument.

func.func @branch_on_a_point(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %r1, %t1, %m = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  // expected-error @+1 {{'pulse.is_set' op reads the level of an integrated readout, which acquires an IQ point and no level; read it out with mode iq to branch on its level}}
  %c = pulse.is_set %m : !pulse.measurement -> i1
  return %c : i1
}

// -----

func.func @level_of_a_point(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i64 {
  %r1, %t1, %m = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  // expected-error @+1 {{'pulse.outcome' op reads the level of an integrated readout}}
  %k = pulse.outcome %m : !pulse.measurement -> i64
  return %k : i64
}

// -----

// One arm of an if reads out a point.
func.func @a_point_in_one_arm(%b: i1, %r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %m = scf.if %b -> (!pulse.measurement) {
    %r1, %t1, %m1 = pulse.readout %r0, %wf, %t0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %m1 : !pulse.measurement
  } else {
    %r1, %t1, %m1 = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %m1 : !pulse.measurement
  }
  // expected-error @+1 {{'pulse.is_set' op reads the level of an integrated readout}}
  %res = pulse.is_set %m : !pulse.measurement -> i1
  return %res : i1
}

// -----

// A loop carries the last iteration's point.
func.func @carried_by_a_loop(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %r1, %t1, %m0 = pulse.readout %r0, %wf, %t0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %rn, %tn, %mn = scf.for %i = %lb to %ub step %step iter_args(%r = %r1, %t = %t1, %m = %m0) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    %r2, %t2, %m2 = pulse.readout %r, %wf, %t, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r2, %t2, %m2 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  // expected-error @+1 {{'pulse.is_set' op reads the level of an integrated readout}}
  %res = pulse.is_set %mn : !pulse.measurement -> i1
  return %res : i1
}

// -----

// Sent as bits, a call would read its level; as an IQ argument it reads the
// point, as it may.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>,
    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 8.0 : f64,
                              fixed_ns = 4.0 : f64, slots = 64 : i64,
                              slot_bytes = 1024 : i64}}} {

classical.callable @decide {args = ["bits:1"], device = 0 : i64,
    dispatch_key = 3994489633 : i64, latency_max_ns = 100.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 100.0 : f64,
    mode = "bounded", results = ["f64"], submit = "eager"}
classical.callable @fit {args = ["iq:1"], device = 0 : i64,
    dispatch_key = 2934209580 : i64, latency_max_ns = 100.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 100.0 : f64,
    mode = "bounded", results = ["f64"], submit = "eager"}

func.func @sent(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  %r1, %t1, %m = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l0, @fit(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  // expected-error @+1 {{'classical.submit' op sends value 0 as bits:1, the level of a readout, but it comes from an integrated readout, which acquires an IQ point and no level; read it out with mode iq, or pass it as an iq argument}}
  %l2, %h2 = classical.submit %l1, @decide(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  return
}

}
