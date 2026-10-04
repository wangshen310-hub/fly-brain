"""Sensory-event statistics and native injection invariants, not learning tests."""
import ctypes as C
from types import SimpleNamespace
import numpy as np
import pytest
from flysnake.research.poisson_orn import draw_events, build, EventKernel


def test_poisson_events_replay_and_have_independent_rates():
    indices = np.array([3, 7, 9], np.int32); rates = np.array([0., 40., 90.])
    first = draw_events(np.random.default_rng(42), indices, rates, 100, 1000000, .1)
    second = draw_events(np.random.default_rng(42), indices, rates, 100, 1000000, .1)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    ticks, targets = first
    assert 100 <= ticks.min() and ticks.max() < 1000100
    assert not (targets == 3).any()
    # 100s of input with a predeclared five-standard-deviation count bound.
    for cell, rate in zip(indices[1:], rates[1:]):
        observed = np.count_nonzero(targets == cell); expected = 100 * rate
        assert abs(observed - expected) < 5 * np.sqrt(expected)
    silent = draw_events(np.random.default_rng(0), indices, np.zeros(3), 0, 100, .1)
    assert all(len(x) == 0 for x in silent)
    with pytest.raises(ValueError):
        draw_events(np.random.default_rng(0), indices, [-1., 1., 2.], 0, 100, .1)


def fixture():
    n = 2
    b = SimpleNamespace(n=n, ptr=np.array([0, 1, 1], np.int64), post=np.array([1], np.int32),
        weight=np.array([10.], np.float32), v=np.full(n, -52., np.float32),
        g=np.zeros(n, np.float32), refractory=np.zeros(n, np.int16),
        drive=np.zeros(n, np.float32), previous_drive=np.zeros(n, np.float32),
        queue=np.zeros((19, n), np.int32), queue_count=np.zeros(19, np.int32),
        clock=np.array([0], np.int64), counts=np.zeros(n, np.int32),
        active=np.arange(n, dtype=np.int32), flags=np.ones(n, np.uint8),
        nactive=np.array([n], np.int32), last=np.full(n, -1, np.int64),
        kc_mask=np.zeros(n, np.uint8), dan_index=np.full(n, -1, np.int8),
        eligibility=np.zeros(n, np.float64), eligibility_last=np.zeros(n, np.int64),
        plastic_edge=np.empty(0, np.int64), plastic_pre=np.empty(0, np.int32),
        baseline_weight=np.empty(0, np.float32), dan_gain=np.empty(0, np.float32),
        modulation=np.zeros(n, np.float32), modulation_last=np.zeros(n, np.int64),
        modulation_mask=np.zeros(n, np.uint8), rest=np.full(n, -52., np.float32),
        adaptation=np.zeros(n, np.float32))
    return b


def invoke(b, advance, steps):
    pointers = ['ptr', 'post', 'weight', 'v', 'g', 'refractory', 'drive', 'previous_drive',
                'queue', 'queue_count', 'clock']
    advance(b.n, *[getattr(b, k).ctypes.data for k in pointers], steps, .1,
        *[getattr(b, k).ctypes.data for k in ['counts', 'active', 'flags', 'nactive', 'last']],
        *[getattr(b, k).ctypes.data for k in ['kc_mask', 'dan_index', 'eligibility', 'eligibility_last']],
        0, *[getattr(b, k).ctypes.data for k in ['plastic_edge', 'plastic_pre', 'baseline_weight', 'dan_gain']],
        .001, 1000., .1, 0, *[getattr(b, k).ctypes.data for k in ['modulation', 'modulation_last',
        'modulation_mask', 'rest', 'adaptation']], 8., 200.)


def test_real_native_event_stage_and_refractory_override(tmp_path):
    library, _ = build(tmp_path)
    signature = SimpleNamespace(argtypes=[C.c_int] + [C.c_void_p]*11 + [C.c_int, C.c_float]
        + [C.c_void_p]*9 + [C.c_int] + [C.c_void_p]*4 + [C.c_float]*3 + [C.c_int]
        + [C.c_void_p]*5 + [C.c_float]*2, restype=None)
    kernel = EventKernel(signature, library, 2, [0], 68.75)
    kernel.ticks = np.array([0, 2], np.int64); kernel.targets = np.array([0, 0], np.int32)
    free = fixture(); invoke(free, kernel, 4)
    assert free.counts.tolist() == [2, 0]
    assert free.refractory[0] == 0 and free.g[0] == 0
    kernel.mask.fill(0)
    refractory = fixture(); invoke(refractory, kernel, 4)
    assert refractory.counts.tolist() == [1, 0]
    assert refractory.refractory[0] > 0
    # A synapses-stage event delivered on the same tick as the previous
    # event's spike is cleared by that tick's reset, matching declared timing.
    kernel.mask[0] = 1; kernel.ticks = np.array([0, 1], np.int64)
    same_tick = fixture(); invoke(same_tick, kernel, 3)
    assert same_tick.counts.tolist() == [1, 0]
    # A voltage event at tick0 fires at tick1, and its real outgoing edge
    # arrives at tick19, retaining the existing 1.8ms synaptic delay.
    kernel.mask[0] = 1; kernel.ticks = np.array([0], np.int64); kernel.targets = np.array([0], np.int32)
    delayed = fixture(); invoke(delayed, kernel, 19)
    assert delayed.g[1] == 0
    invoke(delayed, kernel.plain, 1)
    assert delayed.g[1] == 10.
