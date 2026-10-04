"""Frozen full-native comparison of tonic and independent-event sensory input.

No game, score, weight search or action labels. The no-event export must first
match the original kernel, including complete dynamic arrays every 10ms. Actual
cell traces then test a declared transfer hypothesis with three input RNG seeds.
"""
from pathlib import Path
import argparse
from dataclasses import replace
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from flysnake.research.poisson_orn import PoissonORNBrain, PoissonConfig, VERSION
from flysnake.research.vision import light_to_current
from flysnake.research.raw_turn import decode_raw_turn
from flysnake.research.paired import writer_lock, digest
from flysnake.lab.experiment import atomic_json
from assay_native_compartment import sources as parent_sources, inputs, immutable_arrays


def sources():
    result = parent_sources()
    for name in ('flysnake/research/input_isolation.py', 'flysnake/research/poisson_orn.py',
                 'flysnake/research/kernels/poisson_sensory.cpp',
                 'flysnake/research/kernels/poisson_sensory.provenance.json',
                 'scripts/assay_poisson_encoding.py'):
        result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output):
        path = args.output / 'status.json'
        if path.exists(): raise ValueError('Preserve prior encoding evidence')
        source = sources()
        report = {'protocol': VERSION, 'status': 'loading', 'sources': source,
                  'process_id': os.getpid(), 'started_at': time.time(), 'rows': [],
                  'learning': False, 'game_played': False, 'behavioral_learning_proven': False,
                  'settings': {'cue_ms': 300, 'wash_ms': 300, 'bin_ms': 10,
                               'concentration': .8, 'maximum_event_hz': 60.,
                               'seeds': [100201, 100202, 100203],
                               'comparisons': ['tonic', 'events_native_refractory', 'events_zero_refractory']}}
        for name in source:
            target = args.output / 'sources' / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        atomic_json(path, report)
        try:
            brain = PoissonORNBrain(args.output / 'build'); b = brain.b
            report['brain'] = brain.identity
            original_advance = brain.original_advance
            # Use active full-graph current input for the unchanged-export audit,
            # rather than verifying only a silent or disconnected fixture.
            stim = [(ix, light_to_current(.8, -45 - b.rest[ix], brain.retina_config))
                    for ix in list(brain.orn.values())[0].values()]
            snapshots = []
            for advance in (original_advance, brain.event_kernel.plain):
                b.advance = advance; b.reset(keep_memory=True); states = []
                for _ in range(30):
                    brain.original_neural_step(np.zeros(len(b.retina)), 10,
                                               stimulation=stim, learning=False, lamina_bias=0.)
                    states.append({k: digest(getattr(b, k)) for k in b.fields})
                snapshots.append(states)
            if snapshots[0] != snapshots[1]:
                raise AssertionError('Independent no-event kernel differs from retained native dynamics')
            report['no_event_full_native_equivalence'] = {
                'equal': True, 'boundaries': 30, 'all_dynamic_fields': list(b.fields),
                'actual_native_spikes': int(b.total_spikes), 'snapshots': snapshots[0]}
            b.advance = brain.event_kernel
            atomic_json(path, report)
            unique_pre = np.unique(brain.pre)
            apl = np.array([np.flatnonzero(b.ids == cell)[0] for cell in (10540, 10977)])
            selected = np.unique(np.r_[unique_pre, apl, brain.event_indices,
                                       *brain.memory_readouts.values()])
            samples = []; totals = []; pattern_blocks = {}
            conditions = [('tonic', None, False), ('events_native_refractory', 100201, False)]
            conditions += [('events_zero_refractory', seed, True) for seed in (100201, 100202, 100203)]
            blank = np.full((180, 320, 3), 85, np.uint8)
            for mode, seed, refractory_free in conditions:
                block = {}
                cues = inputs(brain) if mode == 'tonic' or seed == 100201 else inputs(brain)[1:]
                for cue, rgb, concentration in cues:
                    brain.events_config = replace(brain.events_config, seed=100201 if seed is None else seed,
                                                   refractory_free=refractory_free)
                    brain.event_kernel.mask.fill(0)
                    if refractory_free: brain.event_kernel.mask[brain.event_indices] = 1
                    brain.rng = np.random.default_rng(100201 if seed is None else seed)
                    brain.reset_trial()
                    rows = []; trace = []; phase_totals = []
                    for phase, physical in (('cue', concentration), ('wash', np.zeros((2, 2)))):
                        aggregate = np.zeros(b.n, np.int32); event_counts = 0
                        for _ in range(30):
                            if mode == 'tonic':
                                pulses = []
                                for channel, groups in enumerate(brain.orn.values()):
                                    for side, value in zip(('L', 'R'), physical[channel]):
                                        ix = groups[side]
                                        pulses.append((ix, 0. if value == 0. else light_to_current(
                                            value, -45-b.rest[ix], brain.retina_config)))
                                b.advance = original_advance
                                emitted, _ = brain.original_neural_step(np.zeros(len(b.retina)), 10,
                                    stimulation=pulses, learning=False, lamina_bias=0.)
                            else:
                                b.advance = brain.event_kernel
                                receipt = brain.integrate(blank, 10, physical, learning=False)
                                emitted = brain.last_counts.copy(); event_counts += receipt['sensory_event_count']
                            aggregate += emitted; trace.append(emitted[selected].copy())
                            rows.append({'phase': phase, 'full_counts_sha256': digest(emitted)})
                        phase_totals.append(aggregate); totals.append(aggregate)
                        rows[-1].update(phase_full_counts_sha256=digest(aggregate),
                            gamma1_KC_active=int(np.count_nonzero(aggregate[unique_pre])),
                            gamma1_KC_spikes=int(aggregate[unique_pre].sum()),
                            APL_spikes=aggregate[apl].tolist(),
                            readouts={k: aggregate[ix].tolist() for k, ix in brain.memory_readouts.items() if k != 'KC'},
                            ORN_spikes={k: {side: int(aggregate[ix].sum()) for side, ix in groups.items()}
                                        for k, groups in brain.orn.items()}, sensory_event_count=event_counts,
                            motor=decode_raw_turn({k: aggregate[ix].tolist() for k, ix in brain.groups.items()}, 300))
                    samples.append(np.asarray(trace))
                    row = {'mode': mode, 'seed': seed, 'refractory_free': refractory_free,
                           'cue': cue, 'concentrations': concentration.tolist(), 'bins': rows,
                           'cue_summary': rows[29], 'wash_summary': rows[59]}
                    report['rows'].append(row); block[cue] = phase_totals[0][unique_pre]
                    report.update(status='running', phase=f'{mode}/{seed}/{cue}', updated_at=time.time())
                    atomic_json(path, report)
                    print(json.dumps({'mode': mode, 'seed': seed, 'cue': cue,
                        'KC_active': row['cue_summary']['gamma1_KC_active'],
                        'PPL101': row['cue_summary']['readouts']['PPL101'],
                        'APL': row['cue_summary']['APL_spikes'],
                        'ORN': row['cue_summary']['ORN_spikes']}), flush=True)
                norm = np.linalg.norm(block['DM1']) * np.linalg.norm(block['DM4'])
                pattern_blocks[f'{mode}/{seed}'] = {
                    'gamma1_KC_count': len(unique_pre),
                    'raw_cue_cosine_similarity': float(np.dot(block['DM1'].astype(float), block['DM4'].astype(float))/norm) if norm else None,
                    'active_gamma1_KC': {k: int(np.count_nonzero(v)) for k, v in block.items()}}
            report['encoding_summary'] = pattern_blocks
            report['native_spikes'] = {'file': 'native-spikes.npz', 'sha256': immutable_arrays(
                args.output / 'native-spikes.npz', counts=np.asarray(totals), selected_counts=np.asarray(samples),
                cell_ids=b.ids, selected_indices=selected, gamma1_KC_indices=unique_pre,
                plastic_edges=brain.edges, plastic_pre=brain.pre, base_weights=brain.base)}
            if digest(b.weight) != brain.initial_all_weights_sha256:
                raise AssertionError('Frozen encoding assay changed graph weights')
            report.update(status='complete', completed_at=time.time(), all_weights_unchanged=True,
                interpretation='Bundled external transduction comparison; refractory control separates that assumption. '
                    'Independent sensory RNG, not random actions. No conditioning or game learning demonstrated.')
            atomic_json(path, report)
            print(json.dumps({'status': 'complete', 'encoding_summary': pattern_blocks}), flush=True)
        except Exception:
            report.update(status='error', error=traceback.format_exc()); atomic_json(path, report); raise


if __name__ == '__main__': main()
