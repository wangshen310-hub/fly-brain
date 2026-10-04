"""Read-only synthesis of actual native assays, not a learning-score generator."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_run(name):
    directory = ROOT / 'runtime' / name
    data = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    if data['status'] != 'complete':
        raise ValueError(f'{name} is not a completed assay')
    for relative, sha in data['sources'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Historical assay dependency changed: {relative}')
        if hashlib.sha256((directory / 'sources' / relative).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Historical source snapshot changed: {relative}')
    if 'native_spikes' in data:
        entry = data['native_spikes']
        if hashlib.sha256((directory / entry['file']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Actual native count arrays changed')
    return directory, data


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first_dir, first = read_run('native-compartment-mechanism-audit-20261002-v1')
    isolated_dir, isolated = read_run('native-input-isolation-20261002-v1')
    paired_dir, paired = read_run('native-gamma1-conditioning-20261002-v1')
    event_dir, event = read_run('native-poisson-encoding-20261002-v1')
    restore = json.loads((ROOT / 'runtime/native-gamma1-restore-20261002-v1/result.json').read_text(encoding='utf-8'))
    if restore['status'] != 'complete': raise ValueError('No completed native memory replay')
    if restore['source_sha256'] != hashlib.sha256((ROOT / 'scripts/verify_gamma1_memory.py').read_bytes()).hexdigest():
        raise ValueError('Restore verifier source changed')
    arms = {}
    for arm, record in paired['arms'].items():
        for cp in record['checkpoints']:
            if hashlib.sha256((paired_dir / cp['file']).read_bytes()).hexdigest() != cp['sha256']:
                raise ValueError('Committed native memory changed')
        final = record['checkpoints'][-1]
        with np.load(paired_dir / final['file'], allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata'])); weights = z['weights'].copy()
        if metadata['memory_sha256'] != final['memory_sha256']:
            raise ValueError('Memory manifest does not match weights')
        original = {row['cue']: row['receipt'] for row in paired['before']}
        retained = {row['cue']: row['receipt'] for row in paired['retained'][arm]}
        arms[arm] = {
            'mean_fraction': record['trials'][-1]['mean_fraction'],
            'minimum_fraction': record['trials'][-1]['minimum_fraction'],
            'retained_MBON11_spikes': {cue: row['readouts']['gamma1_MBON11'] for cue, row in retained.items()},
            'retained_raw_actions': {cue: row['action'] for cue, row in retained.items()},
            'cue_response_suppression': {cue: 1-sum(retained[cue]['readouts']['gamma1_MBON11']) /
                                             sum(original[cue]['readouts']['gamma1_MBON11'])
                                        for cue in ('DM1', 'DM4')},
            'checkpoint_sha256': final['sha256'], 'weights_sha256': final['memory_sha256'],
            'native_replay_equal': restore['arms'][arm]['native_trial_replay_equal'],
            'restored_frozen_probes_equal': restore['arms'][arm]['restored_frozen_probes_equal']}
    pre_us = paired['arms']['paired']['trials'][0]['rows'][1]
    reference = isolated['summaries']['no_background']
    result = {
        'generated_at': time.time(), 'version': 'native-mechanism-audit-synthesis-v1',
        'graph': {'neurons': first['brain']['neurons'], 'edges': first['brain']['graph_edges'],
                  'gamma1_existing_plastic_edges': first['brain']['plastic_edges']},
        'sensory_background': isolated['summaries'],
        'kernel_no_event_equivalence': {k: v for k, v in event['no_event_full_native_equivalence'].items() if k != 'snapshots'},
        'event_encoding': event['encoding_summary'],
        'before_pairing': {'changed_existing_edges': pre_us['changed_edges'],
                           'mean_efficacy_change': pre_us['mean_fraction_change'],
                           'actual_PPL101_spikes': pre_us['receipt']['readouts']['PPL101']},
        'conditioning_arms': arms,
        'learning_gates': {
            'raw_motor_readout_interventions': first['gates']['motor_interface'],
            'existing_gamma1_weights_can_change_motor_counts': first['gates']['untaught_gamma1_intervention_changes_motor_counts'],
            'zero_external_background_gives_zero_neutral_gamma1_KC_spikes': reference['active_gamma1_KC'][0] == 0,
            'native_event_rule_changes_existing_weights': arms['paired']['mean_fraction'] < 1,
            'frozen_control_stays_at_baseline': arms['frozen']['mean_fraction'] == 1,
            'trial_boundary_memory_and_actual_native_receipts_restore': all(a['native_replay_equal'] and a['restored_frozen_probes_equal'] == 3 for a in arms.values()),
            'stimulus_specific_reward_association_demonstrated': False,
            'ordinary_snake_learning_demonstrated': False},
        'scope': 'Finite native mechanism tests. Four conditioning arms share one deterministic model and one round. '
                 'Three event-input seeds are encoding diagnostics, not independent training seeds. No Snake was played.',
        'reasons_not_to_resume_bulk_training': [
            'Cue-only and wrong-time arms also depress most shared sensory KC->MBON11 inputs.',
            'Actual cue-driven PPL101 spikes cause LTD before any added aversive stimulus.',
            'Gamma1 event rule is LTD-only and is not a calibrated appetitive alpha1 model.',
            'Spike differences and preserved weights do not demonstrate a useful body/world representation.'],
        'primary_kernel_reference': 'https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py',
        'primary_refractory_reference': 'https://brian2.readthedocs.io/en/stable/user/refractoriness.html',
        'interpretation': 'Persistent native plasticity demonstrated, selective reward association failed. No claim that the connectome cannot learn.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise ValueError('Preserve previous synthesis')
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'gates': result['learning_gates'], 'event_encoding': result['event_encoding']}, ensure_ascii=False))


if __name__ == '__main__': main()
