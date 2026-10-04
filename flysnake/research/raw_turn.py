"""Direct actual turn-cell pulse readout, independent of neutral offsets.

This is a fixed brain-machine interface, not an action policy. It receives no
game state and cannot predict or avoid collisions. Current runs are unchanged.
"""
import hashlib
from pathlib import Path
import numpy as np
from flysnake.lab.protocol import decode,Config
from .pulse_balance import PulseBalancedBrain,PulseBalancedGame,CELL_IDS

VERSION='native-direct-raw-turn-pulse-readout-v1'


def decode_raw_turn(counts,duration_ms,margin=.2):
    if not np.isfinite(margin) or not 0<margin<1:raise ValueError('Bounded fixed raw turn margin required')
    # Use the common count/window validation, with no fitted offsets or scales.
    decision=decode(counts,duration_ms,{'baseline_hz':[0.,0.,0.],'scale_hz':[1.,1.,1.]})
    left,right=counts['left'][0],counts['right'][0]
    decision.update(version=VERSION,raw_turn_difference=right-left,
                    raw_turn_balance_limit=margin*max(left,right,1),neutral_offsets_used=False)
    if sum(sum(c) for c in counts.values())==0:return decision
    if abs(right-left)<=decision['raw_turn_balance_limit']:
        decision.update(action='forward',reason='raw_turn_balance')
    else:decision.update(action='right' if right>left else 'left',reason='stronger_raw_turn_cell')
    return decision


class RawTurnBrain(PulseBalancedBrain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.identity.update(version=VERSION,decoder={'version':VERSION,'margin':.2,
            'neutral_offsets_used':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def observe(self,rgb,concentration,noise=None):
        receipt,trace=super().observe(rgb,concentration,noise)
        receipt.update(decode_raw_turn(receipt['counts'],self.brain.config.observation_ms))
        # This independent module is for fixed-readout audits; inherited actor
        # eligibility is not valid for the new raw readout and MUST NOT train.
        receipt['inherited_eligibility_valid_for_learning']=False
        return receipt,trace


class RawTurnGame(PulseBalancedGame):
    def _validate_receipt(self,receipt):
        try:
            if receipt['version']!=VERSION or self.calibration is None:return 'wrong_protocol'
            if receipt['selected_cell_ids']!=CELL_IDS:return 'wrong_native_cells'
            if receipt['episode_id']!=self.episode_id or receipt['step_id']!=self.step_id:return 'stale_step'
            if receipt['observation_sha256']!=self.observation_hash():return 'wrong_pixels'
            if receipt['calibration']!=self.calibration:return 'wrong_calibration'
            start,end=receipt['neural_time_start'],receipt['neural_time_end']
            if not np.isfinite([start,end]).all() or start<0 or end-start!=Config().observation_ms:return 'invalid_window'
            if self._last_neural_end is not None and start<self._last_neural_end:return 'reused_window'
            actual=decode_raw_turn(receipt['counts'],end-start)
            if actual['action'] is None or actual['action']!=receipt['action']:return 'non_neural_action'
            if receipt['scores']!=actual['scores']:return 'changed_native_scores'
        except (KeyError,TypeError,ValueError):return 'invalid_receipt'
        return None
