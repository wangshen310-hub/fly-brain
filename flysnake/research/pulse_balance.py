"""Honor raw neural pulse balance before the locked calibrated BCI.

Equal/near-equal raw turn counts mean forward. This fixes the old baseline
subtraction artifact without changing sensory input, topology or native spikes.
"""
import hashlib
from pathlib import Path
import numpy as np
from flysnake.lab.protocol import decode,Config
from .actor_credit import NativeActorCreditBrain,self_action_credit
from .projection import CoveredGame

DECODER_VERSION='native-raw-pulse-balance-gate-v1'
LEARNING_VERSION='native-pulse-balanced-self-action-credit-v1'
CELL_IDS={'left':[523769],'forward':[10162,10059],'right':[10360]}


def decode_balanced(counts,duration_ms,calibration,margin=.2):
    if not np.isfinite(margin) or not 0<margin<1:raise ValueError('Bounded pulse margin required')
    result=decode(counts,duration_ms,calibration)
    result['version']=DECODER_VERSION
    result['ungated_calibrated_action']=result['action']
    left,right=counts['left'][0],counts['right'][0]
    result['raw_turn_difference']=right-left
    result['raw_turn_balance_limit']=margin*max(left,right,1)
    if sum(sum(v) for v in counts.values())>0 and abs(right-left)<=result['raw_turn_balance_limit']:
        result.update(action='forward',reason='raw_turn_balance')
    return result


class PulseBalancedBrain(NativeActorCreditBrain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.identity.update(version=LEARNING_VERSION,
            decoder={'version':DECODER_VERSION,'margin':.2,'raw_balance_precedes_calibration':True,
                     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def observe(self,rgb,concentration,noise=None):
        receipt,_=super().observe(rgb,concentration,noise)
        decision=decode_balanced(receipt['counts'],self.brain.config.observation_ms,self.brain.calibration)
        receipt.update(decision)
        # Recompute credit for the FINAL actual action, never the old action.
        counts=self.brain.last['counts'][self.pre]
        trace=self_action_credit(counts,self.mean_counts,self.brain.b.weight[self.edges],self.sides,
                                 receipt,self.brain.calibration,self.actor_config)
        from .paired import digest
        receipt['eligibility_sha256']=digest(trace)
        return receipt,trace


class PulseBalancedGame(CoveredGame):
    def __init__(self,*,stage,**kwargs):
        self.stage=stage;super().__init__(**kwargs)

    def _sample_food(self):
        if self.stage!='uniform_food':
            from flysnake.game import DIRECTIONS
            x,y=self.snake[0]
            available=[(x+dx,y+dy) for dx,dy in DIRECTIONS
                       if self._inside((x+dx,y+dy)) and (x+dx,y+dy) not in self.snake]
            if available:return self._rng.choice(available)
        return super()._sample_food()

    def _validate_receipt(self,receipt):
        try:
            if receipt['version']!=DECODER_VERSION or self.calibration is None:return 'wrong_protocol'
            if receipt['selected_cell_ids']!=CELL_IDS:return 'wrong_native_cells'
            if receipt['episode_id']!=self.episode_id or receipt['step_id']!=self.step_id:return 'stale_step'
            if receipt['observation_sha256']!=self.observation_hash():return 'wrong_pixels'
            if receipt['calibration']!=self.calibration:return 'wrong_calibration'
            start,end=receipt['neural_time_start'],receipt['neural_time_end']
            if not np.isfinite([start,end]).all() or start<0 or end-start!=Config().observation_ms:return 'invalid_window'
            if self._last_neural_end is not None and start<self._last_neural_end:return 'reused_window'
            actual=decode_balanced(receipt['counts'],end-start,self.calibration)
            if actual['action'] is None or actual['action']!=receipt['action']:return 'non_neural_action'
            if receipt['scores']!=actual['scores']:return 'changed_native_scores'
        except (KeyError,TypeError,ValueError):return 'invalid_receipt'
        return None
