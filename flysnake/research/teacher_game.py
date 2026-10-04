"""Ordinary Snake host: fixed RGB, actual spike receipts, no teacher policy."""
import copy
from .vision import ReadableGame
from flysnake.game import SnakeGame
from .teacher_spiking import VERSION, decode_spikes

CELL_IDS={'left':[523769],'right':[10360],'forward_left':[10162],'forward_right':[10059]}


class SpikingSnake(ReadableGame):
    def __init__(self,seed=0,size=12,*,parameter_sha256,window_ms=100):
        if not isinstance(parameter_sha256,str) or len(parameter_sha256)!=64:
            raise ValueError('Frozen neural parameter identity required')
        self.parameter_sha256=parameter_sha256;self.window_ms=window_ms
        super().__init__(seed=seed,size=size)

    def _validate_receipt(self,receipt):
        try:
            if receipt['version']!=VERSION:return 'wrong_neural_version'
            if receipt['episode_id']!=self.episode_id or receipt['step_id']!=self.step_id:return 'stale_step'
            if receipt['input_sha256']!=self.observation_hash():return 'wrong_pixels'
            if receipt['parameter_sha256']!=self.parameter_sha256:return 'wrong_parameters'
            if receipt['selected_cell_ids']!=CELL_IDS:return 'wrong_cell_ids'
            start,end=receipt['start_tick'],receipt['end_tick']
            if type(start) is not int or type(end) is not int or start<0 or end-start!=self.window_ms:return 'wrong_neural_window'
            if self._last_neural_end is not None and start!=round(self._last_neural_end*1000):return 'reused_or_discontinuous_window'
            decision=decode_spikes(receipt['counts'])
            if receipt['action']!=decision['action']:return 'action_disagrees_with_real_spikes'
        except (KeyError,TypeError,ValueError):return 'invalid_receipt'
        return None

    def apply_receipt(self,receipt):
        if self._pending_feedback is not None:return self._reject('feedback_pending')
        if self.terminal:return self._reject('episode_terminal')
        invalid=self._validate_receipt(receipt)
        if invalid:return self._reject(invalid)
        if receipt['action'] is None:
            before=self.step_id;pixels=self.render_observation();self.step_id+=1
            self._last_neural_end=receipt['end_tick']/1000.
            event={'accepted':True,'reason':'neural_silent','kind':'aversive','scalar_reward':-.1,
                'event_id':f'{self.seed}:{self.episode_id}:{before}:silent','episode_id':self.episode_id,
                'step_id':before,'next_step_id':self.step_id,'action':None,'terminal':False,
                'score':self.score,'score_delta':0,'death_reason':None,'outcome':None,
                'collision_at':None,'feedback_complete':False,'observation_hash':receipt['input_sha256']}
            self._pending_feedback=copy.deepcopy(event);self._feedback_observation=pixels
        else:
            event=SnakeGame.apply_receipt(self,{**receipt,'observation_hash':receipt['input_sha256'],
                'neural_time_end':receipt['end_tick']/1000.})
            if not event['accepted']:return event
            event['scalar_reward']={'reward':1.,'aversive':-1.,'none':0.}[event['kind']]
            if self._pending_feedback is not None:self._pending_feedback=copy.deepcopy(event)
        self.last_event=copy.deepcopy(event)
        return event


def attach_game_receipt(receipt,game,parameter_sha256):
    """Add transaction identity only; action/counts must remain brain output."""
    return {**receipt,'episode_id':game.episode_id,'step_id':game.step_id,
            'parameter_sha256':parameter_sha256}
