"""Fixed-world experimental scenes and graded Snake; contains no controller."""
import hashlib
import numpy as np
from PIL import Image, ImageDraw
from flysnake.game import SnakeGame, DIRECTIONS
from .protocol import Config, VERSION, decode


class LabGame(SnakeGame):
    def __init__(self, seed=0, size=6, calibration=None):
        self.calibration=calibration
        super().__init__(seed=seed,size=size)

    def render_observation(self):
        if self._feedback_observation is not None:
            return self._feedback_observation.copy()
        # Same world coordinates throughout; larger cells use the available field.
        # Spectral contrast is supplied as RGB, never as semantic neuron labels.
        tile=12
        ox,oy=(320-self.size*tile)//2,(180-self.size*tile)//2
        image=Image.new('RGB',(320,180),(85,85,85)); draw=ImageDraw.Draw(image)
        occupied=set(self.snake)
        for y in range(-1,self.size+1):
            for x in range(-1,self.size+1):
                p=(x,y); c=(3,3,3) if not self._inside(p) else (85,85,85)
                if p in occupied: c=(25,210,45)
                if p==self.food: c=(180,200,255)
                if p==self.snake[0]: c=(250,160,20)
                draw.rectangle((ox+x*tile,oy+y*tile,ox+(x+1)*tile-1,oy+(y+1)*tile-1),fill=c)
        hx,hy=self.snake[0]; dx,dy=DIRECTIONS[self.direction]
        cx,cy=ox+hx*tile+6,oy+hy*tile+6
        draw.line((cx,cy,cx+4*dx,cy+4*dy),fill=(250,250,250),width=3)
        return np.asarray(image,dtype=np.uint8).copy()

    def _validate_receipt(self, receipt):
        try:
            if receipt['version']!=VERSION or self.calibration is None: return 'wrong_protocol'
            if receipt['episode_id']!=self.episode_id or receipt['step_id']!=self.step_id: return 'stale_step'
            if receipt['observation_sha256']!=self.observation_hash(): return 'wrong_pixels'
            if receipt['calibration']!=self.calibration: return 'wrong_calibration'
            start,end=receipt['neural_time_start'],receipt['neural_time_end']
            if not np.isfinite([start,end]).all() or end-start!=Config().observation_ms: return 'invalid_window'
            if self._last_neural_end is not None and start<self._last_neural_end: return 'reused_window'
            decision=decode(receipt['counts'],end-start,self.calibration)
            if decision['action'] is None or decision['action']!=receipt['action']: return 'non_neural_action'
        except (KeyError,TypeError,ValueError):
            return 'invalid_receipt'
        return None

    def apply_receipt(self,receipt):
        # Reuse tested wall/self collision and feedback transaction semantics.
        if isinstance(receipt,dict):
            receipt={**receipt,'observation_hash':receipt.get('observation_sha256')}
        return super().apply_receipt(receipt)


def choice_scene(seed, side, *, size=6, distract=False):
    if side not in ('left','right'): raise ValueError('Invalid cue side')
    rng=np.random.default_rng(seed)
    game=LabGame(seed=seed,size=size)
    direction=int(rng.integers(0,4))
    head=(int(rng.integers(2,size-1)),int(rng.integers(2,size-1)))
    dx,dy=DIRECTIONS[direction]
    game.snake=[head,(head[0]-dx,head[1]-dy)]
    game.direction=direction
    tx,ty=DIRECTIONS[(direction+(-1 if side=='left' else 1))%4]
    game.food=(head[0]+tx,head[1]+ty)
    if distract:
        game.snake.append((head[0]-2*dx,head[1]-2*dy))
    return game.render_observation()


def neutral_scene(seed):
    # No food/correct action is involved in calibration.
    game=LabGame(seed=seed,size=6); game.food=None
    return game.render_observation()


def cue_scene(seed, side):
    """Elementary screen-side cue. No coordinate-to-action transformation yet."""
    frame=np.full((180,320,3),85,dtype=np.uint8)
    rng=np.random.default_rng(seed)
    # Start with a large hemifield cue. The inherited retinal map has highly
    # nonuniform coverage; tiny targets are a later curriculum, not stage zero.
    y=int(rng.integers(8,16)); x=(16 if side=='left' else 176)+int(rng.integers(0,5))
    frame[y:y+148,x:x+124]=(180,200,255)
    return frame
