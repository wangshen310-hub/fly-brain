"""A rate-calibrated photoreceptor proxy and a readable fixed-world board.

The original graph uses spiking LIF proxies for biological graded receptors.
Linear current alone can put several different colors below spike threshold.
This explicit engineering transfer maps linear light to a target rate and then
to a constant LIF current; it knows nothing about game objects or actions.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from flysnake.lab.brain import LabBrain
from flysnake.lab.environment import LabGame
from flysnake.lab.protocol import sample_linear
from flysnake.game import DIRECTIONS

VISION_VERSION = 'rate-calibrated-retina-v1'
PALETTE = {'background': (85,85,85), 'wall': (3,3,3), 'head': (245,245,245),
           'body': (40,155,195), 'food': (100,240,70), 'heading': (3,3,3)}


@dataclass(frozen=True)
class RetinaConfig:
    dark_hz: float = 5.
    white_hz: float = 60.
    membrane_ms: float = 20.
    refractory_ms: float = 2.2


def light_to_current(light, gap, config=RetinaConfig()):
    """Invert the isolated LIF rate equation; recurrence may change actual rates."""
    values = np.array([config.dark_hz,config.white_hz,config.membrane_ms,config.refractory_ms])
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Invalid finite positive retinal parameters')
    if not 0 < config.dark_hz < config.white_hz < 1000/config.refractory_ms:
        raise ValueError('Invalid finite photoreceptor firing-rate range')
    light = np.asarray(light, dtype=float)
    gap = np.asarray(gap, dtype=float)
    if not np.isfinite(light).all() or not np.isfinite(gap).all() or (gap <= 0).any():
        raise ValueError('Invalid light or neural voltage gap')
    rates = config.dark_hz + (config.white_hz-config.dark_hz)*np.clip(light,0,1)
    charging_ms = 1000/rates-config.refractory_ms
    return gap/(-np.expm1(-charging_ms/config.membrane_ms))


class RateRetinaBrain(LabBrain):
    def __init__(self):
        super().__init__()
        self.retina_config = RetinaConfig()
        self.source_signature = hashlib.sha256((self.source_signature + VISION_VERSION +
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() +
            json.dumps(asdict(self.retina_config),sort_keys=True)).encode()).hexdigest()

    def integrate(self, rgb, duration, stimulus=None):
        b = self.b
        linear = sample_linear(rgb, b.uv)
        target = linear @ np.asarray([.2126,.7152,.0722])
        r8rgb = sample_linear(rgb,b.r8_uv)
        target8 = r8rgb[np.arange(len(b.r8)),b.r8_channel]
        counts = np.zeros(b.n,dtype=np.int32)
        for _ in range(duration//10):
            alpha = 1-np.exp(-10/self.config.sensory_tau_ms)
            self.retinal_light += alpha*(target-self.retinal_light)
            self.r8_light += alpha*(target8-self.r8_light)
            pulses = [(b.retina,light_to_current(self.retinal_light,-45-b.rest[b.retina],self.retina_config)),
                      (b.r8,light_to_current(self.r8_light,-45-b.rest[b.r8],self.retina_config))]
            if stimulus is not None:
                pulses.append(stimulus)
            c,_ = b.step(np.zeros(len(b.retina)),10,learning=False,stimulation=pulses)
            counts += c
        self.neural_ms += duration
        if not np.isfinite(b.v).all() or not np.isfinite(b.g).all():
            raise FloatingPointError('Nonfinite neural state')
        return counts

    def metadata(self):
        return {**super().metadata(),'vision_version':VISION_VERSION,'retina_config':asdict(self.retina_config),
                'sensory_assumption':'Declared LIF rate proxy; not a validated natural phototransduction model'}


class ReadableGame(LabGame):
    def observation_geometry(self):
        tile = min(24,180//(self.size+2))
        return {'projection':'fixed_world','tile_pixels':tile,'board_cells':self.size,
                'board_origin':[(320-self.size*tile)//2,(180-self.size*tile)//2],
                'image_size':[320,180],'wall_cells':1}

    def render_observation(self):
        if self._feedback_observation is not None:
            return self._feedback_observation.copy()
        g = self.observation_geometry()
        tile = g['tile_pixels']; ox,oy = g['board_origin']
        image = Image.new('RGB',(320,180),PALETTE['background'])
        draw = ImageDraw.Draw(image)
        occupied = set(self.snake)
        for y in range(-1,self.size+1):
            for x in range(-1,self.size+1):
                p = (x,y)
                name = 'background' if self._inside(p) else 'wall'
                if p in occupied: name = 'body'
                if p == self.food: name = 'food'
                if p == self.snake[0]: name = 'head'
                draw.rectangle((ox+x*tile,oy+y*tile,ox+(x+1)*tile-1,oy+(y+1)*tile-1),fill=PALETTE[name])
        hx,hy = self.snake[0]; dx,dy = DIRECTIONS[self.direction]
        cx,cy = ox+(hx+.5)*tile,oy+(hy+.5)*tile
        # Large dark orientation glyph within the bright head; no auxiliary labels.
        tip = (cx+dx*tile*.38,cy+dy*tile*.38)
        base = (cx-dx*tile*.18,cy-dy*tile*.18)
        draw.polygon([tip,(base[0]-dy*tile*.23,base[1]+dx*tile*.23),
                          (base[0]+dy*tile*.23,base[1]-dx*tile*.23)],fill=PALETTE['heading'])
        return np.asarray(image,dtype=np.uint8).copy()
