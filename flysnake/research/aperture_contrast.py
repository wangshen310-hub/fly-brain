"""A brighter blue body keeps it distinct from the orange rear of the head.

This changes only rendered world pixels. No extra object features cross the
sensory boundary; camera geometry, original receptors and graph stay fixed.
"""
import hashlib
from pathlib import Path
from .aperture_native import NativeAperturePerceptionBrain
from .projection import CoveredGame
BODY_COLOR=(90,190,245)
VERSION='fixed-aperture-bright-blue-body-v1'


class ApertureContrastGame(CoveredGame):
    def render_observation(self):
        if self._feedback_observation is not None:return self._feedback_observation.copy()
        rgb=super().render_observation();tile=self.fixed_geometry['tile_pixels']
        ox,oy=self.fixed_geometry['board_origin']
        for x,y in self.snake[1:]:
            rgb[oy+y*tile:oy+(y+1)*tile,ox+x*tile:ox+(x+1)*tile]=BODY_COLOR
        return rgb


class NativeContrastApertureBrain(NativeAperturePerceptionBrain):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        neutral=[]
        for seed in range(901,909):
            game=ApertureContrastGame(seed=seed,size=self.geometry['board_cells'],geometry=self.geometry)
            game.food=None;neutral.append(game.render_observation())
        self.brain.calibrated=False;self.set_sensation([0.,0.]);self.brain.calibrate(neutral)
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.brain.source_signature=hashlib.sha256((self.brain.source_signature+source).encode()).hexdigest()
        self.identity.update(version=VERSION,body_color=list(BODY_COLOR),calibration=self.brain.calibration,
                             display_source_sha256=source,native_source_signature=self.brain.source_signature)
