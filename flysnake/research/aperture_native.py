"""Fixed-world optical aperture connected to original native receptors.

The original point-sampling classes are left untouched for old checkpoints.
This module creates an independent sensory identity and calibration.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import MethodType
import numpy as np
from .sensory_basis import NativeSensoryBasisBrain,BasisConfig
from .optical_sampling import ApertureConfig,sample_aperture_linear,coverage_uv,identity as optical_identity
from .viewport_candidates import find_viewport
from .projection import CoveredGame
from .olfaction import OdorConfig
from .vision import light_to_current

VERSION='native-fixed-aperture-retina-v1'


def aperture_integrate(brain,rgb,duration,stimulus=None):
    b=brain.b;config=brain.aperture_config
    linear=sample_aperture_linear(rgb,b.uv,config)
    target=linear@np.asarray([.2126,.7152,.0722])
    r8rgb=sample_aperture_linear(rgb,b.r8_uv,config)
    target8=r8rgb[np.arange(len(b.r8)),b.r8_channel]
    counts=np.zeros(b.n,dtype=np.int32)
    for _ in range(duration//10):
        alpha=1-np.exp(-10/brain.config.sensory_tau_ms)
        brain.retinal_light+=alpha*(target-brain.retinal_light)
        brain.r8_light+=alpha*(target8-brain.r8_light)
        pulses=[(b.retina,light_to_current(brain.retinal_light,-45-b.rest[b.retina],brain.retina_config)),
                (b.r8,light_to_current(brain.r8_light,-45-b.rest[b.r8],brain.retina_config))]
        if stimulus is not None:pulses.append(stimulus)
        c,_=b.step(np.zeros(len(b.retina)),10,learning=False,stimulation=pulses);counts+=c
    brain.neural_ms+=duration
    if not np.isfinite(b.v).all() or not np.isfinite(b.g).all():raise FloatingPointError('Nonfinite native optical state')
    return counts


class NativeAperturePerceptionBrain(NativeSensoryBasisBrain):
    def __init__(self,*,size=12,config=BasisConfig(),odor=OdorConfig(),aperture=ApertureConfig(radius_pixels=1.5)):
        super().__init__(size=size,config=config,odor=odor)
        self.brain.aperture_config=aperture
        self.brain.integrate=MethodType(aperture_integrate,self.brain)
        combined=np.r_[coverage_uv(self.brain.b.uv,aperture),coverage_uv(self.brain.b.r8_uv,aperture)]
        self.geometry=find_viewport(combined,size,pixel_stride=1,tile_stride=1)
        optics=optical_identity(np.r_[self.brain.b.uv,self.brain.b.r8_uv],aperture)
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.brain.source_signature=hashlib.sha256((self.brain.source_signature+source+json.dumps(optics,sort_keys=True)).encode()).hexdigest()
        neutral=[]
        for seed in range(901,909):
            game=CoveredGame(seed=seed,size=size,geometry=self.geometry);game.food=None
            neutral.append(game.render_observation())
        self.brain.calibrated=False;self.set_sensation([0.,0.]);self.brain.calibrate(neutral)
        self.identity.update(version=VERSION,geometry=self.geometry,calibration=self.brain.calibration,
            optics=optics,source_sha256=source,native_source_signature=self.brain.source_signature,
            optical_config=asdict(aperture),
            assumption='Fixed anatomical finite-aperture linear radiance proxy and native LIF rate transfer; natural fly optics unvalidated')
