"""Static finite-aperture light sampling at unchanged anatomical UVs.

The aperture is an engineering camera proxy, not measured fly optical angles.
No game object, coordinates, action, reward or graph changes enter sampling.
"""
from dataclasses import dataclass,asdict
import hashlib
from pathlib import Path
import numpy as np

VERSION='fixed-anatomical-bilinear-aperture-v1'


@dataclass(frozen=True)
class ApertureConfig:
    radius_pixels: float=.75
    subdivisions: int=3


def validate_uv(uv):
    uv=np.asarray(uv,dtype=float)
    if (uv.ndim!=2 or uv.shape[1]!=2 or not len(uv)
            or not np.isfinite(uv).all() or (uv<0).any() or (uv>1).any()):
        raise ValueError('Finite anatomical UVs in [0,1] required')
    return uv


def footprint(uv,config=ApertureConfig()):
    uv=validate_uv(uv)
    if (not np.isfinite(config.radius_pixels) or not 0<=config.radius_pixels<=4
            or type(config.subdivisions) is not int or config.subdivisions not in (1,3,5)):
        raise ValueError('Bounded finite static aperture required')
    offsets=np.linspace(-config.radius_pixels,config.radius_pixels,config.subdivisions) if config.subdivisions>1 else [0.]
    parts=[]
    for dy in offsets:
        for dx in offsets:
            xy=np.clip(uv*np.array([319.,179.])+[dx,dy],[0.,0.],[319.,179.])
            lower=np.floor(xy).astype(int);upper=np.minimum(lower+[1,1],[319,179])
            fraction=xy-lower
            for iy in (0,1):
                for ix in (0,1):
                    pixels=np.c_[upper[:,0] if ix else lower[:,0],upper[:,1] if iy else lower[:,1]]
                    weight=(fraction[:,0] if ix else 1-fraction[:,0])*(fraction[:,1] if iy else 1-fraction[:,1])
                    parts.append((pixels,weight/config.subdivisions**2))
    return parts


def sample_aperture_linear(rgb,uv,config=ApertureConfig()):
    rgb=np.asarray(rgb)
    if rgb.shape!=(180,320,3) or rgb.dtype!=np.uint8:
        raise ValueError('Fixed full-world RGB uint8 required')
    result=np.zeros((len(uv),3),dtype=float)
    for pixels,weight in footprint(uv,config):
        values=rgb[pixels[:,1],pixels[:,0]].astype(float)/255.
        # Interpolate radiance AFTER linearizing RGB, rather than blending
        # gamma-coded display values. All interpolation weights are positive.
        linear=np.where(values<=.04045,values/12.92,((values+.055)/1.055)**2.4)
        result+=weight[:,None]*linear
    return result


def coverage_uv(uv,config=ApertureConfig()):
    points=np.unique(np.concatenate([pixels[weight>1e-10] for pixels,weight in footprint(uv,config)]),axis=0)
    # Interior-bin coordinates avoid float roundoff when the diagnostic search
    # converts these support pixels back to integer bins. Receptor UVs are NOT
    # changed, and these coordinates never enter neural integration.
    return np.minimum(points+.25,[319.,179.])/[319.,179.]


def identity(uv,config=ApertureConfig()):
    return {'version':VERSION,'config':asdict(config),
        'original_uv_sha256':hashlib.sha256(np.ascontiguousarray(uv).tobytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'assumption':'Static finite-aperture linear light proxy; natural optical field size unvalidated',
        'head_centering':False,'original_uv_modified':False,'objects_or_labels_used':False}
