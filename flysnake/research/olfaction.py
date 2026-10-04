"""Physical odor samples mapped to annotated native sensory cells.

This is an engineering receptor proxy, not measured fly olfactory physiology.
There is no desired action, collision mask, reward or path in the neural input.
"""
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
from flysnake.game import DIRECTIONS
from .vision import RetinaConfig, light_to_current
from .paired import digest

VERSION = 'native-bilateral-odor-receptor-proxy-v1'


@dataclass(frozen=True)
class OdorConfig:
    decay_cells: float = 4.
    antenna_forward: float = .4
    antenna_separation: float = .8
    receptor_class: str = 'ORN_DM1'
    dark_hz: float = 5.
    maximum_hz: float = 60.


def odor_samples(head, heading, food, config=OdorConfig()):
    """Sample a stationary isotropic food field at two physical antenna tips.

    Coordinates belong to the environment only. Rotating the body changes the
    antenna locations, not the fixed camera or odor source. No steering rule.
    """
    if (type(heading) is not int or heading not in range(4)
            or not np.isfinite([*head, config.decay_cells, config.antenna_forward,
                                config.antenna_separation]).all()
            or config.decay_cells <= 0 or config.antenna_separation <= 0):
        raise ValueError('Valid geometry and positive physical odor scales required')
    forward = np.array(DIRECTIONS[heading], dtype=float)
    left = np.array([forward[1], -forward[0]])
    center = np.asarray(head, dtype=float)+config.antenna_forward*forward
    tips = np.array([center+config.antenna_separation/2*left,
                     center-config.antenna_separation/2*left])
    if food is None:
        concentrations = np.zeros(2)
    else:
        food = np.asarray(food, dtype=float)
        if food.shape != (2,) or not np.isfinite(food).all():
            raise ValueError('Valid stationary odor source required')
        concentrations = np.exp(-np.linalg.norm(tips-food, axis=1)/config.decay_cells)
    return {'concentration': concentrations.tolist(), 'antenna_positions': tips.tolist(),
            'field': 'exp(-Euclidean distance / decay_cells)', 'config': asdict(config)}


def native_orn_groups(brain, receptor_class='ORN_DM1'):
    """Join real IDs and anatomical rootSide; unknown sides are excluded."""
    data = brain.native.data
    neurons = feather.read_table(data/'normalized/neurons.feather',
        columns=['source_id', 'cell_type']).to_pandas()
    annotations = feather.read_table(data/'annotations.feather',
        columns=['bodyId', 'rootSide']).to_pandas()
    selected = neurons[neurons.cell_type == receptor_class].merge(
        annotations, left_on='source_id', right_on='bodyId', how='inner', validate='one_to_one')
    groups = {}
    b = brain.b
    for side in ('L', 'R'):
        ids = selected.loc[selected.rootSide == side, 'source_id'].to_numpy(dtype=np.int64)
        indices = np.flatnonzero(np.isin(b.ids, ids))
        if not len(indices) or len(indices) != len(ids):
            raise ValueError('Annotated native receptor identity missing: '+receptor_class+' '+side)
        if np.intersect1d(indices, np.r_[b.retina, b.r8]).size:
            raise ValueError('Olfactory cells overlap visual receptors')
        groups[side] = indices
    return groups


def receptor_currents(brain, groups, concentration, config=OdorConfig()):
    concentration = np.asarray(concentration, dtype=float)
    if (concentration.shape != (2,) or not np.isfinite(concentration).all()
            or (concentration < 0).any() or (concentration > 1).any()):
        raise ValueError('Two physical concentrations in [0,1] required')
    proxy = RetinaConfig(dark_hz=config.dark_hz, white_hz=config.maximum_hz)
    return [(groups[side], light_to_current(value, -45-brain.b.rest[groups[side]], proxy))
            for side, value in zip(('L', 'R'), concentration)]


def receptor_identity(brain, groups, config=OdorConfig()):
    return {'version': VERSION, 'config': asdict(config),
            'cells': {side: brain.b.ids[ix].tolist() for side, ix in groups.items()},
            'indices_sha256': {side: digest(ix) for side, ix in groups.items()},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'assumption': 'Engineered virtual odor and LIF sensory rate; natural odor tuning and current amplitude unvalidated'}
