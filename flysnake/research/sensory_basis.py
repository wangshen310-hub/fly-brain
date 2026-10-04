"""Reward-searchable native motor synapses in a label-free sensory subspace.

Principal components only choose STATIC perturbation directions for existing
synapses. They never encode a scene at inference or output a game action.
Inference is the full native spiking connectome with RGB and two odor samples.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .paired import NativeSearchBrain, digest
from .projection import CoveredGame, coverage_geometry
from .olfaction import OdorConfig, native_orn_groups, receptor_currents, receptor_identity

VERSION = 'native-sensory-subspace-actual-reward-search-v1'


@dataclass(frozen=True)
class BasisConfig:
    rank: int = 4
    gain_bound: float = 2.
    coefficient_bound: float = 3.
    mean_current_limit: float = 3.
    target_component_mV: float = 2.


def activity_basis(counts, base, sides, rank=4, target_mV=2., duration=300.):
    """No action/reward labels: SVD of measured native afferent count changes."""
    counts, base, sides = np.asarray(counts, float), np.asarray(base), np.asarray(sides)
    if (counts.ndim!=2 or counts.shape[1]!=len(base) or sides.shape!=base.shape
            or len(counts)<16 or rank<1 or rank>=len(counts) or target_mV<=0 or duration<=0
            or not np.isfinite(counts).all() or (counts<0).any()
            or not np.isfinite(base).all() or (base==0).any()
            or not np.isin(sides, [0,1]).all()):
        raise ValueError('Finite measured afferent activity and native signed edges required')
    mean = counts.mean(0)
    centered = counts-mean
    # Economical SVD: the number of measured scenes, not the full brain size,
    # bounds this static calibration. The actor never evaluates this matrix.
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    vectors = vectors[:rank].copy()
    for vector in vectors:
        if vector[np.argmax(abs(vector))]<0:
            vector *= -1
    basis = np.zeros((len(base), 2*(rank+1)))
    for side in (0,1):
        mask = sides==side
        # Bias proposal also adjusts only existing weights, not decoder or
        # sensory currents. Mean compensation applies uniformly to all edges.
        basis[mask, side*(rank+1)] = np.sign(base[mask])
        for component, vector in enumerate(vectors):
            direction = np.sign(base)*vector
            direction[~mask] = 0
            sensitivity = float(np.std(centered@(base*direction)*5./duration))
            direction *= target_mV/max(sensitivity, .1)
            basis[:,side*(rank+1)+component+1] = np.clip(direction, -1., 1.)
    return basis, mean, singular.tolist()


def choose_reward_candidate(parameters, returns, *, arm, shuffle_seed):
    """Accept a strictly better actual return, or its deliberately wrong control.

    No preferred motor action is received. This is a bounded reward-only
    engineering optimizer, not a reconstructed natural fly learning rule.
    """
    parameters = np.asarray(parameters, dtype=float)
    values = np.asarray(returns, dtype=float)
    if (parameters.ndim!=2 or parameters.shape[0]!=3 or values.shape!=(3,)
            or not np.isfinite(parameters).all() or not np.isfinite(values).all()
            or arm not in ('learning','frozen','shuffled')):
        raise ValueError('Incumbent and two matched finite native candidate returns required')
    assigned = values.copy()
    if arm=='shuffled':
        np.random.default_rng(shuffle_seed).shuffle(assigned)
    winner = int(np.argmax(assigned))
    if arm=='frozen' or assigned[winner]<=assigned[0]:
        winner = 0
    return parameters[winner].copy(), {'winner': winner, 'actual_returns': values.tolist(),
        'assigned_returns': assigned.tolist(), 'feedback_permuted': arm=='shuffled',
        'accepted': winner!=0, 'step_rms': float(np.sqrt(np.mean((parameters[winner]-parameters[0])**2)))}


class NativeSensoryBasisBrain:
    def __init__(self, *, size=4, config=BasisConfig(), odor=OdorConfig()):
        if (type(config.rank) is not int or not 1<=config.rank<=12
                or not np.isfinite(list(asdict(config).values())).all()
                or min(asdict(config).values())<=0):
            raise ValueError('Positive finite native subspace configuration required')
        native = NativeSearchBrain()
        self.brain = native.brain
        b = self.brain.b
        self.config, self.odor = config, odor
        self.geometry = coverage_geometry(b.uv, size)
        self.groups = native_orn_groups(self.brain, odor.receptor_class)
        self.base_tonic = b.tonic.copy()
        motors = np.r_[self.brain.native.groups['left'],self.brain.native.groups['right']]
        self.edges = native.edges[np.isin(b.post[native.edges],motors)].copy()
        self.pre = np.searchsorted(b.ptr,self.edges,side='right')-1
        self.post = b.post[self.edges].copy()
        self.sides = np.isin(self.post,self.brain.native.groups['right']).astype(int)
        self.base = b.weight[self.edges].copy()
        self.basis = self.mean_counts = None
        self.coefficients = np.zeros(2*(config.rank+1))
        self.correction = np.zeros(b.n)
        self.concentration = np.zeros(2)
        # Independent neutral calibration for the fixed display and dark odor.
        # No food, feedback or task labels enter the decoder calibration.
        neutral = []
        for seed in range(901,909):
            game = CoveredGame(seed=seed,size=size,geometry=self.geometry)
            game.food = None
            neutral.append(game.render_observation())
        self.brain.calibrated = False
        self.set_sensation([0.,0.])
        self.brain.calibrate(neutral)
        self.identity = {'version': VERSION, 'neurons': b.n, 'graph_edges': len(b.post),
            'graph_sha256': native.identity['graph_sha256'],
            'mutable_edges': len(self.edges),'dimensions':len(self.coefficients),
            'edges_sha256':digest(self.edges),'base_sha256':digest(self.base),
            'selected_cells':self.brain.native.ids,'calibration':self.brain.calibration,
            'geometry':self.geometry,'operating':native.operating,
            'config':asdict(config),'olfaction':receptor_identity(self.brain,self.groups,odor),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_source_signature':self.brain.source_signature,
            'topology_modified':False,'external_policy':False,'basis_fitted':False,
            'assumption':'Static label-free native activity subspace and reward-only synaptic optimization; not established in-vivo plasticity'}

    def set_sensation(self, concentration):
        pulses = receptor_currents(self.brain,self.groups,concentration,self.odor)
        self.brain.b.tonic[:] = self.base_tonic+self.correction
        for indices, currents in pulses:
            self.brain.b.tonic[indices] += currents
        self.concentration = np.asarray(concentration,dtype=float).copy()

    def observe(self, rgb, concentration):
        self.set_sensation(concentration)
        receipt = self.brain.observe(rgb,task='snake',learning=False,reset=False)
        receipt.pop('afferent_counts',None)
        receipt.update(memory_sha256=self.weight_hash(),
                       olfactory_concentration=self.concentration.tolist(),
                       olfactory_receptor_spikes={side:int(self.brain.last['counts'][ix].sum())
                                                 for side,ix in self.groups.items()},
                       homeostatic_current_sha256=digest(self.correction))
        return receipt

    def feedback(self, kind):
        before = self.weight_hash()
        receipt = self.brain.feedback(kind,learning=False)
        if receipt['delivered_spikes']<=0 or self.weight_hash()!=before:
            raise RuntimeError('Native feedback delivery or frozen rollout failed')
        return receipt

    def reset_trial(self):
        self.set_sensation([0.,0.])
        self.brain.reset_trial()

    def weight_hash(self):
        return digest(self.brain.b.weight[self.edges])

    def fit_basis(self, sensory_frames):
        if self.basis is not None or np.any(self.coefficients):
            raise ValueError('Static native activity basis is locked before rewards')
        samples, hashes = [], []
        for rgb, concentration in sensory_frames:
            self.reset_trial()
            receipt = self.observe(rgb,concentration)
            samples.append(self.brain.last['counts'][self.pre].copy())
            hashes.append({'rgb':receipt['observation_sha256'],
                           'odor':receipt['olfactory_concentration']})
        basis, mean, singular = activity_basis(samples,self.base,self.sides,
            self.config.rank,self.config.target_component_mV,self.brain.config.observation_ms)
        self._accept_basis(basis,mean,hashes,singular)
        self.set_parameters(np.zeros_like(self.coefficients))
        return {'frames':len(samples),'images':len({h['rgb'] for h in hashes}),
                'variable_afferents':int(np.count_nonzero(np.std(samples,axis=0))),
                'labels':False,'reward':False,'singular_values':singular}

    def _accept_basis(self,basis,mean,hashes,singular):
        basis,mean = np.asarray(basis,float),np.asarray(mean,float)
        if (basis.shape!=(len(self.edges),len(self.coefficients)) or mean.shape!=self.base.shape
                or not np.isfinite(basis).all() or not np.isfinite(mean).all()
                or (mean<0).any() or len(hashes)<16 or (abs(basis)>1.000001).any()):
            raise ValueError('Valid measured native subspace required')
        self.basis,self.mean_counts = basis.copy(),mean.copy()
        self.identity.update(basis_fitted=True,basis_sha256=digest(basis),
            mean_counts_sha256=digest(mean),baseline_inputs=hashes,singular_values=singular)

    def set_parameters(self,coefficients):
        coefficients = np.asarray(coefficients,dtype=float)
        if (self.basis is None or coefficients.shape!=self.coefficients.shape
                or not np.isfinite(coefficients).all()
                or (abs(coefficients)>self.config.coefficient_bound+1e-9).any()):
            raise ValueError('Valid bounded synaptic proposal in frozen subspace required')
        gains = np.clip(self.basis@coefficients,-self.config.gain_bound,self.config.gain_bound)
        weights = (self.base*np.exp(gains)).astype(self.base.dtype)
        if not np.array_equal(np.sign(weights),np.sign(self.base)):
            raise RuntimeError('Native synaptic sign changed')
        self.brain.b.weight[self.edges] = weights
        self.coefficients = coefficients.copy()
        correction = np.zeros(self.brain.b.n)
        np.add.at(correction,self.post,-(weights-self.base)*self.mean_counts*5./self.brain.config.observation_ms)
        self.correction = np.clip(correction,-self.config.mean_current_limit,self.config.mean_current_limit)
        self.set_sensation(self.concentration)

    def save_basis(self,path):
        if self.basis is None:
            raise ValueError('Native activity basis is not fitted')
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream,basis=self.basis,mean=self.mean_counts,
                identity=json.dumps(self.identity,sort_keys=True))
            stream.flush();os.fsync(stream.fileno())
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def restore_basis(self,path,expected_sha):
        if self.basis is not None or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError('Native subspace hash mismatch or already fitted')
        with np.load(path,allow_pickle=False) as z:
            identity = json.loads(str(z['identity']))
            # Before accepting any arrays verify all immutable graph, sensory,
            # operating-point, calibration and source identities.
            for key,value in self.identity.items():
                if key=='basis_fitted':continue
                if identity.get(key)!=json.loads(json.dumps(value)):
                    raise ValueError('Incompatible native activity basis: '+key)
            if digest(z['basis'])!=identity['basis_sha256'] or digest(z['mean'])!=identity['mean_counts_sha256']:
                raise ValueError('Native subspace internal hash mismatch')
            self._accept_basis(z['basis'],z['mean'],identity['baseline_inputs'],identity['singular_values'])
        self.set_parameters(np.zeros_like(self.coefficients))
