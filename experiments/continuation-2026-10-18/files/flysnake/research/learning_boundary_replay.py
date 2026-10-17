"""Training-only conditional dynamics of three existing relays and two motors.

The five original cells retain their native equations, thresholds, reset,
refractory clock and two-step release queue. Existing edges between them use
their newly emitted spikes. Every incoming release from outside the five cells
is a recorded, float32 release of the identified parent complete brain. This
includes graded early-visual releases; they must never be rounded to spikes.

The helper owns no trainable parameters and is not an inference policy. Its
external boundary still cuts real feedback paths. Changed parameters require
fresh complete-brain RGB evaluation; conditional fit is not learning evidence.
"""
from collections.abc import Mapping
import copy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .teacher_spiking import Dynamics, _Spike


VERSION = "recorded-external-release-five-native-cell-replay-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NODE_CELL_IDS = (19331, 10067, 10260, 523769, 10360)
STATE_FIELDS = ("v", "ge", "gi", "adaptation", "refractory")
NATIVE_NAMES = ("rest_delta", "gain_raw", "tau_raw")


def _hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _tensor(value, *, dtype, name, shape=None, nonnegative=False):
    # Copies also avoid sharing immutable NPZ buffers with the neural module.
    value = (value.detach().cpu().clone() if isinstance(value, torch.Tensor)
             else torch.from_numpy(np.array(value, copy=True)))
    if value.dtype != dtype or (shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError(name + " has a different shape or dtype from the boundary schema")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(name + " must be finite")
    if nonnegative and bool((value < 0).any()):
        raise ValueError(name + " must be nonnegative")
    return value.contiguous()


class ConditionalBoundaryReplay(nn.Module):
    """Optimize copies of existing native parameters within a fixed boundary.

    Node order is the original three ``relay_targets``, then the left and right
    turn motors. Edge parameter family 0 addresses ``afferent_log_gain``;
    family 1 addresses ``relay_edge_log_gain``. The native vector ordering is
    preserved even when the incoming edge table has another validated order.

    ``identity`` and all arrays must originate from the audited native trace
    reader. Native graph IDs/topology must additionally be bound by that reader
    and the trainer to the actual full brain before installing fitted values.
    """

    def __init__(self, *, arrays, identity, config=Dynamics()):
        super().__init__()
        if not isinstance(arrays, Mapping) or not isinstance(identity, Mapping):
            raise ValueError("Verified boundary arrays and native identity are required")
        if not isinstance(config, Dynamics) or identity.get("dynamics") != asdict(config):
            raise ValueError("Boundary dynamics must match the original candidate identity")
        scope = identity.get("learning_scope", {})
        relay_scope = scope.get("relay_edges", {})
        if (not relay_scope.get("enabled")
                or relay_scope.get("target_cell_ids") != list(NODE_CELL_IDS[:3])
                or identity.get("motor_ids", {}).get("left") != [NODE_CELL_IDS[3]]
                or identity.get("motor_ids", {}).get("right") != [NODE_CELL_IDS[4]]):
            raise ValueError("The identified existing relay and turn-motor learning scope is required")
        self.config = config
        self.reference_identity = copy.deepcopy(dict(identity))

        def read(name, dtype, shape=None, nonnegative=False, *, buffer_name=None):
            if name not in arrays:
                raise ValueError("Missing native boundary array: " + name)
            value = _tensor(arrays[name], dtype=dtype, name=name, shape=shape,
                            nonnegative=nonnegative)
            self.register_buffer(buffer_name or name, value)
            return value

        template = read("native_gain_raw", torch.float32)
        if template.ndim != 1 or template.numel() < 5:
            raise ValueError("Full-length native raw parameter templates are required")
        self.n = int(template.numel())
        read("native_tau_raw", torch.float32, (self.n,))
        read("native_rest_delta", torch.float32, (self.n,))
        for name in ("base_rest", "bias_mask", "motor_drive"):
            read("native_" + name, torch.float32, (self.n,))
        indices = read("node_indices", torch.int64, (5,))
        ids = read("node_cell_ids", torch.int64, (5,))
        if (ids.tolist() != list(NODE_CELL_IDS) or indices.unique().numel() != 5
                or bool((indices < 0).any()) or bool((indices >= self.n).any())):
            raise ValueError("Exactly the three original relays and two original motors are required")
        for name in ("graded_mask", "kc_mask", "receptor_mask"):
            value = read("node_" + name, torch.bool, (5,))
            if bool(value.any()):
                raise ValueError("All five boundary cells must be non-graded, non-KC and non-receptor")
        for name in ("base_rest", "bias_mask", "motor_drive"):
            value = read("node_" + name, torch.float32, (5,))
            if not torch.equal(value, getattr(self, "native_" + name)[indices]):
                raise ValueError("Selected native constants disagree with full-length templates: " + name)
        if (not torch.equal(self.node_bias_mask, torch.tensor([1., 1., 1., 0., 0.]))
                or not torch.equal(self.node_motor_drive, torch.tensor([0., 0., 0., .6, .6]))):
            raise ValueError("Original relay bias permissions and fixed motor drives must remain unchanged")
        for name in NATIVE_NAMES:
            allowed = read("native_" + name + "_train_mask", torch.bool, (self.n,))
            selected = indices[:3] if name == "rest_delta" else indices
            if not bool(allowed[selected].all()):
                raise ValueError("The parent candidate scope forbids a requested native entry: " + name)
            if scope.get("parameter_masks", {}).get(name) != {
                    "count": int(allowed.sum()), "mask_sha256": _hash(allowed)}:
                raise ValueError("Native learning mask differs from the candidate identity: " + name)
        self.register_buffer("baseline_relay_rest_raw", self.native_rest_delta[indices[:3]].clone())
        self.register_buffer("baseline_node_gain_raw", self.native_gain_raw[indices].clone())
        self.register_buffer("baseline_node_tau_raw", self.native_tau_raw[indices].clone())

        aff = read("baseline_afferent_raw", torch.float32)
        relay = read("baseline_relay_raw", torch.float32)
        if aff.ndim != 1 or relay.ndim != 1 or min(aff.numel(), relay.numel()) < 1:
            raise ValueError("Both complete original positive-gain parameter vectors are required")
        self.afferent_edges, self.relay_edges = int(aff.numel()), int(relay.numel())
        self.edges = self.afferent_edges + self.relay_edges
        if relay_scope.get("edges") != self.relay_edges:
            raise ValueError("The candidate relay-edge identity has a different parameter count")
        if self.n == 166700 and (self.afferent_edges, self.relay_edges) != (2319, 6157):
            raise ValueError("The complete brain must retain all 2319 motor and 6157 relay input gains")
        pre = read("edge_pre", torch.int64, (self.edges,))
        post = read("edge_post", torch.int64, (self.edges,))
        absolute = read("edge_abs", torch.float32, (self.edges,))
        sign = read("edge_exc", torch.bool, (self.edges,))
        family = read("edge_parameter_family", torch.int8, (self.edges,))
        parameter_index = read("edge_parameter_index", torch.int64, (self.edges,))
        if (bool((absolute <= 0).any()) or bool((pre < 0).any())
                or bool((pre >= self.n).any()) or not bool(torch.isin(post, indices).all())
                or not bool(((family == 0) | (family == 1)).all())):
            raise ValueError("Only existing positive-magnitude signed edges into the five cells are permitted")
        edge_slots = (post[:, None] == indices[None, :]).to(torch.int64).argmax(dim=1)
        for kind, length, targets in ((0, self.afferent_edges, indices[3:]),
                                      (1, self.relay_edges, indices[:3])):
            selected = family == kind
            if (int(selected.sum()) != length
                    or not torch.equal(parameter_index[selected].sort().values, torch.arange(length))
                    or not bool(torch.isin(post[selected], targets).all())):
                raise ValueError("Every original gain must map once to its original parameter family and target")
            # The afferent identity hashes the export's original endpoint
            # integer dtype; relay identities use explicit int64 endpoints.
            # Reconstruct each original vector order before verifying it.
            ordered = selected.nonzero(as_tuple=True)[0][parameter_index[selected].argsort()]
            signed = absolute[ordered].numpy() * np.where(sign[ordered].numpy(), 1., -1.).astype(np.float32)
            endpoint = torch.stack((pre[ordered], post[ordered]), dim=1).numpy()
            dtypes = (np.int32, np.int64) if kind == 0 else (np.int64,)
            candidates = {hashlib.sha256(endpoint.astype(dtype).tobytes() + signed.tobytes()).hexdigest()
                          for dtype in dtypes}
            declared = (identity.get("afferent_identity_sha256") if kind == 0
                        else relay_scope.get("edge_identity_sha256"))
            if declared not in candidates:
                raise ValueError("Existing edge endpoints, signs or weights differ from the native parameter identity")
        self.register_buffer("edge_node_slot", edge_slots)
        combined = parameter_index + torch.where(family == 0, 0, self.afferent_edges)
        self.register_buffer("edge_combined_parameter_index", combined)
        reference = torch.cat((torch.exp(4. * torch.tanh(aff)),
                               torch.exp(4. * torch.tanh(relay))))[combined]
        self.register_buffer("reference_edge_factor", reference)
        self.register_buffer("reference_edge_weight", absolute * reference)
        internal = torch.isin(pre, indices)
        self.register_buffer("edge_internal", internal)
        internal_slot = (pre[:, None] == indices[None, :]).to(torch.int64).argmax(dim=1)
        self.register_buffer("edge_internal_node_slot", internal_slot)

        source_indices = read("external_source_indices", torch.int64)
        expected_external = pre[~internal].unique(sorted=True)
        if source_indices.ndim != 1 or not torch.equal(source_indices, expected_external):
            raise ValueError("External sources must be exactly the sorted unique incoming cells outside the boundary")
        source_graded = read("external_source_graded_mask", torch.bool, (source_indices.numel(),))
        external = read("external_arriving", torch.float32, nonnegative=True)
        if (external.ndim != 3 or external.shape[0] != 2 * config.window_ms
                or external.shape[1] < 1 or external.shape[2] != source_indices.numel()
                or bool((external > 1.).any())):
            raise ValueError("Actual float32 external releases must span two complete native windows")
        ungraded_releases = external[:, :, ~source_graded]
        if not bool(((ungraded_releases == 0) | (ungraded_releases == 1)).all()):
            raise ValueError("Non-graded external cells must retain actual binary spikes")
        self.steps, self.batch, self.external_sources = map(int, external.shape)
        self.windows = 2
        external_slots = torch.zeros(self.edges, dtype=torch.int64)
        external_slots[~internal] = torch.searchsorted(source_indices, pre[~internal])
        self.register_buffer("edge_external_source_slot", external_slots)
        for prefix, excitatory in (("exc", True), ("inh", False)):
            read("delivered_" + prefix, torch.float32, (self.steps, self.batch, 5), nonnegative=True)
            for slot in range(5):
                group = (sign == excitatory) & (edge_slots == slot)
                for part, selected in (("external", group & ~internal), ("internal", group & internal)):
                    self.register_buffer(f"{prefix}_{slot}_{part}", selected.nonzero(as_tuple=True)[0])
        read("node_arriving", torch.float32, (self.steps, self.batch, 5), nonnegative=True)
        if not bool(((self.node_arriving == 0) | (self.node_arriving == 1)).all()):
            raise ValueError("The five non-graded cells must have binary delayed releases")
        for name in ("rest", "target_rest", "gain", "tau"):
            read(name, torch.float32, (5,))
        mapped = self._effective_intrinsics(self.baseline_relay_rest_raw,
                                            self.baseline_node_gain_raw, self.baseline_node_tau_raw)
        if any(not torch.equal(actual, getattr(self, name)) for name, actual in zip(
                ("rest", "target_rest", "gain", "tau"), mapped)):
            raise ValueError("Full-length native raw mappings do not reproduce the recorded effective parameters")
        for name in STATE_FIELDS:
            dtype = torch.int32 if name == "refractory" else torch.float32
            initial = read("initial_" + name, dtype, (self.batch, 5))
            expected = (self.node_base_rest.expand(self.batch, 5) if name == "v"
                        else torch.zeros_like(initial))
            if not torch.equal(initial, expected):
                raise ValueError("Only native cold states with fixed base_rest voltage and zero other state are supported")
            read("end_" + name, dtype, (2, self.batch, 5), buffer_name="recorded_end_" + name)
        read("initial_release_queue", torch.float32, (config.delay_steps, self.batch, 5))
        read("initial_cursor", torch.int64, ())
        read("initial_ticks", torch.int64, ())
        if (bool(self.initial_release_queue.any()) or int(self.initial_cursor) != 0
                or int(self.initial_ticks) != 0):
            raise ValueError("Only the original zero queue, cursor and tick cold boundary is supported")
        counts = read("node_counts", torch.float32, (2, self.batch, 5), nonnegative=True,
                      buffer_name="recorded_node_counts")
        spikes = read("node_spikes", torch.uint8, (self.steps, self.batch, 5),
                      buffer_name="recorded_node_spikes")
        if (bool((spikes > 1).any()) or not torch.equal(counts, counts.floor())
                or not torch.equal(spikes.reshape(2, config.window_ms, self.batch, 5)
                                   .sum(dim=1).float(), counts)):
            raise ValueError("Actual five-cell spikes must reproduce all recorded integer window counts")
        expected_arriving = torch.cat((self.initial_release_queue, spikes.float()), dim=0)[:self.steps]
        if not torch.equal(expected_arriving, self.node_arriving):
            raise ValueError("Recorded internal arrivals do not have the original native queue delay")
        queues = read("end_release_queue", torch.float32, (2, config.delay_steps, self.batch, 5),
                      buffer_name="recorded_end_release_queue")
        end_cursor = read("end_cursor", torch.int64, (2,), buffer_name="recorded_end_cursor")
        end_ticks = read("end_ticks", torch.int64, (2,), buffer_name="recorded_end_ticks")
        if (not torch.equal(end_ticks, torch.tensor([config.window_ms, 2 * config.window_ms]))
                or not torch.equal(end_cursor, end_ticks % config.delay_steps)):
            raise ValueError("The native endpoint clocks must continue through both cold-start windows")
        queue, cursor = list(self.initial_release_queue.unbind(0)), 0
        for tick, value in enumerate(spikes.float(), 1):
            queue[cursor] = value
            cursor = (cursor + 1) % config.delay_steps
            if tick % config.window_ms == 0 and not torch.equal(torch.stack(queue), queues[tick // config.window_ms - 1]):
                raise ValueError("Recorded native queue endpoints disagree with the actual spike history")
        # Production observation needs only actual arrivals, spikes and window
        # endpoints. Tiny native 1-ms tests may additionally supply all states.
        supplied = ["state_" + name in arrays for name in STATE_FIELDS]
        if any(supplied) and not all(supplied):
            raise ValueError("An optional millisecond-state reference must contain all five fields")
        self.has_step_state_reference = all(supplied)
        if self.has_step_state_reference:
            for name in STATE_FIELDS:
                read("state_" + name, torch.int32 if name == "refractory" else torch.float32,
                     (self.steps, self.batch, 5), buffer_name="recorded_step_" + name)

    @staticmethod
    def _raw(value, name, shape):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                or value.dtype != torch.float32 or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())):
            raise ValueError(name + " must contain exactly the finite CPU float32 native raw entries")

    def _effective_intrinsics(self, relay_rest_raw, node_gain_raw, node_tau_raw):
        self._raw(relay_rest_raw, "relay_rest_raw", (3,))
        self._raw(node_gain_raw, "node_gain_raw", (5,))
        self._raw(node_tau_raw, "node_tau_raw", (5,))
        r = self.native_rest_delta.index_copy(0, self.node_indices[:3], relay_rest_raw)
        g = self.native_gain_raw.index_copy(0, self.node_indices, node_gain_raw)
        t = self.native_tau_raw.index_copy(0, self.node_indices, node_tau_raw)
        # Original full-N expressions before selecting the five destinations.
        rest = (self.native_base_rest[:, None] + .4 * torch.tanh(r)[:, None]
                * self.native_bias_mask[:, None])
        target = rest + self.native_motor_drive[:, None]
        gain = (.05 + 3.95 * torch.sigmoid(g))[:, None]
        tau = (5. + 35. * torch.sigmoid(t))[:, None]
        return tuple(value[self.node_indices, 0] for value in (rest, target, gain, tau))

    def _indices(self, indices):
        if indices is None:
            return None
        value = torch.as_tensor(indices)
        if (value.dtype != torch.int64 or value.device.type != "cpu" or value.ndim != 1
                or not value.numel() or bool((value < 0).any()) or bool((value >= self.batch).any())):
            raise ValueError("Replay sample indices must be a nonempty in-range CPU int64 vector")
        return value

    def forward(self, afferent_raw, relay_raw, relay_rest_raw, node_gain_raw, node_tau_raw,
                indices=None, *, detach_windows=False, return_spikes=False, return_step_states=False):
        """Return W x B x 5 native counts and five window-end state arrays.

        Optional spikes append a T x B x 5 array. With both ``return_spikes``
        and ``return_step_states``, a fourth item contains every integration
        state and queue endpoints for the software equivalence gates.
        Detachment changes training gradients only; physical state and the
        internal delay queue always continue through both complete windows.
        """
        for name, value, length in (("afferent_raw", afferent_raw, self.afferent_edges),
                                    ("relay_raw", relay_raw, self.relay_edges)):
            self._raw(value, name, (length,))
        if (type(detach_windows) is not bool or type(return_spikes) is not bool
                or type(return_step_states) is not bool or (return_step_states and not return_spikes)):
            raise ValueError("Explicit boolean flags and spike output for step-state recording are required")
        rest, target_rest, gain, tau = self._effective_intrinsics(relay_rest_raw, node_gain_raw, node_tau_raw)
        indices = self._indices(indices)
        batch = self.batch if indices is None else int(indices.numel())
        def select(value):
            return value if indices is None else value[:, indices]
        factor = torch.cat((torch.exp(4. * torch.tanh(afferent_raw)),
                            torch.exp(4. * torch.tanh(relay_raw))))[self.edge_combined_parameter_index]
        weight = self.edge_abs * factor
        delta_weight = self.edge_abs * (factor - self.reference_edge_factor)
        external = select(self.external_arriving)
        parent_internal = select(self.node_arriving)
        delivered = {}
        for prefix in ("exc", "inh"):
            columns = []
            for slot in range(5):
                edge = getattr(self, f"{prefix}_{slot}_external")
                source = external[:, :, self.edge_external_source_slot[edge]]
                delta = source.reshape(self.steps * batch, edge.numel()).matmul(delta_weight[edge])
                columns.append(delta.reshape(self.steps, batch))
            delivered[prefix] = select(getattr(self, "delivered_" + prefix)) + torch.stack(columns, dim=2)
        values = [getattr(self, "initial_" + name) if indices is None
                  else getattr(self, "initial_" + name)[indices] for name in STATE_FIELDS]
        v, ge, gi, adaptation, refractory = values
        queue_tensor = self.initial_release_queue if indices is None else self.initial_release_queue[:, indices]
        queue, cursor = list(queue_tensor.unbind(0)), int(self.initial_cursor)
        cfg = self.config
        decay = math.exp(-cfg.dt_ms / cfg.synapse_ms)
        counts = torch.zeros_like(v)
        windows, boundary = [], {name: [] for name in STATE_FIELDS}
        spikes, step_states = [], {name: [] for name in STATE_FIELDS}
        end_queues, arriving_history = [], []
        for tick in range(self.steps):
            arriving = queue[cursor]
            if return_step_states:
                arriving_history.append(arriving)
            current = {}
            for prefix in ("exc", "inh"):
                columns = []
                for slot in range(5):
                    edge = getattr(self, f"{prefix}_{slot}_internal")
                    source = self.edge_internal_node_slot[edge]
                    # Difference first: at the exact parent this is exactly
                    # zero, retaining the original bound delivery unchanged.
                    change = (arriving[:, source] * weight[edge]
                              - parent_internal[tick, :, source] * self.reference_edge_weight[edge])
                    columns.append(change.sum(dim=1))
                current[prefix] = delivered[prefix][tick] + torch.stack(columns, dim=1)
            adaptation = adaptation * math.exp(-cfg.dt_ms / 200.)
            ge = ge * decay + gain * current["exc"]
            gi = gi * decay + gain * current["inh"]
            total = 1. + ge + gi + adaptation
            equilibrium = (target_rest + cfg.exc_reversal * ge
                           + cfg.inh_reversal * (gi + adaptation)) / total
            v = equilibrium + (v - equilibrium) * torch.exp(-cfg.dt_ms * total / tau)
            active = refractory == 0
            spike = _Spike.apply(v - cfg.threshold, cfg.derivative_width) * active
            counts = counts + spike
            if return_spikes:
                spikes.append(spike)
            reset = spike.detach().bool()
            v = torch.where(reset | ~active, rest, v)
            refractory = torch.where(reset, cfg.refractory_steps, (refractory - 1).clamp_min(0))
            queue[cursor] = spike.clone()
            cursor = (cursor + 1) % cfg.delay_steps
            current_states = (v, ge, gi, adaptation, refractory)
            if return_step_states:
                for name, value in zip(STATE_FIELDS, current_states):
                    step_states[name].append(value)
            if (tick + 1) % cfg.window_ms == 0:
                windows.append(counts)
                for name, value in zip(STATE_FIELDS, current_states):
                    boundary[name].append(value)
                if return_step_states:
                    end_queues.append(torch.stack(queue))
                counts = torch.zeros_like(v)
                if detach_windows:
                    v, ge, gi, adaptation = (value.detach() for value in (v, ge, gi, adaptation))
                    queue = [value.detach() for value in queue]
        result = (torch.stack(windows), {name: torch.stack(values) for name, values in boundary.items()})
        if return_spikes:
            result = (*result, torch.stack(spikes))
        if return_step_states:
            detail = {name: torch.stack(values) for name, values in step_states.items()}
            detail.update(node_arriving=torch.stack(arriving_history),
                          end_release_queue=torch.stack(end_queues),
                          end_cursor=torch.tensor([(window * cfg.window_ms) % cfg.delay_steps
                                                   for window in (1, 2)], dtype=torch.int64),
                          end_ticks=torch.tensor([cfg.window_ms, 2 * cfg.window_ms], dtype=torch.int64))
            result = (*result, detail)
        return result

    def native_parameter_record(self):
        """Describe the only destinations of the fitted native parameter copies."""
        full, complement, allowed = {}, {}, {}
        for name in NATIVE_NAMES:
            template = getattr(self, "native_" + name)
            mask = torch.ones(self.n, dtype=torch.bool)
            mask[self.node_indices[:3] if name == "rest_delta" else self.node_indices] = False
            full[name], complement[name] = _hash(template), _hash(template[mask])
            permission = getattr(self, "native_" + name + "_train_mask")
            allowed[name] = {"count": int(permission.sum()), "mask_sha256": _hash(permission)}
        internal = []
        for edge in self.edge_internal.nonzero(as_tuple=True)[0].tolist():
            internal.append({"pre_cell_id": int(self.node_cell_ids[self.edge_internal_node_slot[edge]]),
                "post_cell_id": int(self.node_cell_ids[self.edge_node_slot[edge]]),
                "fast_sign": "exc" if bool(self.edge_exc[edge]) else "inh",
                "base_abs_weight": float(self.edge_abs[edge]),
                "native_parameter": "afferent_log_gain" if int(self.edge_parameter_family[edge]) == 0
                                    else "relay_edge_log_gain",
                "native_parameter_index": int(self.edge_parameter_index[edge])})
        return {"version": VERSION, "source_sha256": SOURCE_SHA256,
            "node_cell_ids": self.node_cell_ids.tolist(), "node_indices": self.node_indices.tolist(),
            "native_intrinsic_scalar_parameters": 13, "afferent_scalar_parameters": self.afferent_edges,
            "relay_scalar_parameters": self.relay_edges,
            "parameter_scope": {"rest_delta_indices": self.node_indices[:3].tolist(),
                                "gain_raw_indices": self.node_indices.tolist(),
                                "tau_raw_indices": self.node_indices.tolist()},
            "full_native_vector_shape": [self.n], "full_native_template_sha256": full,
            "nonselected_native_template_sha256": complement, "candidate_allowed_masks": allowed,
            "existing_edge_arrays_sha256": {name: _hash(getattr(self, name)) for name in
                ("edge_pre", "edge_post", "edge_abs", "edge_exc", "edge_parameter_family", "edge_parameter_index")},
            "internal_edges": internal, "external_sources": self.external_sources,
            "external_graded_sources": int(self.external_source_graded_mask.sum()),
            "external_release_storage": "float32 actual native delayed releases, including graded sources",
            "native_positive_synapse_factor": "exp(4*tanh(raw))", "synapse_factor_bounds": [math.exp(-4.), math.exp(4.)],
            "intrinsic_mapping": {"rest": "base_rest + .4*tanh(raw)*original_bias_mask",
                                  "gain": ".05 + 3.95*sigmoid(raw)", "tau": "5 + 35*sigmoid(raw)"},
            "baseline_relay_rest_raw": self.baseline_relay_rest_raw.tolist(),
            "baseline_node_gain_raw": self.baseline_node_gain_raw.tolist(),
            "baseline_node_tau_raw": self.baseline_node_tau_raw.tolist(),
            "baseline_effective_rest": self.rest.tolist(), "baseline_effective_gain": self.gain.tolist(),
            "baseline_effective_tau": self.tau.tolist(),
            "cold_voltage": "Fixed native base_rest, independent of learned rest_delta",
            "learned_rest_effect": "Original equilibrium target and spike/refractory reset only",
            "full_length_intrinsic_mapping_before_node_slice": True,
            "internal_release_delay_steps": self.config.delay_steps,
            "internal_releases_recomputed_from_current_spikes": True,
            "external_parent_releases_held_fixed": True,
            "native_indices_and_edges_must_be_bound_to_graph_by_reader_and_trainer": True,
            "model_equations_changed": False, "new_endpoints_or_sign_changes": False,
            "motor_drive_threshold_decoder_changed": False,
            "recurrent_equivalence_after_parameter_updates_claimed": False,
            "inference_policy": False, "learning_success_claimed": False}

    @torch.no_grad()
    def verify_reference(self, recorded_counts=None, recorded_states=None, *, recorded_spikes=None, atol=1e-6):
        """Gate the identified parent on every spike, state endpoint and queue.

        Defaults come from the same verified arrays used for construction.
        Optional per-ms float states are checked only when explicitly present
        (typically in tiny native equivalence tests). Production does not need
        introspection of the native integrator's local variables.
        """
        if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
            raise ValueError("A finite nonnegative native reference tolerance is required")
        predicted, states, emitted, detail = self(self.baseline_afferent_raw, self.baseline_relay_raw,
            self.baseline_relay_rest_raw, self.baseline_node_gain_raw, self.baseline_node_tau_raw,
            return_spikes=True, return_step_states=True)
        counts = _tensor(self.recorded_node_counts if recorded_counts is None else recorded_counts,
                         dtype=torch.float32, name="recorded_counts", shape=predicted.shape)
        spikes = _tensor(self.recorded_node_spikes if recorded_spikes is None else recorded_spikes,
                         dtype=torch.uint8, name="recorded_spikes", shape=emitted.shape)
        if not torch.equal(counts, predicted) or not torch.equal(predicted, predicted.floor()):
            raise ValueError("Conditional parent five-cell counts differ from native recorded spikes")
        if bool((spikes > 1).any()) or not torch.equal(spikes, emitted):
            raise ValueError("Conditional parent millisecond spike timing differs from the original five cells")
        reference_states = ({name: getattr(self, "recorded_end_" + name) for name in STATE_FIELDS}
                            if recorded_states is None else recorded_states)
        if not isinstance(reference_states, Mapping) or set(reference_states) != set(STATE_FIELDS):
            raise ValueError("All five native window-end state fields are required")
        errors = {}
        for name in STATE_FIELDS:
            actual = states[name]
            expected = _tensor(reference_states[name], dtype=actual.dtype, name="recorded_end_" + name,
                               shape=actual.shape)
            error = float((actual - expected).abs().max())
            if error > (0. if name == "refractory" else atol):
                raise ValueError("Conditional parent native window state differs: " + name)
            errors[name] = error
        if (not torch.equal(detail["node_arriving"], self.node_arriving)
                or not torch.equal(detail["end_release_queue"], self.recorded_end_release_queue)
                or not torch.equal(detail["end_cursor"], self.recorded_end_cursor)
                or not torch.equal(detail["end_ticks"], self.recorded_end_ticks)):
            raise ValueError("Conditional parent native internal queue or delayed arrivals differ")
        step_errors = None
        if self.has_step_state_reference:
            step_errors = {}
            for name in STATE_FIELDS:
                error = float((detail[name] - getattr(self, "recorded_step_" + name)).abs().max())
                if error > (0. if name == "refractory" else atol):
                    raise ValueError("Conditional parent millisecond native state differs: " + name)
                step_errors[name] = error
        return {"version": VERSION, "source_sha256": SOURCE_SHA256,
            "parent_spikes_exact": True, "parent_millisecond_spike_timing_exact": True,
            "parent_internal_arrivals_exact": True, "parent_internal_queue_endpoints_exact": True,
            "float_state_atol": float(atol), "native_boundary_max_abs_error": errors,
            "parent_millisecond_float_states_verified": self.has_step_state_reference,
            "native_millisecond_max_abs_error": step_errors,
            "windows": 2, "samples": self.batch, "native_parameters": self.native_parameter_record(),
            "recurrent_equivalence_after_parameter_updates_claimed": False,
            "learning_success_claimed": False}
