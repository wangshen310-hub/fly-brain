"""Training-only margins for the unchanged native integer-count decoder.

The input is ``[2, scenes, 4]`` actual hard-spike counts in native order
``left, right, forward_left, forward_right``. The second window must already
have been obtained by continuing the first window's complete physical state.
This module neither runs the brain nor changes its state or executed actions.
Labels enter only the supervised objective and the evaluation diagnostics.

For L/R counts, d = R-L, q = relative_margin*max(L,R,1), and S is the sum of
all four motor counts. The decision surpluses, in count units, are

    left:    -d-q
    forward: min(q-abs(d), S-0.5)
    right:    d-q.

Left/right require a strictly positive surplus. Forward accepts zero. The
half-count activity guard is exact for integer forward counts: silence has
negative forward surplus, while any nonzero count satisfies the guard. Hard
predictions nevertheless call the ORIGINAL Python decoder, so float32 margin
rounding cannot change reported actions at a tie such as L=5, R=4.

Let H(x)=relu(x)^2, a continuously differentiable squared hinge. With defaults
m=1 count and f=0.5, the per-scene training penalties are

    left/right: H((m - decision_surplus) / m)
    forward:    H((f*q - (q-abs(d))) / max(q,m)) + H(1-S).

Thus a forward target asks for the inner half of the existing tolerance band,
not an absolute L/R firing count; a balanced active pair [1,1] already has zero
penalty. A turn asks for an extra one-count surplus. This target is a training
margin, NOT a guarantee of robustness to every one-count perturbation.

Average these penalties separately for each of the six window/action groups.
The final default objective is 0.75 times the CVaR of the two worst group means
plus 0.25 times the equally weighted mean of all six. Groups tied at the CVaR
cutoff share its remaining weight equally. This removes arbitrary category
ordering from the loss subgradient, while all groups retain equal positive
mean-term weight even when their sample counts differ.

The differentiable loss preserves the input floating dtype (native counts use
float32) and relies on the caller's surrogate derivatives for hard spikes.
It is not an exact gradient of the discontinuous native count objective.
Native acceptance must recompute this same objective from a fresh complete
brain run; a lower conditional loss is not evidence of recurrent learning.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


VERSION = "native-decoder-two-window-group-margin-v1"
ACTIONS = ("left", "forward", "right")
COUNT_ORDER = ("left", "right", "forward_left", "forward_right")


def _validate_counts(counts, relative_margin):
    if (not isinstance(counts, torch.Tensor) or counts.ndim != 3
            or counts.shape[0] != 2 or counts.shape[1] < 1 or counts.shape[2] != 4
            or counts.dtype not in (torch.float32, torch.float64)):
        raise ValueError("Two windows of floating [2, scenes, 4] native motor counts required")
    if (isinstance(relative_margin, bool)
            or not isinstance(relative_margin, (float, int))
            or not math.isfinite(relative_margin) or not 0 < relative_margin < 1):
        raise ValueError("The original finite relative margin must lie strictly between 0 and 1")
    detached = counts.detach()
    if (not bool(torch.isfinite(detached).all()) or bool((detached < 0).any())
            or bool((detached > 2**53).any())
            or not torch.equal(detached, detached.round())):
        raise ValueError("Forward values must be actual finite nonnegative integer spike counts <= 2**53")


def _surplus_components(counts, relative_margin):
    left, right = counts[..., 0], counts[..., 1]
    difference = right - left
    gap = relative_margin * torch.maximum(torch.maximum(left, right), torch.ones_like(left))
    total = counts.sum(dim=-1)
    balance = gap - difference.abs()
    surpluses = torch.stack((-difference - gap,
                             torch.minimum(balance, total - 0.5),
                             difference - gap), dim=-1)
    return surpluses, gap, balance, total


def decoder_surpluses(motor_counts, *, relative_margin=0.2):
    """Return count-unit surpluses in L/F/R order, including forward silence.

    These are differentiable training values. Use ``hard_predictions`` for
    exact reported native actions, including the Python decoder's tie rule.
    """
    _validate_counts(motor_counts, relative_margin)
    return _surplus_components(motor_counts, relative_margin)[0]


def _native_predictions(counts, relative_margin):
    # Import the original decoder, never substitute a rounded tensor argmax.
    # It receives no labels, loss weights, previous action or game state.
    from .teacher_spiking import decode_spikes

    raw = counts.detach().cpu().to(torch.int64).tolist()
    predictions = []
    for window in raw:
        row = []
        for left, right, forward_left, forward_right in window:
            action = decode_spikes({"left": [left], "right": [right],
                                    "forward": [forward_left, forward_right]},
                                   margin=float(relative_margin))["action"]
            row.append(-1 if action is None else ACTIONS.index(action))
        predictions.append(row)
    return torch.tensor(predictions, dtype=torch.int64, device=counts.device), raw


def hard_predictions(motor_counts, *, relative_margin=0.2):
    """Native decoder diagnostics: 0=L, 1=F, 2=R, -1=NO_ACTION.

    This evaluation helper calls the original integer-count decoder. Its
    signature deliberately has no training labels or loss configuration.
    """
    _validate_counts(motor_counts, relative_margin)
    return _native_predictions(motor_counts, relative_margin)[0]


def _balanced_tail_weights(group_means, worst_groups):
    """Equal-mass empirical upper CVaR, with a permutation-invariant tie rule."""
    detached = group_means.detach()
    cutoff = detached.sort(descending=True).values[worst_groups - 1]
    above = detached > cutoff
    equal = detached == cutoff
    remaining = worst_groups - above.to(torch.int64).sum()
    weights = (above.to(group_means.dtype)
               + equal.to(group_means.dtype) * remaining / equal.sum()) / worst_groups
    return weights


def decoder_margin_loss(motor_counts, labels, *, relative_margin=0.2,
                        turn_margin=1.0, forward_interior_fraction=0.5,
                        mean_weight=0.25, worst_groups=2):
    """Return the six-group native-margin loss and auditable plain diagnostics.

    Every class must occur in this training/evaluation batch. A class's group
    mean has the same base weight regardless of how many scenes belong to it.
    ``labels`` is a one-dimensional int64 tensor on the counts' device.
    No data or state is mutated. The returned predictions and integer counts
    are diagnostics only and must never replace the native inference path.
    """
    _validate_counts(motor_counts, relative_margin)
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
            or labels.device != motor_counts.device
            or labels.shape != (motor_counts.shape[1],)
            or not torch.equal(labels.unique(sorted=True), labels.new_tensor([0, 1, 2]))):
        raise ValueError("Int64 scene labels on the same device must include all three actions")
    for name, value in (("turn_margin", turn_margin),
                        ("forward_interior_fraction", forward_interior_fraction),
                        ("mean_weight", mean_weight)):
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value)):
            raise ValueError(name + " must be a finite real number")
    if (turn_margin <= 0 or not 0 < forward_interior_fraction <= 1
            or not 0 < mean_weight <= 1 or type(worst_groups) is not int
            or not 1 <= worst_groups <= 6):
        raise ValueError("Positive margins/mean weight, forward fraction <=1, and 1..6 tail groups required")

    surpluses, gap, balance, total = _surplus_components(motor_counts, relative_margin)
    selected = surpluses.gather(-1, labels[None, :, None].expand(2, -1, 1)).squeeze(-1)
    turn_penalty = F.relu((turn_margin - selected) / turn_margin).square()
    forward_balance_penalty = F.relu(
        (forward_interior_fraction * gap - balance) / gap.clamp_min(turn_margin)).square()
    forward_activity_penalty = F.relu(1.0 - total).square()
    forward_penalty = forward_balance_penalty + forward_activity_penalty
    penalties = torch.where(labels[None] == 1, forward_penalty, turn_penalty)
    groups = torch.stack([penalties[window, labels == action].mean()
                          for window in range(2) for action in range(3)])
    tail_weights = _balanced_tail_weights(groups, worst_groups)
    mean = groups.mean()
    cvar = (tail_weights * groups).sum()
    loss = (1.0 - mean_weight) * cvar + mean_weight * mean
    if not bool(torch.isfinite(loss.detach())):
        raise ValueError("Decoder-margin objective is nonfinite")

    predictions, raw_counts = _native_predictions(motor_counts, relative_margin)
    recalls = torch.stack([(predictions[window, labels == action] == action)
                           .to(torch.float64).mean()
                           for window in range(2) for action in range(3)]).reshape(2, 3)
    group_mean_surplus = torch.stack([selected[window, labels == action].mean()
                                      for window in range(2) for action in range(3)]).reshape(2, 3)
    group_min_surplus = torch.stack([selected[window, labels == action].min()
                                     for window in range(2) for action in range(3)]).reshape(2, 3)
    effective_weights = (1.0 - mean_weight) * tail_weights + mean_weight / 6.0

    def rows(value):
        return value.detach().cpu().tolist()

    diagnostics = {
        "version": VERSION,
        "actions": list(ACTIONS),
        "motor_count_order": list(COUNT_ORDER),
        "loss_dtype": str(motor_counts.dtype),
        "loss": float(loss.detach()),
        "mean_group_loss": float(mean.detach()),
        "worst_group_cvar": float(cvar.detach()),
        "group_loss": rows(groups.reshape(2, 3)),
        "group_weight": rows(effective_weights.reshape(2, 3)),
        "group_sample_count": [int((labels == action).sum()) for action in range(3)],
        "group_recall": rows(recalls),
        "balanced_accuracy": rows(recalls.mean(dim=1)),
        "minimum_class_recall": float(recalls.min()),
        "minimum_window_balanced_accuracy": float(recalls.mean(dim=1).min()),
        "group_mean_decision_surplus_counts": rows(group_mean_surplus),
        "group_min_decision_surplus_counts": rows(group_min_surplus),
        "silent_rows_per_window": rows((predictions == -1).sum(dim=1)),
        "hard_predictions": rows(predictions),
        "native_motor_counts": raw_counts,
        "assigned_labels": rows(labels),
        "relative_margin": float(relative_margin),
        "turn_margin_counts": float(turn_margin),
        "forward_interior_fraction": float(forward_interior_fraction),
        "mean_weight": float(mean_weight),
        "worst_groups": worst_groups,
        "hard_metrics_source": "original teacher_spiking.decode_spikes on Python integer counts",
        "conditional_loss_establishes_native_learning": False,
    }
    return loss, diagnostics
