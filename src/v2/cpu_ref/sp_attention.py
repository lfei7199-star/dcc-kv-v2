"""2A: sequence-parallel causal attention reference + structural accounting.

Layout follows the frozen Stage-2 definition:
  L = 130944, world_size = 4, L_per_rank = 32736, causal, prefill-only.
At reference scale we keep L small (tests use L=64) but the partition /
ownership / remote-block logic is identical.

Ownership contract (2A-01 / 2A-02 / 2A-03):
  - rank r *owns* exactly its own contiguous span of K/V   -> resident
  - rank r *stages* remote spans 0..r-1 (causal: later invisible) -> transient
  - rank r must NOT hold a full copy of K/V
So remote block counts are 0/1/2/3 for ranks 0/1/2/3 under world_size=4.

resident and staged bytes are reported separately and never summed into one
number: the frozen doc requires "owner resident KV" and "remote staging KV" to
be accounted independently, and communication payload must not stand in for
resident tensor bytes.
"""

import math

from . import DTYPE_BYTES
from .state import block_state, merge_all


def partition_bounds(L, world_size):
    """Contiguous equal split; a remainder is spread over the last ranks."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    base, rem = divmod(L, world_size)
    bounds, start = [], 0
    for r in range(world_size):
        size = base + (1 if r >= world_size - rem else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _scores(q_rows, k_block, scale):
    return [[sum(q[d] * k[d] for d in range(len(q))) * scale for k in k_block] for q in q_rows]


def dense_attention(q, k, v, causal=True):
    """Single-device reference output for the same q/k/v."""
    L = len(q)
    scale = 1.0 / math.sqrt(len(q[0]))
    s = _scores(q, k, scale)
    visible = [[(not causal) or (j <= i) for j in range(L)] for i in range(L)]
    return block_state(s, v, visible).output()


def sp_forward(q, k, v, world_size, causal=True, all_gather=False, merge_order=None):
    """Real sequence-parallel prefill over `world_size` ranks.

    all_gather=True builds the *negative arm*: every rank keeps the full K/V.
    Producing it is this function's job; rejecting it is accept.py's job.
    """
    L = len(q)
    dv = len(v[0]) if v else 0
    dk = len(q[0])
    scale = 1.0 / math.sqrt(dk)
    bounds = partition_bounds(L, world_size)

    out = [[0.0] * dv for _ in range(L)]
    accounting = []

    for r, (s_r, e_r) in enumerate(bounds):
        q_rows = q[s_r:e_r]
        n_local = e_r - s_r

        if all_gather:
            own_span = (0, L)
            blocks = [(0, L)]
        else:
            own_span = (s_r, e_r)
            blocks = [bounds[i] for i in range(r)] + [(s_r, e_r)]

        states = []
        staged_numel = 0
        for (bs, be) in blocks:
            block_k, block_v = k[bs:be], v[bs:be]
            is_own = (bs == own_span[0] and be == own_span[1])
            if not is_own:
                staged_numel += (be - bs) * (dk + dv)
            sc = _scores(q_rows, block_k, scale)
            visible = []
            for i in range(n_local):
                gi = s_r + i
                if causal and is_own:
                    visible.append([(bs + j) <= gi for j in range(be - bs)])
                else:
                    visible.append([True] * (be - bs))
            states.append(block_state(sc, block_v, visible))

        if merge_order is not None:
            perm = list(merge_order)
            if sorted(perm) != list(range(len(states))):
                raise ValueError("merge_order must be a permutation of block indices")
            states = [states[i] for i in perm]

        rows = merge_all(states).output()
        for i, row in enumerate(rows):
            out[s_r + i] = row

        own_len = own_span[1] - own_span[0]
        resident_numel = own_len * (dk + dv)
        accounting.append({
            "rank": r,
            "own_span": [own_span[0], own_span[1]],
            "own_len": own_len,
            "resident_kv_numel": resident_numel,
            "resident_kv_bytes": resident_numel * DTYPE_BYTES,
            "staged_kv_numel": staged_numel,
            "staged_kv_bytes": staged_numel * DTYPE_BYTES,
            "remote_block_count": 0 if all_gather else r,
            "source_blocks": [[bs, be] for (bs, be) in blocks],
            "backend": "cpu_ref_f64",
            "dtype_bytes": DTYPE_BYTES,
        })

    return {"output": out, "accounting": accounting, "world_size": world_size, "L": L}


def full_kv_bytes(L, dk, dv):
    """Denominator for the shard-ratio check: L * (dk + dv) * element_size."""
    return L * (dk + dv) * DTYPE_BYTES
