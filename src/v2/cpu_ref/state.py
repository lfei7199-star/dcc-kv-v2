"""Order-independent attention state (m, l, acc) with online-softmax merge.

This is the numeric core both 2A (SP correctness) and 2D (per-edge merge) rely
on. Everything is plain Python floats so the module has zero dependencies.

Conventions
-----------
m   : (nq,)        running max of masked logits, per query row
l   : (nq,)        running sum of exp(score - m)
acc : (nq, dv)     running sum of exp(score - m) * v

merge(a, b) must be associative and commutative up to floating point; the
2A order-independence test asserts that over all permutations.
"""

import math

NEG_INF = float("-inf")


class AttentionState:
    __slots__ = ("m", "l", "acc", "nq", "dv")

    def __init__(self, m, l, acc):
        self.m = m
        self.l = l
        self.acc = acc
        self.nq = len(m)
        self.dv = len(acc[0]) if acc and acc[0] else 0

    @classmethod
    def empty(cls, nq, dv):
        return cls([NEG_INF] * nq, [0.0] * nq, [[0.0] * dv for _ in range(nq)])

    def output(self):
        """Normalized attention output. Rows that saw no key return zeros."""
        out = []
        for i in range(self.nq):
            li = self.l[i]
            if li == 0.0:
                out.append([0.0] * self.dv)
            else:
                inv = 1.0 / li
                out.append([x * inv for x in self.acc[i]])
        return out


def merge(a, b):
    """Merge two partial states covering disjoint key sets."""
    if a.nq != b.nq or a.dv != b.dv:
        raise ValueError("state shape mismatch: %r vs %r" % ((a.nq, a.dv), (b.nq, b.dv)))

    out_m = [0.0] * a.nq
    out_l = [0.0] * a.nq
    out_acc = [[0.0] * a.dv for _ in range(a.nq)]

    for i in range(a.nq):
        ma, mb = a.m[i], b.m[i]
        m = ma if ma >= mb else mb
        if m == NEG_INF:
            out_m[i] = NEG_INF
            out_l[i] = 0.0
            continue
        sa = math.exp(ma - m) if ma != NEG_INF else 0.0
        sb = math.exp(mb - m) if mb != NEG_INF else 0.0
        out_m[i] = m
        out_l[i] = a.l[i] * sa + b.l[i] * sb
        row = out_acc[i]
        ra, rb = a.acc[i], b.acc[i]
        for d in range(a.dv):
            row[d] = ra[d] * sa + rb[d] * sb
    return AttentionState(out_m, out_l, out_acc)


def block_state(scores, values, visible):
    """Build a state from one source block.

    scores  : (nq, nk) raw logits q.k / sqrt(d), already including any beta bias
    values  : (nk, dv)
    visible : (nq, nk) booleans; False == masked to -inf
    """
    nq = len(scores)
    nk = len(scores[0]) if nq else 0
    dv = len(values[0]) if values else 0

    m = [NEG_INF] * nq
    for i in range(nq):
        best = NEG_INF
        for j in range(nk):
            if visible[i][j] and scores[i][j] > best:
                best = scores[i][j]
        m[i] = best

    l = [0.0] * nq
    acc = [[0.0] * dv for _ in range(nq)]
    for i in range(nq):
        if m[i] == NEG_INF:
            continue
        li = 0.0
        row = acc[i]
        for j in range(nk):
            if not visible[i][j]:
                continue
            w = math.exp(scores[i][j] - m[i])
            li += w
            vj = values[j]
            for d in range(dv):
                row[d] += w * vj[d]
        l[i] = li
    return AttentionState(m, l, acc)


def merge_all(states):
    """Merge a list of states left-to-right. Order must not matter."""
    if not states:
        raise ValueError("no states to merge")
    acc = states[0]
    for s in states[1:]:
        acc = merge(acc, s)
    return acc
