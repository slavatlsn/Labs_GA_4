from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from scheduling import Case, decode, check


@dataclass(frozen=True)
class Candidate:
    order: tuple[int, ...]
    machine: tuple[int, ...]
    f: tuple[int, int, int]


def dominates(a, b):
    return all(x <= y for x,y in zip(a,b)) and a != b


def rank_crowd(pop):
    f = np.asarray([x.f for x in pop])
    dom = np.all(f[:,None] <= f[None,:],axis=2) & np.any(f[:,None] < f[None,:],axis=2)
    counts = dom.sum(axis=0)
    front = np.flatnonzero(counts == 0)
    rank = np.empty(len(pop), dtype=int)
    crowd = np.zeros(len(pop))
    k = 0
    while len(front):
        rank[front] = k
        for col in f.T:
            ids = front[np.argsort(col[front], kind="stable")]
            extent = col[ids[-1]] - col[ids[0]]
            if extent:
                crowd[ids[[0,-1]]] = np.inf
                crowd[ids[1:-1]] += (col[ids[2:]]-col[ids[:-2]]) / extent
        if len(front) <= 2:
            crowd[front] = np.inf
        counts[front] = -1
        counts -= dom[front].sum(axis=0)
        front = np.flatnonzero(counts == 0)
        k += 1
    return rank,crowd


def front(pop):
    unique = list({c.f:c for c in pop}.values())
    f = np.asarray([x.f for x in unique])
    dom = np.all(f[:,None] <= f[None,:],axis=2) & np.any(f[:,None] < f[None,:],axis=2)
    return sorted((c for c, bad in zip(unique,dom.any(axis=0)) if not bad),key=lambda c:c.f)


def hypervolume(pts, ref):
    # Точный 3D sweep после нормировки. Все направления — минимум.
    f = sorted({tuple(x/r for x,r in zip(t,ref)) for t in pts})
    if not f:
        return 0.
    if any(not 0 <= x < 1 for row in f for x in row):
        raise ValueError("Точка вне общего референсного бокса")
    xs = sorted({t[0] for t in f})+[1.]
    area = 0.
    for lo,hi in zip(xs,xs[1:]):
        yz = sorted((y,z) for x,y,z in f if x <= lo)
        prev_y, best_z, volume = 0.,1.,0.
        for y,z in yz:
            volume += (y-prev_y)*(1-best_z)
            prev_y,best_z = y,min(z,best_z)
        area += (hi-lo)*(volume+(1-prev_y)*(1-best_z))
    return area


def ox(a, b, rng):
    lo,hi = sorted(rng.sample(range(len(a)+1), 2))
    copied = set(a[lo:hi])
    rotated = b[hi:]+b[:hi]
    rest = tuple(i for i in rotated if i not in copied)
    tail = len(a)-hi
    return rest[tail:]+a[lo:hi]+rest[:tail]


def solve(case: Case, seed: int, budget: int, size: int, local: bool):
    n = len(case.ops)
    if n < 2 or size < 8 or budget < 2*size or budget % size:
        raise ValueError("Нужны ≥2 операции, популяция ≥8 и бюджет, кратный популяции (минимум две популяции)")
    rng = random.Random(seed)
    count = feasible = attempts = accepted = 0

    def evaluate(order,machine):
        nonlocal count,feasible
        f, rows = decode(case, order, machine)
        count += 1
        if check(case,rows) != f:
            raise AssertionError("Внешняя проверка не подтвердила цели")
        feasible += 1
        return Candidate(tuple(order),tuple(machine),f)

    pop = [evaluate(rng.sample(range(n), n),[rng.choice(rs) for rs in case.eligible]) for _ in range(size)]
    hist = []
    while True:
        hist.append((count,hypervolume((c.f for c in front(pop)),case.reference)))
        if count == budget:
            break
        rank,crowd = rank_crowd(pop)

        def parent():
            a,b = rng.sample(range(size),2)
            return pop[min((a,b),key=lambda i: (rank[i],-crowd[i]))]

        moves = size//4 if local else 0
        kids = []
        for _ in range(size-moves):
            a,b = parent(),parent()
            if rng.random() < .9:
                order = list(ox(a.order,b.order,rng))
                machine = [x if rng.random() < .5 else y for x,y in zip(a.machine,b.machine)]
            else:
                order,machine = list(a.order),list(a.machine)
            if rng.random() < .35:
                i,j = rng.sample(range(n),2)
                order[i],order[j] = order[j],order[i]
            if rng.random() < .35:
                i = rng.randrange(n)
                machine[i] = rng.choice(case.eligible[i])
            kids.append(evaluate(order,machine))
        for i in rng.sample(range(len(kids)),moves):
            c = kids[i]
            order,machine = list(c.order),list(c.machine)
            flexible = [j for j,opt in enumerate(case.eligible) if len(opt)>1]
            if flexible and rng.random() < .5:
                j = rng.choice(flexible)
                machine[j] = rng.choice([r for r in case.eligible[j] if r != machine[j]])
            else:
                a,b = rng.sample(range(n),2)
                order.insert(b,order.pop(a))
            v = evaluate(order,machine)
            attempts += 1
            if dominates(v.f,c.f):
                kids[i] = v
                accepted += 1
        union = pop+kids
        # На равном ранге и расстоянии сначала пропускаем уникальные векторы целей.
        rank,crowd = rank_crowd(union)
        ids = sorted(range(len(union)),key=lambda i: (rank[i],-crowd[i]))
        seen, distinct, repeated = set(),[],[]
        for i in ids:
            c = union[i]
            (repeated if c.f in seen else distinct).append(i)
            seen.add(c.f)
        pos = {i:k for k,i in enumerate(ids)}
        use = sorted(distinct+repeated,key=lambda i:(rank[i],i not in distinct,pos[i]))[:size]
        pop = [union[i] for i in use]
    return front(pop),hist,{"evaluations":count,"valid":feasible,"attempts":attempts,"accepted":accepted}
