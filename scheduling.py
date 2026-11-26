from __future__ import annotations

from dataclasses import dataclass
from heapq import heapify, heappop, heappush
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Case:
    id: str
    machines: tuple[str, ...]
    downtime: tuple[tuple[tuple[int, int], ...], ...]
    jobs: tuple[dict, ...]
    ops: tuple[dict, ...]
    pred: tuple[tuple[int, ...], ...]
    succ: tuple[tuple[int, ...], ...]
    eligible: tuple[tuple[int, ...], ...]
    duration: tuple[dict[int, int], ...]
    setup: tuple[tuple[int, ...], ...]
    reference: tuple[int, int, int]


def read_case(path: Path) -> Case:
    d = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(r["name"] for r in d["resources"])
    down = tuple(tuple(sorted(tuple(w) for w in r["downtime"])) for r in d["resources"])
    jobs, ops = tuple(d["jobs"]), tuple(d["operations"])
    setup = tuple(tuple(row) for row in d["setup"])
    n, m = len(ops), len(names)
    if not (n and m and jobs and setup) or len(set(names)) != m:
        raise ValueError("Нужны операции, задания и неповторяющиеся ресурсы")
    if any(type(a) is not int or type(b) is not int or not 0 <= a < b
           for windows in down for a, b in windows) or any(
            b > c for windows in down for (_, b), (c, _) in zip(windows, windows[1:])):
        raise ValueError("Некорректные интервалы недоступности")
    if any(len(row) != len(setup) or any(type(x) is not int or x < 0 for x in row)
           or row[i] != 0 for i, row in enumerate(setup)):
        raise ValueError("Некорректная матрица переналадок")
    for i, job in enumerate(jobs):
        if any(type(job[k]) is not int for k in ("release", "due", "weight")) or not (
                0 <= job["release"] <= job["due"] and job["weight"] > 0):
            raise ValueError(f"Некорректное задание {i}")
    pred = tuple(tuple(op["pred"]) for op in ops)
    succ = [[] for _ in ops]
    for i, row in enumerate(pred):
        if len(set(row)) != len(row) or any(type(p) is not int or not 0 <= p < n for p in row):
            raise ValueError(f"Некорректные зависимости O{i}")
        for p in row:
            succ[p].append(i)
    deg = list(map(len, pred))
    ready = [i for i, count in enumerate(deg) if not count]
    visited = 0
    while ready:
        visited += 1
        for k in succ[ready.pop()]:
            deg[k] -= 1
            if deg[k] == 0:
                ready.append(k)
    if visited != n:
        raise ValueError("Цикл зависимостей")
    durations = tuple({int(k): v for k, v in op["duration"].items()} for op in ops)
    for i, (op, times) in enumerate(zip(ops, durations)):
        if (type(op["job"]) is not int or not 0 <= op["job"] < len(jobs)
                or type(op["family"]) is not int or not 0 <= op["family"] < len(setup) or not times) or any(
                not 0 <= k < m or type(v) is not int or v <= 0 for k, v in times.items()):
            raise ValueError(f"Некорректная операция O{i}")
    if {op["job"] for op in ops} != set(range(len(jobs))):
        raise ValueError("Каждому заданию нужна операция")
    max_setup = max(map(max, setup))
    bound = max([0, *(j["release"] for j in jobs), *(b for ws in down for _, b in ws)])
    bound += sum(max(a.values()) + max_setup for a in durations)
    ref = (bound + 1, 1 + sum(j["weight"] * max(0, bound - j["due"]) for j in jobs), 1 + n * max_setup)
    return Case(d["id"], names, down, jobs, ops, pred, tuple(map(tuple,succ)),
                tuple(tuple(sorted(row)) for row in durations), durations, setup, ref)


def decode(case: Case, order, machine):
    n = len(case.ops)
    if len(order) != n or set(order) != set(range(n)) or any(type(i) is not int for i in order):
        raise ValueError("Приоритеты должны быть перестановкой индексов операций")
    if len(machine) != n or any(type(r) is not int or r not in case.duration[i]
                                for i, r in enumerate(machine)):
        raise ValueError("Для каждой операции требуется допустимый ресурс")
    priority = [0] * n
    for rank, i in enumerate(order):
        priority[i] = rank
    deg = list(map(len, case.pred))
    q = [(priority[i], i) for i, v in enumerate(deg) if v == 0]
    heapify(q)
    end = [0] * n
    available = [0] * len(case.machines)
    family = [None] * len(case.machines)
    job_end = [0] * len(case.jobs)
    rows = [None] * n
    setup_total = 0
    while q:
        _, i = heappop(q)
        r = machine[i]
        op = case.ops[i]
        j, f = op["job"], op["family"]
        s = max(available[r], case.jobs[j]["release"], max((end[k] for k in case.pred[i]), default=0))
        change = case.setup[family[r]][f] if family[r] is not None else 0
        span = change + case.duration[i][r]
        for lo, hi in case.downtime[r]:
            if s + span <= lo:
                break
            if s < hi:
                s = hi
        finish = s + span
        available[r], family[r], end[i] = finish, f, finish
        job_end[j] = max(job_end[j], finish)
        setup_total += change
        rows[i] = (r, s, s + change, finish)
        for k in case.succ[i]:
            deg[k] -= 1
            if deg[k] == 0:
                heappush(q, (priority[k], k))
    late = sum(j["weight"] * max(0, t-j["due"]) for j, t in zip(case.jobs, job_end))
    return (max(end), late, setup_total), rows


def check(case: Case, rows):
    # Независимая проверка по готовым интервалам, без повторного декодирования.
    if len(rows) != len(case.ops):
        raise ValueError("Число операций не совпадает")
    by_resource = [[] for _ in case.machines]
    by_job = [[] for _ in case.jobs]
    for i, (r, s, p, e) in enumerate(rows):
        op = case.ops[i]
        j = op["job"]
        if r not in case.duration[i] or e-p != case.duration[i][r] or not 0 <= s <= p < e:
            raise ValueError(f"Недопустимый ресурс или длительность O{i}")
        if s < case.jobs[j]["release"] or any(s < rows[k][3] for k in case.pred[i]):
            raise ValueError(f"Нарушено время готовности O{i}")
        if any(s < b and a < e for a, b in case.downtime[r]):
            raise ValueError(f"Занятие ресурса вне календаря O{i}")
        by_resource[r].append((s,p,e,op["family"]))
        by_job[j].append(e)
    total = 0
    for intervals in by_resource:
        last, prev = 0, None
        for s,p,e,f in sorted(intervals):
            change = 0 if prev is None else case.setup[prev][f]
            if s < last or p-s != change:
                raise ValueError("Пересечение операций или неверная переналадка")
            total += change
            last, prev = e, f
    late = sum(j["weight"] * max(0, max(times)-j["due"]) for j,times in zip(case.jobs,by_job))
    return max(row[3] for row in rows), late, total


def generate(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, jobs, machines, seed, mult in [
        ("balanced", 6, 4, 5172401, 1), ("bottleneck", 8, 4, 5172402, 1),
        ("setups", 10, 5, 5172403, 3)
    ]:
        rng = random.Random(seed)
        d = {"id": name, "seed": seed, "resources": [], "jobs": [], "operations": [],
             "setup": [[0 if i == j else mult*(2 + (2*i+j)%3) for j in range(3)] for i in range(3)]}
        for r in range(machines):
            a = 18+r*6
            d["resources"].append({"name": f"R{r+1}", "downtime": [[a,a+6+r],[75+r*5,81+r*5]]})
        for j in range(jobs):
            release = rng.randrange(0,17)
            d["jobs"].append({"name": f"J{j+1:02}", "release": release,
                              "due": release + rng.randrange(34,72), "weight": rng.randint(1,4)})
            family = j % 3
            for step in range(4):
                i = len(d["operations"])
                primary = 0 if name == "bottleneck" and step in (1,2) else (j+step) % machines
                alternate = rng.choice([r for r in range(machines) if r != primary])
                base = rng.randint(5, 16)
                duration = {primary: base}
                if name != "bottleneck" or step != 1:
                    duration[alternate] = base+rng.randint(1,7)
                pred = [i-1] if step else []
                if j >= 2 and step == 2 and j % 2 == 0:
                    pred.append(i-6)
                d["operations"].append({"job": j, "step": step+1, "family": family,
                                        "pred": pred, "duration": duration})
        (out/f"{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
