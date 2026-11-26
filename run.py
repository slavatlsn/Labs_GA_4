from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import numpy as np
from scheduling import read_case, generate, decode, check
from evolution import solve

ROOT = Path(__file__).resolve().parent


def run(task):
    path, seed, budget, pop, local = task
    case = read_case(path)
    t = perf_counter()
    front, history, info = solve(case,seed,budget,pop,local)
    return {"case":case.id,"method":"local" if local else "base","seed":seed,
            "seconds":perf_counter()-t, "hv":history[-1][1],"size":len(front),
            **info,"history":history,"front":[{"f":c.f,"order":c.order,"machine":c.machine} for c in front]}


def stats(seq):
    return dict(min=float(min(seq)), mean=float(mean(seq)), median=float(median(seq)),
                std=float(stdev(seq)) if len(seq)>1 else 0., max=float(max(seq)))


def preferred(points):
    f = np.array([p["f"] for p in points],dtype=float)
    lo,hi = f.min(axis=0),f.max(axis=0)
    norm = (f-lo)/np.where(hi>lo,hi-lo,1)
    return {**{name:int(np.lexsort((f.sum(axis=1),f[:,j]))[0]) for j,name in enumerate(("fast","due","setup"))},
            "balanced":int(np.argmin(np.linalg.norm(norm,axis=1)))}


def main():
    p = argparse.ArgumentParser(description="ЛР4: воспроизводимые парные эксперименты")
    p.add_argument("--runs",type=int,default=30)
    p.add_argument("--budget",type=int,default=1200)
    p.add_argument("--population",type=int,default=40)
    p.add_argument("--first-seed",type=int,default=4000)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--out",type=Path,default=ROOT/"results")
    p.add_argument("--generate",action="store_true")
    a=p.parse_args()
    if a.runs < 1 or a.workers < 1:
        p.error("runs и workers должны быть положительными")
    if a.population < 8 or a.budget < 2*a.population or a.budget % a.population:
        p.error("популяция должна быть не меньше 8; бюджет кратен популяции и не меньше двух её размеров")
    if a.generate:
        generate(a.data)
    paths=sorted(a.data.glob("*.json"))
    if not paths:
        p.error("нет JSON сценариев")
    cases = {}
    for path in paths:
        try:
            case = read_case(path)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            p.error(f"{path}: {exc}")
        if len(case.ops) < 2:
            p.error(f"{path}: нужны минимум две операции")
        cases[case.id] = case
    if len(cases) != len(paths):
        p.error("ID сценариев должны быть уникальными")
    tasks=[(path,seed,a.budget,a.population,local) for path in paths
           for seed in range(a.first_seed,a.first_seed+a.runs) for local in (False,True)]
    if a.workers == 1:
        runs=list(map(run,tasks))
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            runs=list(pool.map(run,tasks,chunksize=2))
    a.out.mkdir(parents=True,exist_ok=True)
    columns=["case","method","seed","evaluations","valid","hv","size","seconds","attempts","accepted"]
    with (a.out/"runs.csv").open("w",encoding="utf-8",newline="") as fp:
        w=csv.DictWriter(fp,columns,extrasaction="ignore"); w.writeheader(); w.writerows(runs)
    with (a.out/"runs.jsonl").open("w",encoding="utf-8") as fp:
        for r in runs:
            fp.write(json.dumps(r,separators=(",",":"))+"\n")
    summ={"runs_per_method_case":a.runs,"budget":a.budget,"population":a.population,
          "first_seed":a.first_seed,"workers":a.workers,"cases":{}}
    for case_id,case in cases.items():
        base,local=[[r for r in runs if r["case"]==case_id and r["method"]==m] for m in ("base","local")]
        assert [r["seed"] for r in base]==[r["seed"] for r in local]
        delta=np.array([b["hv"]-a["hv"] for a,b in zip(base,local)])
        boot=np.random.default_rng(51724).choice(delta,(10000,len(delta))).mean(axis=1)
        best={}
        for m,rs in (("base",base),("local",local)):
            r=sorted(rs,key=lambda r:(r["hv"],r["seed"]))[(len(rs)-1)//2]
            best[m]={"seed":r["seed"],"hv":r["hv"],"picks":preferred(r["front"]),
                     "front":[]}
            for point in r["front"]:
                f,rows=decode(case,point["order"],point["machine"])
                assert f==check(case,rows)==tuple(point["f"])
                best[m]["front"].append({"f":f,"schedule":rows})
        summ["cases"][case_id]={"reference":case.reference,"jobs":len(case.jobs),"ops":len(case.ops),
            "resources":len(case.machines),"base":stats([r["hv"] for r in base]),
            "local":stats([r["hv"] for r in local]),"delta":stats(delta),
            "ci":np.quantile(boot,[.025,.975]).tolist(),
            "wins":[int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum()),int((delta < -1e-12).sum())],
            "runtimes":[median(r["seconds"] for r in v) for v in (base,local)],
            "median_front_sizes":[median(r["size"] for r in v) for v in (base,local)],
            "local_attempts":sum(r["attempts"] for r in local),"local_accepted":sum(r["accepted"] for r in local),
            "examples":best}
        print(case_id, "delta =",mean(delta), "W/T/L =",summ["cases"][case_id]["wins"],flush=True)
    (a.out/"summary.json").write_text(json.dumps(summ,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
