from __future__ import annotations
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scheduling import read_case

ROOT=Path(__file__).resolve().parent
LABELS=["Время завершения, мин","Взвешенная просрочка, балл·мин","Переналадки, мин"]
COLORS=["#146a8e","#b75921"]
NAMES={"balanced":"Умеренная нагрузка","bottleneck":"Узкое место","setups":"Дорогие переналадки"}
PICKS={"fast":"Раннее завершение","due":"Меньше просрочек","setup":"Меньше переналадок",
       "balanced":"Ближе к идеалу"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=ROOT/"data")
    p.add_argument("--results",type=Path,default=ROOT/"results")
    a=p.parse_args()
    out=a.results/"figures"
    out.mkdir(exist_ok=True)
    summary=json.loads((a.results/"summary.json").read_text())
    runs=[json.loads(x) for x in (a.results/"runs.jsonl").read_text().splitlines()]
    cases={c.id:c for c in map(read_case,sorted(a.data.glob("*.json")))}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False})
    fig,axes=plt.subplots(1,len(cases),figsize=(11,3.15),squeeze=False)
    for (key,info),ax in zip(summary["cases"].items(),axes.flat):
        for m,color,name in zip(("base","local"),COLORS,("NSGA-II","С локальным поиском")):
            h=np.array([r["history"] for r in runs if r["case"]==key and r["method"]==m])
            x=h[0,:,0]; q=np.quantile(h[:,:,1],[.25,.5,.75],axis=0)
            ax.plot(x,q[1],color=color,label=name,lw=1.6)
            ax.fill_between(x,q[0],q[2],color=color,alpha=.16)
        ax.set(title=NAMES.get(key,key),xlabel="Оценок расписания")
        ax.grid(alpha=.18)
    axes[0,0].set_ylabel("Нормированный HV")
    axes[0,0].legend(fontsize=7,loc="lower right")
    fig.tight_layout(); fig.savefig(out/"convergence.png",dpi=180); plt.close(fig)

    gantt_palette=plt.get_cmap("tab10")
    selectors=[]; payload={}
    for key,info in summary["cases"].items():
        case=cases[key]
        payload[key]={"title":NAMES.get(key,key),"machines":case.machines,"down":case.downtime,
                      "ops":case.ops,"jobs":case.jobs,"methods":info["examples"]}
        fig,axes=plt.subplots(1,3,figsize=(10.7,3.05))
        for ax,(x,y) in zip(axes,((0,1),(0,2),(1,2))):
            for m,color,label,mark in zip(("base","local"),COLORS,("NSGA-II","С локальным поиском"),("o","x")):
                vals=np.array([pt["f"] for pt in info["examples"][m]["front"]])
                ax.scatter(vals[:,x],vals[:,y],s=23,color=color,label=label,marker=mark,alpha=.8)
            ax.set(xlabel=LABELS[x],ylabel=LABELS[y]); ax.grid(alpha=.18)
        axes[0].legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f"front_{key}.png",dpi=180);plt.close(fig)
        ex=info["examples"]["local"]
        choices=list(dict.fromkeys(ex["picks"].values()))
        with (a.results/f"schedules_{key}.csv").open("w",encoding="utf-8",newline="") as fp:
            w=csv.writer(fp);w.writerow(["point","seed","operation","job","resource","setup_start","process_start","end"])
            for idx in choices:
                for i,(r,s,t,e) in enumerate(ex["front"][idx]["schedule"]):
                    w.writerow([idx,ex["seed"],i,case.jobs[case.ops[i]["job"]]["name"],case.machines[r],s,t,e])
        fig,axes=plt.subplots(len(choices),1,figsize=(10.7,1.68*len(choices)),squeeze=False)
        xmax=max(ex["front"][i]["f"][0] for i in choices)
        for ax,idx in zip(axes.flat,choices):
            row=ex["front"][idx]
            for r,windows in enumerate(case.downtime):
                for lo,hi in windows:
                    ax.barh(r,hi-lo,left=lo,height=.78,color="#d8e0e5",zorder=1)
            for i,(r,s,t,e) in enumerate(row["schedule"]):
                j=case.ops[i]["job"]
                ax.barh(r,e-t,left=t,height=.64,color=gantt_palette(j%10),zorder=3)
                if t>s:
                    ax.barh(r,t-s,left=s,height=.64,color="#354355",zorder=2)
                if e-t>=7:
                    ax.text((t+e)/2,r,str(j+1),ha="center",va="center",fontsize=6,color="white")
            label=" / ".join(PICKS[k] for k,i in ex["picks"].items() if i==idx)
            ax.set(title=f'{label}: {tuple(row["f"])}',yticks=range(len(case.machines)),
                   yticklabels=case.machines,xlim=(0,xmax*1.03))
            ax.invert_yaxis(); ax.grid(axis="x",alpha=.15)
            ax.title.set_fontsize(8)
        axes[-1,0].set_xlabel("Модельное время, мин")
        fig.tight_layout(); fig.savefig(out/f"gantt_{key}.png",dpi=170);plt.close(fig)

    raw=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")
    template=(ROOT/"docs"/"viewer.template.html").read_text(encoding="utf-8")
    (a.results/"viewer.html").write_text(template.replace("__LAB_DATA__",raw),encoding="utf-8")


if __name__=="__main__":
    main()
