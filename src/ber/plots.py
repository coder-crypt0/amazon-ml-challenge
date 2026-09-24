"""Recreate report figures from JSON, without loading the dataset or model."""
import argparse
import json
from pathlib import Path


def create_plots(report_path,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    report=json.loads(Path(report_path).read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    paths=[]
    def save(fig,name):
        fig.tight_layout()
        for suffix in ("png","svg"):
            fig.savefig(output/f"{name}.{suffix}",dpi=180,bbox_inches="tight")
        paths.append(str(output/f"{name}.png"));plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for weight in sorted({r.get("xgboost_weight",0) for r in report["threshold_curve"]}):
        curve=[r for r in report["threshold_curve"] if r.get("xgboost_weight",0)==weight]
        ax.plot([r["threshold"] for r in curve],[r["macro_f05"] for r in curve],label=f"XGBoost weight {weight:g}")
    ax.axvline(report["threshold"],color="#d97706",linestyle="--",label="Selected threshold")
    ax.set(xlabel="Match probability threshold",ylabel="Macro F0.5",title="Tuning decisions • entity-disjoint calibration set")
    ax.legend(fontsize=8);ax.grid(alpha=.15);save(fig,"threshold_calibration")
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    groups={"Overall":report["holdout"],**report["holdout_by_country"]}
    labels=list(groups)
    metrics=("macro_f05","micro_precision","micro_recall")
    x=np.arange(len(labels));colors=("#2563eb","#0d9488","#f59e0b")
    for i,(key,color) in enumerate(zip(metrics,colors)):
        axes[0].bar(x+(i-1)*.24,[groups[k][key] for k in labels],width=.24,label=key,color=color)
    axes[0].set_xticks(x,labels);axes[0].set_ylim(0,1.05);axes[0].set_title("Untouched holdout performance")
    axes[0].legend(fontsize=7,loc="lower left")
    for i,key in enumerate(("candidate_recall","singleton_accuracy")):
        axes[1].bar(x+(i-.5)*.32,[groups[k][key] or 0 for k in labels],width=.32,label=key,color=colors[i])
    axes[1].set_xticks(x,labels);axes[1].set_ylim(0,1.05);axes[1].set_title("Retrieval coverage and abstention")
    axes[1].legend(fontsize=7,loc="lower left")
    fig.suptitle("France is unseen in training and has no labeled score",fontsize=10,color="#555555")
    save(fig,"holdout_performance")
    importance=list(report["feature_importance"].items())[:15][::-1]
    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.barh([name for name,_ in importance],[value for _,value in importance],color="#2563eb")
    ax.set(title="LightGBM feature importance",xlabel="Training split gain (not causal importance)")
    save(fig,"feature_importance")
    (output/"plot_data.json").write_text(json.dumps({"schema_version":1,"report":report,
        "recreate":"python -m ber.plots --report validation.json --output figures"},indent=2))
    return paths


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--report",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();print(json.dumps(create_plots(args.report,args.output),indent=2))
