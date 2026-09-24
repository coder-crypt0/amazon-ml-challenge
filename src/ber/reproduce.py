"""Single entry point for reproducing code-package outputs."""
import argparse
import json
from pathlib import Path
from .build import build
from .experiments import run
from .pipeline import create_training_candidates,train_and_evaluate
from .predict import predict,assemble
from .audit import audit_outputs


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="run_config.json")
    p.add_argument("--data",required=True)
    p.add_argument("--artifacts",default="artifacts/reproduce")
    p.add_argument("--output",default="output")
    p.add_argument("--device",choices=("cpu","cuda","none"))
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    settings=argparse.Namespace(**{**config,"data":args.data,"artifacts":args.artifacts})
    if args.device:
        settings.xgboost_device=args.device
    build(args.data,args.artifacts,"train",settings.workers)
    create_training_candidates(settings)
    train_and_evaluate(settings)
    run(args.artifacts,settings.workers,config.get("country_transfer",False))
    build(args.data,args.artifacts,"test",settings.workers)
    predict(args.data,args.artifacts,settings.workers,config.get("batch_size",512))
    assemble(args.artifacts,args.output)
    report=audit_outputs(Path(args.output)/"matching_results.tsv",Path(args.output)/"candidate_pairs.tsv",Path(args.data)/"test")
    print(json.dumps(report,indent=2))
    if not report["ok"]:
        raise SystemExit(1)
