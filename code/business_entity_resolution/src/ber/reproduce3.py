"""Single entry point reproducing the v3 package outputs from the supplied dataset."""
import argparse
import json
from pathlib import Path
from .audit import audit_outputs
from .pipeline3 import create_training_candidates, train_and_evaluate
from .predict import assemble
from .predict3 import predict

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="run_config.json")
    p.add_argument("--data", required=True)
    p.add_argument("--artifacts", default="artifacts/reproduce3")
    p.add_argument("--output", default="output")
    p.add_argument("--device", choices=("cpu", "cuda", "none"))
    p.add_argument("--exclusive", action="store_true", help="assign each target to at most one reference")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    settings = argparse.Namespace(**{"df_max": 50000, "k_all": 50, "k_name": 10, **config,
                                     "data": args.data, "artifacts": args.artifacts})
    if args.device:
        settings.xgboost_device = args.device
    create_training_candidates(settings)
    train_and_evaluate(settings)
    predict(args.data, args.artifacts, settings.workers, config.get("batch_size", 512))
    assemble(args.artifacts, args.output, exclusive=args.exclusive)
    report = audit_outputs(Path(args.output) / "matching_results.tsv", Path(args.output) / "candidate_pairs.tsv",
                           Path(args.data) / "test")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)
