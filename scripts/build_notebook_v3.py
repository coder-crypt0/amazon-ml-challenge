"""Build Business_Entity_Resolution_Colab_v3.ipynb with the complete source embedded."""
from pathlib import Path
import base64, hashlib, io, zipfile
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    files = [*sorted((root / "src").rglob("*.py")), *sorted((root / "tests").rglob("*.py"))]
    files += [root / "README.md", root / "requirements.txt", root / "LICENSE"]
    files += sorted((root / "reports").glob("*profile*"))
    for p in files:
        if "__pycache__" not in p.parts:
            info = zipfile.ZipInfo(p.relative_to(root).as_posix(), date_time=(2026, 9, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
payload = base64.b64encode(buf.getvalue()).decode()
sha = hashlib.sha256(buf.getvalue()).hexdigest()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip()))

md('''# Business Entity Resolution v3 • resumable Colab runner

v3 replaces key blocking with **IDF token retrieval plus a transliteration table learned from training pairs**, and adds house-number, retrieval-aware and candidate-context features. Local 5-fold validation on 10k training references: macro F0.5 0.912 (v2 replica) → 0.969.

**Start:** upload the dataset ZIP to Drive, select a GPU runtime (more CPU cores make scoring faster), edit `DATA_ZIP`, and run the cells in order. A new `RUN_NAME` keeps v3 checkpoints apart from earlier runs; record stores from `ber_v2`/`ber_v1` are reused when present.

**Resume:** keep the same `RUN_NAME` and settings and run from the top. Token-index shards, candidate batches, 50-tree model segments and prediction batches are restored from Drive.''')
code('''import os, sys, json, shutil, subprocess, hashlib, zipfile, base64, io
from pathlib import Path
SMOKE = os.environ.get("BER_NOTEBOOK_SMOKE") == "1"
if not SMOKE:
    from google.colab import drive
    drive.mount('/content/drive')
print("Drive connected" if not SMOKE else "Local notebook smoke test")''')
md('''## 1. Configuration
`DF_MAX` stops very common tokens from generating candidates (they still count in features). `K_ALL` and `K_NAME` are the per-reference candidates from the all-token and name-only channels.''')
code('''DATA_ZIP = Path(os.environ.get("BER_DATA_ZIP", "/content/drive/MyDrive/amazon-ml-challenge/6ab10eb3b23ba_student_resource.zip"))
RUN_NAME = "ber_v3"
REUSE_STORES_FROM = ("ber_v2", "ber_v1")
TEAM_NAME = "entity_resolution"
TEAM_MEMBERS = []  # Fill in for your final methodology.
DRIVE_ROOT = Path(os.environ.get("BER_DRIVE_ROOT", "/content/drive/MyDrive/amazon-ml-challenge"))
WORK = Path(os.environ.get("BER_WORK", "/content/ber_work"))
SAMPLES = 100 if SMOKE else 60000
TREES = 20 if SMOKE else 500
DF_MAX, K_ALL, K_NAME = 50000, (12 if SMOKE else 50), (4 if SMOKE else 10)
WORKERS = 1 if SMOKE else min(8, os.cpu_count() or 2)
BATCH_SIZE = 5 if SMOKE else 512
XGBOOST_DEVICE = "cpu" if SMOKE else "cuda"
SEED = 20260925
# Second runtime only: True scores missing test batches from the end backwards while the main
# runtime scores forwards (same Drive, same RUN_NAME). Start it after the main runtime prints "Ready: test".
HELPER = False
CODE = WORK / "code"
ARTIFACTS = WORK / "artifacts" / RUN_NAME
DURABLE = DRIVE_ROOT / "runs" / RUN_NAME
for p in (CODE, ARTIFACTS, DURABLE): p.mkdir(parents=True, exist_ok=True)
assert DATA_ZIP.is_file(), f"Edit DATA_ZIP: file not found: {DATA_ZIP}"
os.environ["BER_ARTIFACT_ROOT"] = str(ARTIFACTS.resolve())
os.environ["BER_CHECKPOINT_ROOT"] = str((DURABLE / "checkpoints").resolve())
os.environ["PYTHONPATH"] = str(CODE / "src") + os.pathsep + str(CODE)
os.environ["PYTHONUTF8"] = "1"
CONFIG = dict(version=3, samples=SAMPLES, trees=TREES, df_max=DF_MAX, k_all=K_ALL, k_name=K_NAME,
              workers=WORKERS, batch_size=BATCH_SIZE, xgboost_device=XGBOOST_DEVICE, seed=SEED)
print(json.dumps(CONFIG, indent=2))''')
md('## 2. Install the embedded, versioned source and dependencies')
code(f'''SOURCE_SHA256 = {sha!r}
SOURCE_PAYLOAD = {payload!r}
source_bytes = base64.b64decode(SOURCE_PAYLOAD)
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
with zipfile.ZipFile(io.BytesIO(source_bytes)) as archive:
    archive.extractall(CODE)
(CODE / "run_config.json").write_text(json.dumps(CONFIG, indent=2))
if not SMOKE:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE / "requirements.txt"), "pytest==8.4.2"])
if str(CODE / "src") not in sys.path: sys.path.insert(0, str(CODE / "src"))
os.chdir(CODE)
from ber.checkpoint import ensure_config, file_digest
manifest = dict(source_sha256=SOURCE_SHA256, dataset_sha256=file_digest(DATA_ZIP), config=CONFIG)
ensure_config(ARTIFACTS / "run_manifest.json", manifest)
(DURABLE / "run_config.json").write_text(json.dumps(CONFIG, indent=2))
(DURABLE / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
(DURABLE / "business_entity_resolution_code.zip").write_bytes(source_bytes)
def run(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=CODE)
run("-m", "pytest", "tests", "-q")
print("Source SHA256:", SOURCE_SHA256)''')
md('''## 3. Extract the supplied dataset locally''')
code('''DATA_ROOT = WORK / "resource"
DATA_ROOT.mkdir(exist_ok=True)
marker = DATA_ROOT / "extraction_complete.json"
if not marker.exists():
    with zipfile.ZipFile(DATA_ZIP) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            if not parts or parts[0] != "student_resource" or info.is_dir() or Path(info.filename).name.startswith("."):
                continue
            target = (DATA_ROOT / info.filename).resolve()
            if not target.is_relative_to(DATA_ROOT.resolve()): raise ValueError("Unsafe archive path")
            archive.extract(info, DATA_ROOT)
    marker.write_text(json.dumps({"archive_sha256": manifest["dataset_sha256"]}))
else:
    assert json.loads(marker.read_text())["archive_sha256"] == manifest["dataset_sha256"], "Use a fresh WORK directory for a different dataset"
DATA = DATA_ROOT / "student_resource" / "dataset"
assert (DATA / "train/train_ground_truth.tsv").is_file()
print("Dataset:", DATA)
print("Runtime disk free (GB):", round(shutil.disk_usage(WORK).free / 1e9, 1))''')
md('''## 3b. Reuse record stores from an earlier run
Record stores are identical across versions (their input checksums are verified on load). Token indexes are new in v3 and are always built.''')
code('''for split in ("train", "test"):
    dst = ARTIFACTS / split / "store"
    if (dst / "COMPLETE").exists():
        continue
    sources = [d for run_name in REUSE_STORES_FROM for d in (WORK / "artifacts" / run_name / split / "store",
               DRIVE_ROOT / "runs" / run_name / "checkpoints" / split / "store")]
    src = next((d for d in sources if (d / "COMPLETE").exists()), None)
    if src is None:
        print(f"{split}/store: nothing to reuse; it will be built")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"{split}/store: reused from {src}")''')
md('''## 4. Learn transliteration, build the training token index, generate candidates
The transliteration table is learned from training pairs whose reference entities are **not** in the sampled validation queries. All sampled references search the complete 10.3M-record training target pool; the split is 70/15/15 by reference.''')
code('''if HELPER: print("HELPER: skipping training stages")
else: run("-m", "ber.pipeline3", "candidates", "--data", DATA, "--artifacts", ARTIFACTS, "--samples", SAMPLES,
    "--seed", SEED, "--df-max", DF_MAX, "--k-all", K_ALL, "--k-name", K_NAME, "--workers", WORKERS)
if not HELPER:
    print(json.loads((ARTIFACTS / "experiment/retrieval.json").read_text())["candidates_mean"], "candidates per reference")''')
md('''## 5. Train, checkpoint and calibrate
Name-word odds are out-of-fold for fit pairs. Threshold and ensemble weight are chosen on the tuning split; the holdout is only reported.''')
code('''if not HELPER:
  run("-m", "ber.pipeline3", "train", "--data", DATA, "--artifacts", ARTIFACTS, "--samples", SAMPLES, "--seed", SEED,
    "--df-max", DF_MAX, "--k-all", K_ALL, "--k-name", K_NAME, "--workers", WORKERS, "--trees", TREES,
    "--xgboost-device", XGBOOST_DEVICE)
  REPORT = json.loads((ARTIFACTS / "experiment/validation.json").read_text())
  print(json.dumps({k: REPORT[k] for k in ("threshold", "xgboost_weight", "holdout", "holdout_by_country")}, indent=2))
  REPORTS = CODE / "reports"
  REPORTS.mkdir(exist_ok=True)
  shutil.copyfile(ARTIFACTS / "experiment/validation.json", REPORTS / "validation.json")
  run("-m", "ber.plots", "--report", REPORTS / "validation.json", "--output", REPORTS / "figures")
  shutil.copytree(REPORTS, DURABLE / "reports", dirs_exist_ok=True)''')
md('''## 6. Build the test token index and score every reference
All countries, including France, flow through the same pipeline. Re-running skips completed prediction batches. Each checkpoint line shows queries/second and the hours left at that rate: **check it after the first few batches against your deadline.** To roughly halve the wall time, open this notebook in a second runtime on the same Drive with the same `RUN_NAME`, set `HELPER = True`, and run all cells once this runtime has printed `Ready: test`; the helper scores from the last batch backwards and this runtime picks up its batches.''')
code('''run("-m", "ber.predict3", "score", "--data", DATA, "--artifacts", ARTIFACTS, "--workers", WORKERS,
    "--batch-size", BATCH_SIZE, *(["--reverse"] if HELPER else []))''')
md('''## 7. Assemble, audit and package two submissions
`output/` applies the tuned threshold. `output_exclusive/` also assigns each target to at most one reference (the highest probability; ties drop). Every target belongs to one reference in the training labels, but the sampled validation cannot measure this effect, so compare both on the public leaderboard.''')
code('''from ber.predict import assemble
from ber.finalize3 import finalize
RESULTS = {}
for label, exclusive in (() if HELPER else (("", False), ("_exclusive", True))):
    out = WORK / f"output{label}"
    summary = assemble(ARTIFACTS, out, exclusive=exclusive)
    submission, audit = finalize(DATA, ARTIFACTS, out, CODE, TEAM_NAME, TEAM_MEMBERS, label)
    assert audit["ok"], audit["errors"][:5]
    shutil.copytree(out, DURABLE / f"output{label}", dirs_exist_ok=True)
    shutil.copyfile(submission, DURABLE / submission.name)
    shutil.copyfile(CODE / "Documentation_template.md", DURABLE / f"Documentation_template{label}.md")
    RESULTS[label or "plain"] = dict(matched_links=summary["matched_links"], conflicts=summary["conflicts"],
                                     leaderboard_file=str(DURABLE / f"output{label}/matching_results.tsv"),
                                     package=str(DURABLE / submission.name))
print(json.dumps(RESULTS, indent=2))''')
md('''## 8. Download
Upload `output/matching_results.tsv` first, then `output_exclusive/matching_results.tsv`, and keep the better one. Each has its matching ZIP and methodology in Drive.''')
code('''if HELPER:
    print("HELPER done; the main runtime assembles and downloads the outputs.")
elif not SMOKE:
    from google.colab import files
    files.download(str(DURABLE / "output/matching_results.tsv"))
    files.download(str(DURABLE / "output_exclusive/matching_results.tsv"))
else:
    print("Notebook smoke test completed successfully.")''')
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                "language_info": {"name": "python", "version": "3.12"},
                                                "colab": {"name": "Business_Entity_Resolution_Colab_v3.ipynb", "provenance": []},
                                                "accelerator": "GPU"})
nbf.validate(nb)
output = root / "Business_Entity_Resolution_Colab_v3.ipynb"
nbf.write(nb, output)
print(output)
print("cells", len(cells), "bytes", output.stat().st_size, "source", sha)
