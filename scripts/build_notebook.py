from pathlib import Path
import base64,hashlib,io,zipfile,json
import nbformat as nbf
root=Path(__file__).resolve().parents[1]
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    files=[*sorted((root/'src').rglob('*.py')),*sorted((root/'tests').rglob('*.py'))]
    files += [root/'README.md',root/'requirements.txt',root/'LICENSE']
    files += sorted((root/'reports').glob('*profile*'))
    for p in files:
        if '__pycache__' not in p.parts:
            info=zipfile.ZipInfo(p.relative_to(root).as_posix(),date_time=(2026,9,25,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
payload=base64.b64encode(buf.getvalue()).decode()
sha=hashlib.sha256(buf.getvalue()).hexdigest()
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
md('''# Business Entity Resolution • resumable Colab runner

**Start:** upload the original dataset ZIP to Google Drive → select **Runtime → Change runtime type → GPU** → edit `DATA_ZIP` below → run cells in order.

This notebook contains all source code. It uses supplied business data only. The local integration suite verifies training, resume, inference and submission audit. CUDA execution must be verified on the allocated Colab GPU; local validation is not a leaderboard score.

**Resume:** keep the same `RUN_NAME` and configuration, then run from the top. Completed shards, feature batches, models and prediction batches are restored from Drive. Do not run two copies against one run directory. Plan for ~12 GB free Drive space and 25 GB local runtime space. Retrieval runs on CPU; XGBoost training uses GPU. Full inference over 1.73M queries can take hours.
''')
code('''import os, sys, json, shutil, subprocess, hashlib, zipfile, base64, io
from pathlib import Path
SMOKE = os.environ.get("BER_NOTEBOOK_SMOKE") == "1"
if not SMOKE:
    from google.colab import drive
    drive.mount('/content/drive')
print("Drive connected" if not SMOKE else "Local notebook smoke test")''')
md('''## 1. Configuration
Change `DATA_ZIP` to the file you uploaded. `RUN_NAME` identifies checkpoints; use a new name for a changed experiment. Default settings sample 60,000 training references against **all** target records. `RUN_COUNTRY_TRANSFER` adds two diagnostic models and can be disabled to save time.''')
code('''DATA_ZIP = Path(os.environ.get("BER_DATA_ZIP", "/content/drive/MyDrive/amazon-ml-challenge/6ab10eb3b23ba_student_resource.zip"))
RUN_NAME = "ber_v2"
TEAM_NAME = "entity_resolution"
TEAM_MEMBERS = []  # Fill in for your final methodology.
DRIVE_ROOT = Path(os.environ.get("BER_DRIVE_ROOT", "/content/drive/MyDrive/amazon-ml-challenge"))
WORK = Path(os.environ.get("BER_WORK", "/content/ber_work"))
SAMPLES = 100 if SMOKE else 60000
TREES = 20 if SMOKE else 500
TOP_K = 12 if SMOKE else 96
MAX_POSTINGS = 250
WORKERS = 1 if SMOKE else min(8, os.cpu_count() or 2)
BATCH_SIZE = 5 if SMOKE else 512
RUN_COUNTRY_TRANSFER = False  # diagnostic only; enable for the methodology report
XGBOOST_DEVICE = "cpu" if SMOKE else "cuda"
SEED = 20260925
CODE = WORK / "code"
ARTIFACTS = WORK / "artifacts" / RUN_NAME
DURABLE = DRIVE_ROOT / "runs" / RUN_NAME
for p in (CODE, ARTIFACTS, DURABLE): p.mkdir(parents=True, exist_ok=True)
assert DATA_ZIP.is_file(), f"Edit DATA_ZIP: file not found: {DATA_ZIP}"
os.environ["BER_ARTIFACT_ROOT"] = str(ARTIFACTS.resolve())
os.environ["BER_CHECKPOINT_ROOT"] = str((DURABLE / "checkpoints").resolve())
os.environ["PYTHONPATH"] = str(CODE / "src") + os.pathsep + str(CODE)
os.environ["PYTHONUTF8"] = "1"
CONFIG = dict(samples=SAMPLES, trees=TREES, top_k=TOP_K, max_postings=MAX_POSTINGS,
              workers=WORKERS, batch_size=BATCH_SIZE, country_transfer=RUN_COUNTRY_TRANSFER,
              xgboost_device=XGBOOST_DEVICE, seed=SEED)
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
from ber.checkpoint import ensure_config, file_digest, persist_directory, restore_directory
manifest = dict(source_sha256=SOURCE_SHA256, dataset_sha256=file_digest(DATA_ZIP), config=CONFIG)
ensure_config(ARTIFACTS / "run_manifest.json", manifest)
(DURABLE / "run_config.json").write_text(json.dumps(CONFIG, indent=2))
(DURABLE / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
(DURABLE / "business_entity_resolution_code.zip").write_bytes(source_bytes)
def run(*args):
    subprocess.run([sys.executable, *map(str,args)], check=True, cwd=CODE)
run("-m", "pytest", "tests", "-q")
if not SMOKE:
    subprocess.run(["nvidia-smi"], check=True)
print("Source SHA256:", SOURCE_SHA256)''')
md('''## 3. Extract the supplied dataset locally
ZIP data is read from Drive once. Random-access stores and arrays live on the runtime SSD; completed checkpoints are copied to Drive. After a reset, local extraction repeats, while durable checkpoints are restored.''')
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
md('''## 3b. Reuse stores and indexes from an earlier run
Blocking keys and record normalization are unchanged since `ber_v1` (the normalization speed-up was verified output-identical on every train and test record), so its stores and indexes are reused instead of rebuilt. Anything not found is built normally by steps 4 and 9.''')
code('''REUSE_FROM = "ber_v1"
for split in ("train", "test"):
    for part in ("store", "index"):
        dst = ARTIFACTS / split / part
        if (dst / "COMPLETE").exists():
            continue
        local = WORK / "artifacts" / REUSE_FROM / split / part
        remote = DRIVE_ROOT / "runs" / REUSE_FROM / "checkpoints" / split / part
        src = next((d for d in (local, remote) if (d / "COMPLETE").exists()), None)
        if src is None:
            print(f"{split}/{part}: nothing to reuse; it will be built")
            continue
        if dst.exists() or dst.is_symlink():
            shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src == local:
            os.symlink(src, dst, target_is_directory=True)
        else:
            shutil.copytree(src, dst)
        print(f"{split}/{part}: reused from {src}")''')
md('''## 4. Build the full training search index
This CPU stage processes all Source 2/3 records. Every 250,000-record key shard is checkpointed. The final sort is one resumable-stage boundary: if interrupted during sorting, the stored key shards are reused. The index uses ~2 GB for this dataset.''')
code('''run("-m", "ber.build", "--data", DATA, "--artifacts", ARTIFACTS, "--split", "train", "--workers", WORKERS)''')
md('''## 5. Generate entity-disjoint training / tuning / holdout candidates
The split is 70/15/15. All sampled references search the complete 10.3M-record training target pool. Feature checkpoints are saved every 64 references. A random sample makes the ML stage practical without shrinking the distractor pool.''')
code('''run("-m", "ber.pipeline", "candidates", "--data", DATA, "--artifacts", ARTIFACTS,
    "--samples", SAMPLES, "--seed", SEED, "--top-k", TOP_K,
    "--max-postings", MAX_POSTINGS, "--workers", WORKERS)''')
md('''## 6. Train, checkpoint and calibrate
LightGBM trains on CPU. XGBoost uses CUDA histogram training on Colab. Both save every 50 trees. Ensemble weight and threshold are chosen on the tuning split; the holdout is reserved for reporting. Native model files are used instead of pickled estimator objects.''')
code('''run("-m", "ber.pipeline", "train", "--data", DATA, "--artifacts", ARTIFACTS,
    "--samples", SAMPLES, "--seed", SEED, "--top-k", TOP_K, "--max-postings", MAX_POSTINGS,
    "--workers", WORKERS, "--trees", TREES, "--xgboost-device", XGBOOST_DEVICE)''')
md('''## 7. Structural ablation and unseen-country diagnostics
A target record can belong to only one deduplicated reference. The constraint is enabled only if tuning macro-F0.5 improves. The optional leave-country-out tests train without one country and evaluate that country; they are **not** a France score. No external lookup is used.''')
code('''experiment_args = ["-m", "ber.experiments", "--artifacts", ARTIFACTS, "--threads", WORKERS]
if RUN_COUNTRY_TRANSFER: experiment_args.append("--country-transfer")
run(*experiment_args)
REPORT = json.loads((ARTIFACTS / "experiment/validation.json").read_text())
print(json.dumps({k: REPORT[k] for k in ("holdout", "holdout_by_country", "graph_ablation")}, indent=2))''')
md('''## 7b. Tune the match-selection policy
Per query: keep candidates with p >= t, also keep the top candidate when p >= t1, and drop candidates below r x the query's best probability; optionally per country. Chosen on the tuning split only; the holdout line is the unbiased estimate.''')
code('''POLICY = ARTIFACTS / "experiment/policy.json"
run("-m", "ber.policy", ARTIFACTS / "experiment", POLICY)
shutil.copyfile(POLICY, DURABLE / "policy.json")
print(POLICY.read_text())''')
md('''## 8. Export report-quality figures and reconstructable JSON
Plots are written as PNG and SVG. `validation.json` and `figures/plot_data.json` contain the underlying measurements; rerun `ber.plots` later without retraining. France has no labeled score.''')
code('''REPORTS = CODE / "reports"
REPORTS.mkdir(exist_ok=True)
shutil.copyfile(ARTIFACTS / "experiment/validation.json", REPORTS / "validation.json")
run("-m", "ber.plots", "--report", REPORTS / "validation.json", "--output", REPORTS / "figures")
from IPython.display import display, Image
for path in sorted((REPORTS / "figures").glob("*.png")): display(Image(filename=str(path)))
shutil.copytree(REPORTS, DURABLE / "reports", dirs_exist_ok=True)
print("Reports saved:", DURABLE / "reports")''')
md('''## 9. Build the full test index
All countries flow through the same pipeline, including France. This stage also checkpoints record storage, key shards and the completed index.''')
code('''run("-m", "ber.build", "--data", DATA, "--artifacts", ARTIFACTS, "--split", "test", "--workers", WORKERS)''')
md('''## 10. Predict with batch-level recovery
Each checkpoint records the exact candidates scored by the model, their probabilities and query boundaries. Re-running skips completed batches. Full scoring is CPU-heavy even with a GPU runtime. Do not delete checkpoints until the submission is accepted.''')
code('''run("-m", "ber.predict", "score", "--data", DATA, "--artifacts", ARTIFACTS,
    "--workers", WORKERS, "--batch-size", BATCH_SIZE)''')
md('''## 11. Assemble outputs, audit every ID, and package
The assembly stage applies the tuning-selected policy and generates both required TSVs. The strict audit verifies every Source 1 row, target existence, duplicate lists and candidate containment. Methodology and results are filled from the actual run.''')
code('''OUTPUT = WORK / "output"
run("-m", "ber.predict", "assemble", "--artifacts", ARTIFACTS, "--output", OUTPUT,
    "--policy", ARTIFACTS / "experiment/policy.json", "--data", DATA)
from ber.finalize import finalize
submission, audit = finalize(DATA, ARTIFACTS, OUTPUT, CODE, TEAM_NAME, TEAM_MEMBERS)
shutil.copytree(OUTPUT, DURABLE / "output", dirs_exist_ok=True)
shutil.copyfile(submission, DURABLE / submission.name)
shutil.copyfile(CODE / "Documentation_template.md", DURABLE / "Documentation_template.md")
print(json.dumps(audit, indent=2))
print("LEADERBOARD FILE:", DURABLE / "output/matching_results.tsv")
print("FINAL PACKAGE:", DURABLE / submission.name)
assert audit["ok"]''')
md('''## 12. Download
Upload `matching_results.tsv` to the leaderboard. Keep the final ZIP for the required code/methodology submission. Both are already in Drive.''')
code('''if not SMOKE:
    from google.colab import files
    files.download(str(DURABLE / "output/matching_results.tsv"))
else:
    print("Notebook smoke test completed successfully.")''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'colab':{'name':'Business_Entity_Resolution_Colab.ipynb','provenance':[]},'accelerator':'GPU'})
nbf.validate(nb)
output=root/'Business_Entity_Resolution_Colab.ipynb'
nbf.write(nb,output)
(root/'business_entity_resolution_code.zip').write_bytes(buf.getvalue())
print(output)
print('cells',len(cells),'bytes',output.stat().st_size,'source',sha)
