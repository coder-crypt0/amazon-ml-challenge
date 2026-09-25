"""Generate a self-contained Kaggle notebook from the exact tracked source."""
from pathlib import Path
import base64
import hashlib
import io
import zipfile
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
with zipfile.ZipFile(buffer,"w",zipfile.ZIP_DEFLATED) as archive:
    files=[*sorted((root/"src/gpu100").glob("*.py")),
           root/"src/ber/__init__.py",root/"src/ber/audit.py",
           root/"GPU100_README.md",root/"requirements-kaggle.txt",root/"LICENSE",
           root/"tests/__init__.py",root/"tests/fixtures.py"]
    for path in files:
        info=zipfile.ZipInfo(path.relative_to(root).as_posix(),date_time=(2026,9,25,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        archive.writestr(info,path.read_bytes())
payload=base64.b64encode(buffer.getvalue()).decode()
digest=hashlib.sha256(buffer.getvalue()).hexdigest()
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text.strip()))
md('''# CNER — precision-first Kaggle entity resolution

Attach the original student-resource ZIP as a **private Kaggle Dataset** and enable a **GPU** accelerator. Edit the configuration cell, then run from the top. The notebook embeds the exact source code. All business data comes from the supplied archive; package downloads provide libraries only.

The run reproduces the original 60,000-reference entity split, adds fit-only training entities, and reports candidate recall separately from macro F₀.₅. It learns source-specific corruption from fit positives, trains a GPU ranker and hard-negative TabM pair matcher, and validates ownership, a small multilingual cross-encoder tail, and one conservative sibling pass. Each component is kept only when the tuning fold improves. Kaggle [documents a 12-hour GPU session](https://www.kaggle.com/docs/notebooks); 5½ hours is a target, not a measured guarantee.

Method references: [Sparkly TF-IDF blocking](https://pages.cs.wisc.edu/~anhai/papers1/sparkly-tr22.pdf), [SC-Block supervised blocking](https://arxiv.org/abs/2303.03132). This implementation is tuned to the challenge data and uses no external business lookup.''')
code('''import os,sys,json,time,shutil,subprocess,zipfile,hashlib,io,base64
from pathlib import Path
SMOKE=os.environ.get("GPU100_NOTEBOOK_SMOKE")=="1"
RUN_NAME="cner_v1"
SAMPLES=100 if SMOKE else 120000
TREES=25 if SMOKE else 400
EPOCHS=2 if SMOKE else 12
FINAL_K=16 if SMOKE else 128
BATCH_SIZE=20 if SMOKE else 512
THREADS=2 if SMOKE else min(4,os.cpu_count() or 4)
DIMENSION=(1<<14) if SMOKE else (1<<20)
CAP_RATE=.0025
CHANNEL_EXTRA=0 if SMOKE else 300000
TEAM_NAME="entity_resolution"  # Change to your official team name for final package.
TEAM_MEMBERS=[]  # Add the actual team members before final packaging.
TARGET_HOURS=5.5
ENABLE_TAIL=True
BASE=Path(os.environ.get("GPU100_WORKING","/kaggle/working"))
ROOT=BASE/RUN_NAME
CODE=BASE/f"{RUN_NAME}_code"
DATA_ROOT=Path(os.environ.get("GPU100_DATA_ROOT","/kaggle/temp/gpu100_resource"))
OUTPUT=ROOT/"output"
ROOT.mkdir(parents=True,exist_ok=True);CODE.mkdir(parents=True,exist_ok=True)
START=time.monotonic()
print("Run directory:",ROOT)
print("Configuration:",dict(samples=SAMPLES,trees=TREES,epochs=EPOCHS,final_k=FINAL_K,
    batch_size=BATCH_SIZE,threads=THREADS,dimension=DIMENSION,cap_rate=CAP_RATE,
    channel_extra=CHANNEL_EXTRA))
print("Free working GB:",round(shutil.disk_usage(BASE).free/1e9,1))''')
md('''## 1. GPU and library check
The attention model and XGBoost ranker train on CUDA. Parsing and sparse retrieval use compiled CPU operations; the notebook reports both stage runtimes.''')
code('''if not SMOKE:
    subprocess.run(["nvidia-smi"],check=True)
    import torch
    assert torch.cuda.is_available(),"Enable a GPU accelerator in Kaggle Notebook Settings"
    print("Torch CUDA:",torch.version.cuda,"GPUs:",torch.cuda.device_count())
else:
    print("Local CPU smoke mode")
if not SMOKE:
    subprocess.run([sys.executable,"-m","pip","install","-q","--disable-pip-version-check",
                    "anyascii==0.3.3","sparse-dot-topn==1.1.5","rapidfuzz==3.14.3",
                    "polars==1.39.3","xgboost==3.0.5","matplotlib==3.10.7",
                    "tabm==0.0.3","rtdl_num_embeddings==0.0.12","transformers==4.56.2"],check=True)
import numpy,scipy,sklearn,xgboost,torch
if not SMOKE:
    try:
        import cupy
    except ImportError:
        cuda_family="13x" if torch.version.cuda and torch.version.cuda.startswith("13") else "12x"
        subprocess.run([sys.executable,"-m","pip","install","-q",f"cupy-cuda{cuda_family}==14.2.0"],check=True)
        import cupy
    assert cupy.cuda.runtime.getDeviceCount()>0,"CuPy cannot access the selected GPU"
print("Versions:",{"python":sys.version.split()[0],"numpy":numpy.__version__,"scipy":scipy.__version__,
                   "sklearn":sklearn.__version__,"xgboost":xgboost.__version__,"torch":torch.__version__,
                   "cupy":cupy.__version__ if not SMOKE else None})''')
md('''## 2. Install the source bundled in this notebook
The SHA-256 fingerprint ensures that code and checkpoints belong to the same run. Changing the notebook source/configuration requires a new `RUN_NAME`.''')
code(f'''SOURCE_SHA256={digest!r}
SOURCE_PAYLOAD={payload!r}
source_bytes=base64.b64decode(SOURCE_PAYLOAD)
assert hashlib.sha256(source_bytes).hexdigest()==SOURCE_SHA256
with zipfile.ZipFile(io.BytesIO(source_bytes)) as bundle:
    bundle.extractall(CODE)
manifest_path=ROOT/"source_manifest.json"
manifest=dict(source_sha256=SOURCE_SHA256,run_name=RUN_NAME,samples=SAMPLES,trees=TREES,
              epochs=EPOCHS,final_k=FINAL_K,batch_size=BATCH_SIZE,dimension=DIMENSION,
              cap_rate=CAP_RATE,channel_extra=CHANNEL_EXTRA,enable_tail=ENABLE_TAIL)
if manifest_path.exists():
    assert json.loads(manifest_path.read_text())==manifest,"Checkpoint settings changed; choose a new RUN_NAME"
else:
    manifest_path.write_text(json.dumps(manifest,indent=2))
(CODE/"run_config.json").write_text(json.dumps(manifest,indent=2))
os.environ["PYTHONPATH"]=str(CODE/"src")+os.pathsep+str(CODE)
os.environ["PYTHONUTF8"]="1"
if str(CODE/"src") not in sys.path:sys.path.insert(0,str(CODE/"src"))
print("Source SHA-256:",SOURCE_SHA256)''')
md('''## 3. Locate the supplied dataset
Attach the original archive as a private Kaggle Dataset. Kaggle may unpack ZIP uploads automatically; this cell accepts either the mounted archive or its extracted `student_resource/dataset` directory and fingerprints the mounted input before using checkpoints.''')
code('''if SMOKE:
    DATA=Path(os.environ["GPU100_SMOKE_DATA"])
    dataset_sha256=hashlib.sha256((DATA/"train/train_source1.tsv").read_bytes()).hexdigest()
else:
    candidate_zips=list(Path("/kaggle/input").rglob("6ab10eb3b23ba_student_resource.zip"))
    extracted=[p.parent.parent for p in Path("/kaggle/input").rglob("train_source1.tsv")
               if p.parent.name=="train" and p.parent.parent.name=="dataset"]
    if len(candidate_zips)==1:
        source_zip=candidate_zips[0]
        archive_hash=hashlib.sha256()
        with source_zip.open("rb") as archive_file:
            for block in iter(lambda:archive_file.read(4*1024*1024),b""):
                archive_hash.update(block)
        dataset_sha256=archive_hash.hexdigest()
        DATA=DATA_ROOT/"student_resource/dataset"
        marker=DATA_ROOT/"COMPLETE"
        if not marker.exists():
            DATA_ROOT.mkdir(parents=True,exist_ok=True)
            with zipfile.ZipFile(source_zip) as archive:
                for item in archive.infolist():
                    parts=Path(item.filename).parts
                    if not parts or parts[0]!="student_resource" or item.is_dir() or Path(item.filename).name.startswith("."):
                        continue
                    target=(DATA_ROOT/item.filename).resolve()
                    if not target.is_relative_to(DATA_ROOT.resolve()):raise ValueError("Unsafe ZIP path")
                    archive.extract(item,DATA_ROOT)
            marker.write_text(dataset_sha256)
        else:
            assert marker.read_text()==dataset_sha256,"Input archive changed; use fresh scratch directory"
    elif not candidate_zips and len(extracted)==1:
        DATA=extracted[0]
        input_hash=hashlib.sha256()
        for relative in ("train/train_source1.tsv","train/train_source2.tsv","train/train_source3.tsv",
                         "train/train_ground_truth.tsv","test/test_source1.tsv","test/test_source2.tsv","test/test_source3.tsv"):
            input_hash.update(relative.encode())
            with (DATA/relative).open("rb") as input_file:
                for block in iter(lambda:input_file.read(4*1024*1024),b""):
                    input_hash.update(block)
        dataset_sha256=input_hash.hexdigest()
    else:
        raise FileNotFoundError("Attach exactly one original student-resource archive or extracted Kaggle dataset")
dataset_manifest=ROOT/"dataset_manifest.json"
if dataset_manifest.exists():
    assert json.loads(dataset_manifest.read_text())["sha256"]==dataset_sha256,"Run checkpoint belongs to another dataset input"
else:
    dataset_manifest.write_text(json.dumps({"sha256":dataset_sha256},indent=2))
for required in ("train/train_source1.tsv","train/train_source2.tsv","train/train_source3.tsv",
                 "train/train_ground_truth.tsv","test/test_source1.tsv","test/test_source2.tsv","test/test_source3.tsv"):
    assert (DATA/required).is_file(),f"Missing {DATA/required}"
print("Dataset:",DATA)
print("Free scratch GB:",round(shutil.disk_usage(DATA_ROOT if DATA_ROOT.exists() else DATA).free/1e9,1))''')
md('''## 4. Stage runner
Completed stages and batches are saved under `ROOT`. Rerunning with the same input/settings skips completed chunks. The runtime printout makes the 5½-hour budget visible.''')
code('''DEVICE="cpu" if SMOKE else "cuda"
def run(stage):
    if (ROOT/"compaction.json").exists() and stage in ("prepare","preflight","raw","ranker","final","matcher","ownership","tail","collective","compact"):
        print(f"{stage}: training completed and caches compacted; resuming from saved models")
        return
    if stage in ("prepare","preflight","raw","ranker","final","matcher") and (ROOT/"calibration.json").exists() and (ROOT/"validation.json").exists():
        print(f"{stage}: completed model checkpoints found; skipping fitting work")
        return
    args=[sys.executable,"-u","-m","gpu100.runner",stage,"--data",str(DATA),"--root",str(ROOT),
          "--output",str(OUTPUT),"--samples",str(SAMPLES),"--trees",str(TREES),
          "--epochs",str(EPOCHS),"--final-k",str(FINAL_K),"--batch-size",str(BATCH_SIZE),
          "--threads",str(THREADS),"--dimension",str(DIMENSION),"--cap-rate",str(CAP_RATE),
          "--channel-extra",str(CHANNEL_EXTRA),
          "--device",DEVICE,"--team",TEAM_NAME,"--tail","on" if ENABLE_TAIL else "off"]
    if TEAM_MEMBERS:args.extend(["--members",*TEAM_MEMBERS])
    began=time.monotonic();subprocess.run(args,check=True,cwd=CODE)
    elapsed=(time.monotonic()-START)/3600
    print(f"{stage}: {(time.monotonic()-began)/60:.1f} min; run elapsed {elapsed:.2f} h; budget remaining {TARGET_HOURS-elapsed:.2f} h")''')
md('''## 5. Fit noisy channels and build indexes
The 70/15/15 entity split is frozen to the prior Colab baseline: its 9,000 held-out entities have 30,932 true links. Additional sampled entities enter only fitting. S1→S2 and S1→S3 edit/token tables are learned from fit positives, then original/sorted character and rare-word/number views are indexed by open country label. France flows through the same test path.''')
code('run("prepare")')
md('''## 6. Retrieval preflight and blocker ablation
A tuning sample searches **all** training targets. The notebook widens name/address search until recall reaches its target or the configured ceiling, then compares four character views against the added learned rare-word/number views. This measures the recall ceiling before any matcher score is quoted.''')
code('''run("preflight")
RETRIEVAL=json.loads((ROOT/"retrieval_config.json").read_text())
print(json.dumps(RETRIEVAL,indent=2))
if RETRIEVAL["trials"][-1]["candidate_recall"]<.98:
    print("WARNING: measured candidate recall is below 0.98. Final score cannot exceed this ceiling on the sampled training distribution.")''')
md('''## 7. Mine and checkpoint labeled candidate batches
The 120,000 sampled Source 1 entities are distinct; each searches the entire training target pool. A candidate record never becomes a positive by its ID pattern alone.''')
code('run("raw")')
md('''## 8. Train GPU ranker and expand from reliable anchors
The ranker uses 18 cheap features and serves as a prefilter. Up to two confident target matches each retrieve nearby Source 2/3 records; the final model sees this complete retained set. Exactly those scored records are written in `candidate_pairs.tsv`.''')
code('run("ranker");run("final")')
md('''## 9. Noisy-channel and hard-negative matcher ablations
The model comparison is controlled on one entity split: rich GPU tree without noisy-channel features, rich GPU tree with those features, TabM with random negatives, and TabM with the highest-scoring false candidates mined as hard negatives. Tuning chooses the threshold/model or blend; the original 9,000-entity holdout is reported separately. Trees checkpoint every 50 rounds and TabM every epoch.''')
code('''run("matcher")
REPORT=json.loads((ROOT/"validation.json").read_text())
print(json.dumps({k:REPORT[k] for k in ("holdout_macro_f05","candidate_recall","micro_precision",
    "micro_recall","singleton_accuracy","countries","selected")},indent=2))''')
md('''## 9a. Reciprocal ownership ablation
For each S2/S3 target, rank the competing S1 references in the generated candidate graph. The notebook measures reverse rank, mutual-top-k and ownership margin, then enables a conflict filter only if tuning macro F₀.₅ improves.''')
code('run("ownership")')
md('''## 9b. Multilingual cross-encoder tail
Only the most ambiguous 3% of reference entities reach the Apache-2.0 306M reranker. It trains on fit positives and mined hard negatives using classification and margin losses. The exact base-model revision is pinned; a tail adjustment is used only if tuning F₀.₅ improves without a material precision drop. The stage records any model-loading or GPU error and retains the audited base matcher.''')
code('run("tail")')
md('''## 9c. One conservative sibling pass
High-confidence S2/S3 matches provide evidence for borderline candidates already scored for that S1. One pass is enabled only if tuning F₀.₅ improves while protecting precision and singletons.''')
code('run("collective")')
md('''## 10. Reproducible figures
PNG/SVG files and their source `plot_data.json` are written to the run directory and later included in the package.''')
code('''run("figures")
from IPython.display import Image,display
display(Image(filename=str(ROOT/"figures/validation.png")))''')
md('''## 10a. Free training-only cache
After validation and model checkpoints are saved, free the large training matrices to keep `/kaggle/working` within its saved-space allowance. A rerun from the top skips fitting when these model checkpoints are complete.''')
code('run("compact")')
md('''## 11. Score the complete test set
The test corpus and country indexes build once. Each prediction batch saves the exact candidate target rows and probabilities. The progress line shows observed speed and estimated remaining time. Rerunning skips completed batches.''')
code('run("test");run("tail_apply")')
md('''## 12. Assemble, audit, and package
Both TSVs are required. The audit checks every Source 1 row, target existence and that each predicted match was scored as a candidate.''')
code('''run("assemble");run("audit");run("package")
PACKAGE=OUTPUT.parent/f"{TEAM_NAME}_submission.zip"
print("LEADERBOARD TSV:",OUTPUT/"matching_results.tsv")
print("FINAL PACKAGE:",PACKAGE)
print("TOTAL ELAPSED HOURS:",round((time.monotonic()-START)/3600,3))''')
md('''Upload `matching_results.tsv` to the leaderboard and retain the ZIP for the required code/methodology submission. A local holdout score is not a leaderboard score; send the actual public score and the notebook's validation JSON back for the next revision.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'accelerator':'GPU'})
nbf.validate(nb)
path=root/'CNER_Kaggle.ipynb'
nbf.write(nb,path)
print(path, 'cells',len(cells),'bytes',path.stat().st_size,'sha256',digest)
