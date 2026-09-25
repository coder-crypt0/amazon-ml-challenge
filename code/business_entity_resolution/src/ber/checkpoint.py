"""Optional durable checkpoints (e.g. mounted Google Drive), with atomic markers."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def remote_path(local):
    root=os.environ.get("BER_CHECKPOINT_ROOT")
    work=os.environ.get("BER_ARTIFACT_ROOT")
    if not root or not work:
        return None
    local=Path(local).resolve()
    work=Path(work).resolve()
    if not local.is_relative_to(work):
        return None
    return Path(root)/local.relative_to(work)


def persist_file(local):
    local=Path(local)
    remote=remote_path(local)
    if remote is None:
        return
    remote.parent.mkdir(parents=True,exist_ok=True)
    temp=remote.with_name(remote.name+".uploading")
    shutil.copyfile(local,temp)
    os.replace(temp,remote)


def restore_file(local):
    local=Path(local)
    if local.exists():
        return True
    remote=remote_path(local)
    if remote is None or not remote.is_file():
        return False
    local.parent.mkdir(parents=True,exist_ok=True)
    temp=local.with_name(local.name+".downloading")
    shutil.copyfile(remote,temp)
    os.replace(temp,local)
    return True


def persist_directory(local):
    local=Path(local)
    files=sorted(p for p in local.rglob("*") if p.is_file() and not p.name.endswith((".uploading",".downloading")))
    for p in files:
        if p.name!="COMPLETE":
            persist_file(p)
    if (local/"COMPLETE").is_file():
        persist_file(local/"COMPLETE")


def restore_directory(local):
    local=Path(local)
    remote=remote_path(local)
    if (local/"COMPLETE").exists():
        return True
    if remote is None or not (remote/"COMPLETE").is_file():
        return False
    for p in remote.rglob("*"):
        if p.is_file() and p.name!="COMPLETE" and not p.name.endswith((".uploading",".downloading")):
            restore_file(local/p.relative_to(remote))
    restore_file(local/"COMPLETE")
    return True


def ensure_config(path, config):
    path=Path(path)
    restore_file(path)
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError(f"Checkpoint configuration differs: {path}. Choose a new RUN_NAME/artifact directory.")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    persist_file(path)
