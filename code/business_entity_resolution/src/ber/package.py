"""Create a reproducible submission archive without bundling the dataset."""
from pathlib import Path
import os
import zipfile

_SKIP_DIRS = {"__pycache__", ".git", ".venv", "data", "dataset"}
_SKIP_NAMES = {".env", ".envrc", "credentials", "secrets.json"}

def package_submission(destination, output_dir, project_root, documentation_path):
    destination = Path(destination); root = Path(project_root)
    out = Path(output_dir); doc = Path(documentation_path)
    for required in (out/"matching_results.tsv",out/"candidate_pairs.tsv",doc,root/"README.md",root/"requirements.txt"):
        if not required.is_file():
            raise FileNotFoundError(required)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def add_tree(z, base, arcbase):
        for p in base.rglob("*"):
            if not p.is_file(): continue
            rel = p.relative_to(base)
            if any(x in _SKIP_DIRS for x in rel.parts) or p.name in _SKIP_NAMES: continue
            if p.suffix in {".pyc", ".pyo"}: continue
            z.write(p, (arcbase / rel).as_posix())
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv","candidate_pairs.tsv"):
            z.write(out/name,f"output/{name}")
        src = root / "src"
        if src.exists(): add_tree(z, src, Path("code/business_entity_resolution/src"))
        for name in ("README.md", "requirements.txt", "LICENSE", "run_config.json"):
            p = root / name
            if p.is_file(): z.write(p, f"code/business_entity_resolution/{name}")
        if doc.is_file(): z.write(doc, "Documentation_template.md")
        if (root/"reports").is_dir():
            add_tree(z,root/"reports",Path("code/business_entity_resolution/reports"))
    return destination
