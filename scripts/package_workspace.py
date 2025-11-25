# scripts/package_workspace.py
# packaging script (from notebook cell that zips workspace artifacts)
import shutil
from pathlib import Path
from src.config import WORKDIR, ZIPPATH, ARTIFACTS_DIR

def package_workspace():
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(ZIPPATH.with_suffix('')), 'zip', root_dir=str(WORKDIR))
    print("Packaged artifacts to:", ZIPPATH, " size:", ZIPPATH.stat().st_size, "bytes")
    print("Sample files:")
    for i, p in enumerate(sorted(Path(WORKDIR).rglob("*"))):
        if i > 40:
            break
        print(" -", p.relative_to(WORKDIR))

if __name__ == "__main__":
    package_workspace()
