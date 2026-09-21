import shutil
from pathlib import Path

rejected_dir = Path("dataset/rejected")
raw_dir = Path("dataset/raw")

# 1. Restore bachaur, khillar, kosali, ghumusari to raw
restores = [
    ("low_resolution/bachaur_0001.jpg", "cattle/bachaur/bachaur_0001.jpg"),
    ("low_resolution/khillar_0001.jpg", "cattle/khillar/khillar_0001.jpg"),
    ("low_resolution/kosali_0001.jpg", "cattle/kosali/kosali_0001.jpg"),
    ("low_resolution/ghumusari_0001.png", "cattle/ghumusari/ghumusari_0001.png")
]

for src_rel, dest_rel in restores:
    src = rejected_dir / src_rel
    dest = raw_dir / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.move(str(src), str(dest))
        print(f"Restored {src_rel} -> {dest_rel}")

# 2. Restore Bargur cattle images
for i in range(1, 5):
    dup_file = rejected_dir / "duplicates" / f"bargur_{i:04d}.jpg"
    cat_dest = raw_dir / "cattle" / "bargur" / f"bargur_{i:04d}.jpg"
    if dup_file.exists():
        shutil.move(str(dup_file), str(cat_dest))
        print(f"Restored Bargur cattle: {cat_dest.name}")
