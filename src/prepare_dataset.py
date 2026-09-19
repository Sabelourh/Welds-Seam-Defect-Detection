from pathlib import Path
import shutil, random, yaml
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "classification"
TARGET = {"Good Welding", "Porosity", "Crack", "Spatters"}
IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}

def find_yaml():
    ys = list(RAW.rglob("data.yaml")) + list(RAW.rglob("*.yaml"))
    if not ys:
        raise FileNotFoundError("No YOLO data.yaml found under data/raw.")
    return ys[0]

def load_names(yaml_path):
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k,v in names.items()}
    if isinstance(names, list):
        return {i:str(v) for i,v in enumerate(names)}
    raise ValueError("Could not read class names from YAML.")

def image_for_label(label):
    # Common YOLO layout: labels/.../x.txt -> images/.../x.jpg
    parts = list(label.parts)
    candidates = []
    if "labels" in parts:
        i = parts.index("labels")
        p2 = parts.copy()
        p2[i] = "images"
        base = Path(*p2).with_suffix("")
        candidates += [base.with_suffix(e) for e in IMG_EXT]
    candidates += [label.with_suffix(e) for e in IMG_EXT]
    for c in candidates:
        if c.exists():
            return c
    # Last-resort basename search
    for c in RAW.rglob(label.stem + ".*"):
        if c.suffix.lower() in IMG_EXT:
            return c
    return None

def infer_split(path):
    low = [p.lower() for p in path.parts]
    if "test" in low:
        return "test"
    if "valid" in low or "val" in low:
        return "val"
    if "train" in low:
        return "train"
    return None

def main():
    yml = find_yaml()
    names = load_names(yml)
    print("Using annotations:", yml)
    print("Classes:", names)

    records = []
    for label in RAW.rglob("*.txt"):
        if label.name.lower() in {"classes.txt"}:
            continue
        try:
            lines = [x.strip() for x in label.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            ids = {int(x.split()[0]) for x in lines if len(x.split()) >= 5}
        except Exception:
            continue
        labels = {names[i] for i in ids if i in names}
        target_labels = labels & TARGET

        # Keep only clean single-label target images.
        if len(target_labels) != 1:
            continue
        cls = next(iter(target_labels))
        img = image_for_label(label)
        if img is None:
            continue
        records.append((img, cls, infer_split(label)))

    if not records:
        raise RuntimeError("No usable single-label images found. Inspect data/raw and data.yaml.")

    # If source has no split, deterministic stratified 70/15/15 split.
    if all(r[2] is None for r in records):
        random.seed(42)
        by_class = defaultdict(list)
        for img, cls, _ in records:
            by_class[cls].append(img)
        records = []
        for cls, imgs in by_class.items():
            random.shuffle(imgs)
            n=len(imgs); a=int(.70*n); b=int(.85*n)
            records += [(x,cls,"train") for x in imgs[:a]]
            records += [(x,cls,"val") for x in imgs[a:b]]
            records += [(x,cls,"test") for x in imgs[b:]]
    else:
        # Normalize missing/valid split conservatively.
        records = [(img,cls,(split or "train")) for img,cls,split in records]

    if OUT.exists():
        shutil.rmtree(OUT)
    counts = defaultdict(int)
    for img, cls, split in records:
        split = "val" if split == "valid" else split
        dest = OUT / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        name = f"{img.stem}_{abs(hash(str(img))) % 10**8}{img.suffix.lower()}"
        shutil.copy2(img, dest/name)
        counts[(split,cls)] += 1

    print("\nPrepared classification dataset:")
    for k in sorted(counts):
        print(k, counts[k])
    print("Output:", OUT)

if __name__ == "__main__":
    main()
