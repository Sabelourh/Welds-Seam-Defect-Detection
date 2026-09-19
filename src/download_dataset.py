from pathlib import Path
import shutil
import kagglehub

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw"
SLUG = "sukmaadhiwijaya/weld-quality-inspection-instance-segmentation"

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print("Downloading real CC0 weld dataset from Kaggle...")
    cache_path = Path(kagglehub.dataset_download(SLUG))
    print("Kaggle cache:", cache_path)
    for item in cache_path.iterdir():
        target = DEST / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print("Dataset copied to:", DEST)

if __name__ == "__main__":
    main()
