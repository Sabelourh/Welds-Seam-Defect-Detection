from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"/"classification"
RES = ROOT/"results"; RES.mkdir(exist_ok=True)

def main():
    rows=[]
    for split in ["train","val","test"]:
        base=DATA/split
        if not base.exists(): continue
        for cls in sorted([p for p in base.iterdir() if p.is_dir()]):
            n=sum(1 for p in cls.iterdir() if p.is_file())
            rows.append({"split":split,"class":cls.name,"count":n})
    if not rows: raise RuntimeError("Run prepare_dataset.py first.")
    df=pd.DataFrame(rows)
    df.to_csv(RES/"dataset_counts.csv", index=False)
    pivot=df.pivot(index="class",columns="split",values="count").fillna(0)
    ax=pivot.plot(kind="bar", figsize=(10,6))
    ax.set_ylabel("Images"); ax.set_title("Weld dataset class distribution")
    plt.tight_layout(); plt.savefig(RES/"class_distribution.png", dpi=160); plt.close()

    classes=sorted((DATA/"train").iterdir())
    fig, axes=plt.subplots(len(classes),3,figsize=(9,3*len(classes)))
    if len(classes)==1: axes=[axes]
    random.seed(42)
    for r,cdir in enumerate(classes):
        imgs=[p for p in cdir.iterdir() if p.is_file()]
        for j in range(3):
            ax=axes[r][j]
            if imgs:
                p=random.choice(imgs)
                ax.imshow(Image.open(p).convert("RGB"))
            ax.axis("off")
            if j==0: ax.set_title(cdir.name)
    plt.tight_layout(); plt.savefig(RES/"sample_images.png", dpi=160); plt.close()
    print(df.to_string(index=False))
    totals=df.groupby("class")["count"].sum()
    print("\nImbalance ratio max/min:", round(totals.max()/max(totals.min(),1),2))

if __name__=="__main__": main()
