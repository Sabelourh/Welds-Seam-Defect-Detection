import json
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from common import DATA_DIR, MODEL_PATH, RESULTS_DIR, build_model, eval_transform

def main():
    ck=torch.load(MODEL_PATH,map_location="cpu")
    classes=ck["class_names"]
    ds=ImageFolder(DATA_DIR/"test",transform=eval_transform(ck.get("image_size",224)))
    if ds.classes!=classes: raise ValueError(f"Checkpoint/test classes differ: {classes} vs {ds.classes}")
    dl=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=build_model(len(classes),pretrained=False).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    yt=[]; yp=[]
    with torch.no_grad():
        for x,y in dl:
            p=model(x.to(device)).argmax(1).cpu()
            yt.extend(y.tolist()); yp.extend(p.tolist())
    acc=accuracy_score(yt,yp)
    pr,re,f1,_=precision_recall_fscore_support(yt,yp,average="weighted",zero_division=0)
    metrics={"accuracy":acc,"weighted_precision":pr,"weighted_recall":re,"weighted_f1":f1,
             "test_images":len(ds),"classes":classes}
    (RESULTS_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    report=classification_report(yt,yp,target_names=classes,zero_division=0)
    (RESULTS_DIR/"classification_report.txt").write_text(report)
    cm=confusion_matrix(yt,yp)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm)
    ax.set_xticks(range(len(classes)),classes,rotation=45,ha="right")
    ax.set_yticks(range(len(classes)),classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("CNN Confusion Matrix")
    for i in range(len(classes)):
        for j in range(len(classes)): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(im); plt.tight_layout(); plt.savefig(RESULTS_DIR/"confusion_matrix.png",dpi=180); plt.close()
    print(json.dumps(metrics,indent=2)); print(report)

if __name__=="__main__": main()
