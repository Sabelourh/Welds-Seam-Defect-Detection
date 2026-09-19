import argparse, copy, csv
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from common import DATA_DIR, MODEL_PATH, RESULTS_DIR, build_model, train_transform, eval_transform

def run_epoch(model, loader, loss_fn, optimizer, device, train):
    model.train(train)
    total_loss=correct=n=0
    with torch.set_grad_enabled(train):
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            if train: optimizer.zero_grad()
            logits=model(x); loss=loss_fn(logits,y)
            if train: loss.backward(); optimizer.step()
            total_loss += loss.item()*x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            n += x.size(0)
    return total_loss/n, correct/n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=16)
    ap.add_argument("--lr",type=float,default=1e-4)
    args=ap.parse_args()

    train_ds=ImageFolder(DATA_DIR/"train", transform=train_transform())
    val_ds=ImageFolder(DATA_DIR/"val", transform=eval_transform())
    if train_ds.classes != val_ds.classes:
        raise ValueError(f"Train/val classes differ: {train_ds.classes} vs {val_ds.classes}")
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val_dl=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=0)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device,"Classes:",train_ds.classes)
    model=build_model(len(train_ds.classes), pretrained=True).to(device)
    loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=-1; best_state=None; hist=[]

    for epoch in range(1,args.epochs+1):
        tr_l,tr_a=run_epoch(model,train_dl,loss_fn,opt,device,True)
        va_l,va_a=run_epoch(model,val_dl,loss_fn,opt,device,False)
        hist.append([epoch,tr_l,tr_a,va_l,va_a])
        print(f"Epoch {epoch:02d}/{args.epochs} train loss={tr_l:.4f} acc={tr_a:.4f} val loss={va_l:.4f} acc={va_a:.4f}")
        if va_a>best:
            best=va_a; best_state=copy.deepcopy(model.state_dict())

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({"model_state_dict":best_state,"class_names":train_ds.classes,
                "image_size":224,"best_val_accuracy":best},MODEL_PATH)

    with open(RESULTS_DIR/"training_history.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","train_accuracy","val_loss","val_accuracy"]); w.writerows(hist)

    e=[x[0] for x in hist]
    plt.figure(); plt.plot(e,[x[2] for x in hist],label="Train"); plt.plot(e,[x[4] for x in hist],label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR/"training_accuracy.png",dpi=160); plt.close()
    plt.figure(); plt.plot(e,[x[1] for x in hist],label="Train"); plt.plot(e,[x[3] for x in hist],label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR/"training_loss.png",dpi=160); plt.close()
    print("Saved real trained checkpoint:",MODEL_PATH)

if __name__=="__main__": main()
