import json, sys
from pathlib import Path
import pandas as pd, numpy as np, cv2, torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from common import DATA_DIR, MODEL_PATH, RESULTS_DIR, build_model, eval_transform
from opencv_baseline import classify_rgb

def main():
    ck=torch.load(MODEL_PATH,map_location="cpu"); classes=ck["class_names"]
    model=build_model(len(classes),pretrained=False); model.load_state_dict(ck["model_state_dict"]); model.eval()
    tf=eval_transform(ck.get("image_size",224))
    rows=[]
    for cls in classes:
        for p in (DATA_DIR/"test"/cls).iterdir():
            if not p.is_file(): continue
            pil=Image.open(p).convert("RGB"); rgb=np.array(pil)
            with torch.no_grad():
                pred=classes[model(tf(pil).unsqueeze(0)).argmax(1).item()]
            op=classify_rgb(rgb)
            rows.append({"file":str(p),"true":cls,"opencv":op,"cnn":pred})
    df=pd.DataFrame(rows); df.to_csv(RESULTS_DIR/"opencv_vs_cnn.csv",index=False)
    out={}
    for name in ["opencv","cnn"]:
        acc=accuracy_score(df.true,df[name])
        pr,re,f1,_=precision_recall_fscore_support(df.true,df[name],average="weighted",zero_division=0)
        out[name]={"accuracy":acc,"weighted_precision":pr,"weighted_recall":re,"weighted_f1":f1}
    (RESULTS_DIR/"opencv_vs_cnn_summary.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
