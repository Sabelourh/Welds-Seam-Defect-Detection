import os
import sys
from pathlib import Path
import gradio as gr
import numpy as np
import cv2
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"src"))
from common import MODEL_PATH, build_model, eval_transform
from opencv_baseline import classify_rgb

COLORS={"Good Welding":(0,255,0),"Porosity":(255,165,0),"Crack":(255,0,0),"Spatters":(0,191,255)}
ICONS={"Good Welding":"🟢","Porosity":"🟠","Crack":"🔴","Spatters":"🔵"}

_model=None; _classes=None; _tf=None
def load_cnn():
    global _model,_classes,_tf
    if _model is None and MODEL_PATH.exists():
        ck=torch.load(MODEL_PATH,map_location="cpu")
        _classes=ck["class_names"]; _tf=eval_transform(ck.get("image_size",224))
        _model=build_model(len(_classes),pretrained=False)
        _model.load_state_dict(ck["model_state_dict"]); _model.eval()
    return _model

def analyze(image, opacity):
    if image is None: return None,"### Please take or upload a weld image."
    rgb=image.astype(np.uint8)
    model=load_cnn()
    if model is not None:
        pil=Image.fromarray(rgb).convert("RGB")
        with torch.no_grad():
            probs=torch.softmax(model(_tf(pil).unsqueeze(0)),dim=1)[0].numpy()
        idx=int(np.argmax(probs)); pred=_classes[idx]; conf=float(probs[idx])
        details="\n".join([f"- **{c}:** {100*p:.2f}%" for c,p in sorted(zip(_classes,probs),key=lambda z:-z[1])])
        method=f"**CNN ResNet18**  \n**Confidence:** {100*conf:.2f}%\n\n### Class probabilities\n{details}"
    else:
        pred=classify_rgb(rgb)
        method="**OpenCV fallback** — no trained `models/best_weld_model.pth` was found."
    color=COLORS.get(pred,(255,255,0))
    layer=np.zeros_like(rgb); layer[:]=color
    result=cv2.addWeighted(rgb,1-float(opacity),layer,float(opacity),0)
    banner=max(55,int(result.shape[0]*.10))
    cv2.rectangle(result,(0,0),(result.shape[1],banner),(0,0,0),-1)
    cv2.putText(result,pred,(15,int(banner*.68)),cv2.FONT_HERSHEY_SIMPLEX,
                max(.55,min(1.0,result.shape[1]/700)),color,2,cv2.LINE_AA)
    icon=ICONS.get(pred,"🔎")
    text=f"""# {icon} {pred}

{method}


"""
    return result,text

CSS="""
.gradio-container{max-width:900px!important;margin:auto!important}
@media(max-width:700px){.gradio-container{padding:8px!important}.mobile-row{display:block!important}}
"""
with gr.Blocks(title="Automated Weld Seam Defect Detection",css=CSS) as demo:
    gr.Markdown("""# 🔬 Automated Weld Seam Defect Detection
Take a photo with your phone camera or upload a weld image.

🟢 **Good Welding** · 🟠 **Porosity** · 🔴 **Crack** · 🔵 **Spatters**
""")
    inp=gr.Image(sources=["upload","webcam"],type="numpy",label="📷 Take Photo or Upload Weld Image")
    with gr.Accordion("⚙️ Display Settings",open=False):
        opacity=gr.Slider(.1,.8,.35,.05,label="Colour Overlay Opacity")
    btn=gr.Button("🔍 Analyze Weld",variant="primary")
    out=gr.Image(type="numpy",label="Classification Result")
    txt=gr.Markdown()
    btn.click(analyze,[inp,opacity],[out,txt])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )