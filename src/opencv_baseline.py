import cv2, numpy as np

def classify_rgb(image, threshold=80, minimum_area=20):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    gray=clahe.apply(gray)
    _,binary=cv2.threshold(gray,int(threshold),255,cv2.THRESH_BINARY_INV)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=gray.shape[:2]; scores={"Porosity":0,"Crack":0,"Spatters":0}; valid=0
    for c in contours:
        area=cv2.contourArea(c)
        if area<minimum_area or area>.20*h*w: continue
        per=cv2.arcLength(c,True)
        if per<=0: continue
        x,y,bw,bh=cv2.boundingRect(c)
        if bw==0 or bh==0: continue
        valid+=1
        circ=4*np.pi*area/(per**2)
        ar=max(bw/bh,bh/bw)
        if circ>=.55: scores["Porosity"]+=1
        elif ar>=3: scores["Crack"]+=1
        else: scores["Spatters"]+=1
    return "Good Welding" if valid==0 else max(scores,key=scores.get)
