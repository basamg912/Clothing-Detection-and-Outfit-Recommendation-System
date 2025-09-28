from PIL import Image, ImageOps
import numpy as np
import colorsys
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet
from torchvision.models import resnet50, ResNet50_Weights


class ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer

    def forward(self, x):
        x = self.backbone(x)  # Output shape: (B, 2048, 1, 1)
        return x.view(x.size(0), -1)  # Flatten to (B, 2048)
        # x.size(0) 은 Batch 사이즈는 유지하면서 자동으로 남은 (1,1) flatten 된 차원을 뜻하는 -1. → 을 삭제한다.

class ImageRelationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNetEmbedder()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img1, img2):
        emb1 = self.resnet(img1)
        emb2 = self.resnet(img2)
        concat = torch.cat([emb1, emb2], dim=1) # concat 했으니깐 총 4096차원
        return self.classifier(concat)

# 코디 추천 모델 
def make_model(model_weight):
    model = ImageRelationClassifier()
    model.load_state_dict(torch.load(model_weight))
    model.eval()
    return model

def recommend_cody(top, bottom, model):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    top_score = 0
    cody_top = None
    cody_bottom = None
    
    items = []
    for t in top:
        for j in bottom:
            img1 = Image.open(f"static/uploads/{t}")
            img2 = Image.open(f"static/uploads/{j}")
            tensor1 = transform(img1).unsqueeze(0)
            tensor2 = transform(img2).unsqueeze(0)
            score = model(tensor1, tensor2)[0].tolist()[1]
            if score > top_score:
                top_score = score
                cody_top = t
                cody_bottom = j
                items.append({"cody_top":cody_top, "cody_bottom":bottom, "score":score})
    # 코디를 score 순으로 정렬
    items.sort(key= lambda x : x['score'], reverse=True)
    return top_score, cody_top, cody_bottom, items
            
            
def classify_clothes(filepath,cls_model):
    pre_img = Image.open(filepath)
    pre_img = ImageOps.exif_transpose(pre_img)
    results = cls_model(pre_img)
    res = results[0]
    if hasattr(res, 'probs') and res.probs is not None:
        probs = res.probs.cpu().numpy()
        cls_id = int(probs.argmax)
        return res.names[cls_id]
    detections=[]
    seen = set()
    
    if len(res.boxes.cls) !=0:
        for i in range(len(res.boxes.cls)):
            cls_names = cls_model.model.names
            cls_name = cls_names[res.boxes.cls[i].item()]
            if cls_name in seen:
                continue
            if res.boxes[i].conf > 0.5:
                xy = res.boxes.xyxy[i].tolist()
                image = Image.open(filepath)
                x1,y1,x2,y2 = map(int,xy)
                cropped = pre_img.crop((x1,y1,x2,y2))
                detections.append((cropped,cls_name))
                seen.add(cls_name)
        return detections
    else:
        return []


def detect_color(image):
    image = image.resize((100, 100))
    np_img = np.array(image)

    h, w, _ = np_img.shape
    h1, h2 = int(h * 0.3), int(h * 0.7)
    w1, w2 = int(w * 0.3), int(w * 0.7)
    center_pixels = np_img[h1:h2, w1:w2].reshape(-1, 3)

    hsv_pixels = np.array([
        colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        for r, g, b in center_pixels
    ])

    h_vals = [h for h, s, v in hsv_pixels if s > 0.2 and v > 0.2]
    avg_h = np.mean(h_vals) if h_vals else 0.0
    avg_s = np.mean([s for h, s, v in hsv_pixels])
    avg_v = np.mean([v for h, s, v in hsv_pixels])

    h_deg = avg_h * 360

    # 무채색 분리
    if avg_s < 0.25 and avg_v > 0.7:
        return "white"
    elif avg_s < 0.2 and avg_v < 0.4:
        return "black"
    elif avg_s < 0.25:
        return "gray"

    # 색상 분류
    if 180 <= h_deg <= 250 and avg_s > 0.3 and avg_v < 0.4:
        return "navy"
    elif 200 <= h_deg <= 250 and avg_v >= 0.5:
        return "blue"
    elif 190 <= h_deg < 210 and avg_v > 0.7:
        return "skyblue"
    elif 30 <= h_deg <= 45 and avg_s < 0.5 and avg_v > 0.7:
        return "beige"
    elif 10 <= h_deg <= 25 and avg_s > 0.4 and avg_v < 0.6:
        return "brown"
    elif 55 <= h_deg <= 85 and avg_s < 0.5 and 0.4 <= avg_v <= 0.75:
        return "khaki"
    elif 290 <= h_deg <= 340:
        return "pink"
    elif 40 <= h_deg <= 65:
        return "yellow"
    elif 85 <= h_deg <= 160:
        return "green"

    return "unknown"
