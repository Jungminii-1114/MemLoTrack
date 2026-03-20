import cv2
import numpy as np
from typing import Dict, List, Tuple



def extract_template(frame, bbox, transform):
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    s_z = SiamFC_crop_size(w, h, context=0.5)
    patch = SiamFC_crop(frame, cx, cy, s_z, 112)
    
    template = transform(patch).unsqueeze(0)
    return template

def extract_search(frame, prev_bbox, transform):
    x, y, w, h = prev_bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    '''
    위 세 줄 : bbox_crop
    '''

    s_z = SiamFC_crop_size(w, h, context=0.5)
    s_x = s_z * 2.0  # 두 배 해서 면적 4배

    patch = SiamFC_crop(frame, cx, cy, s_x, out_size=224)
    search_tensor = transform(patch).unsqueeze(0)
    return search_tensor

def SiamFC_crop_size(w, h, context=0.5):
    p = (w + h) / 2.0 * context # contect padding : 배경 정보를 얻기 위함이며, appearance와 scale 변화에 강해지기 위해
    ## gotta prevent not to let crop area leave the image area.
    val = max(0.0, (w + p) * (h + p))
    s_z = float(np.sqrt(val))
    return max(16.0, s_z)


def SiamFC_crop(img, cx, cy, size, out_size):
    # Tracking에서 사용할 template / Search patch를 이미지에서 추출
    # cv2.getRectSubPix() : 중심 좌표 기준으로 sub-pixel crop을 수행하는 함수
    # 일반 crop은 center 기준 crop이 어려움. -> image boundary 처리 필요함

    
    if img is None or img.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)

    # 🛡️ 2차 입구컷: 정답지(좌표)가 썩었으면 까만 화면 반환!
    if np.isnan(cx) or np.isnan(cy) or np.isnan(size):
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
        
    size = max(16.0, float(size))

    patch = cv2.getRectSubPix(
        img,
        (int(size), int(size)),
        (float(cx), float(cy))
        
    ) # bbox 중심 기준으로 crop
    # 오류 발생했었음 : OpenCV가 받은 좌표가 이상할 때 빈걸로 채워서 오류방지
    if patch is None or patch.size==0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    
    patch = cv2.resize(patch, (out_size, out_size), interpolation = cv2.INTER_LINEAR) # 모델 입력 크기로 resize

    return patch





def prediction_2_Box(score_map, regression_map):
    # (아까 고친 완벽한 해독 로직 그대로, self만 빼고 넣으시면 됩니다)
    B, _, H, W = score_map.shape
    assert B == 1, "PredictionHead의 output은 반드시 1개의 배치만 있어야 함."
    #flatten_scoreMap = score_map.reshape(B, -1)
    flatten_scoreMap = score_map.view(B, -1)
    conf, idx = flatten_scoreMap.max(dim=1)
    
    #max_score_pos = torch.argmax(flatten_scoreMap, dim=1).item() # return Position of Max Score
    #max_score_pos = flatten_scoreMap.argmax(dim=1) # B = 1 일 경우에는 .item() 해도 좋지만, 1보다 클 경우 첫 배치만 강제로 스칼라로 만들기 때문에 나머지 배치 정보 손실됨.
    max_score_pos = int(idx[0].item())
    confidence_score = float(conf[0].item())


    '''
    16 x 16에서 생각해보면 0~15 idx까지 row 인것 생각하기  
        xxxxxxxxxxxxxxxxx
    + 패치가 16x16 사이즈인 것이랑, 실제 원본 이미지 224x224 차이 생각하기
    '''

    grid_x = max_score_pos % W # W : 16
    grid_y = max_score_pos // W # W : 16
    
    l = float(regression_map[0, 0, grid_y, grid_x].item()) * 14.0
    t = float(regression_map[0, 1, grid_y, grid_x].item()) * 14.0
    r = float(regression_map[0, 2, grid_y, grid_x].item()) * 14.0
    b = float(regression_map[0, 3, grid_y, grid_x].item()) * 14.0

    px = grid_x * 14.0 + 7.0    
    py = grid_y * 14.0 + 7.0

    w = l + r
    h = t + b
    x = px - l
    y = py - t

    return confidence_score, [x, y, w, h]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
    return float(iou)