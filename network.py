import cv2
import numpy as np
import torch
import random
import json
import os
import torch.nn as nn


'''
Search Region = Template * search_factor (generally, 4)
s_search = s x search_factor

def temp_and_search와 같이 단순 crop하면 
object aspect ratio가 변하면서 context가 불안정해짐.
'''

'''
##################################################################
12, March, To-do List

1. Data Folder -> Import IR_label.json -> Find Bounding Box in first frame.
2. ** (Vis) BBox --(input)-> Template/Search Region Cropping
3. Cropped SR & T  --(input)--> Linear Projection (DINOv2)
4. ** Token Type Embedding : Template / Serach / Memory (Not to include the first frame.)
    5.1 Memory Token --> Memory Attention Layer
    5.2 Search / Template -> Transformer Encoder

    6. Modeling Transformer Encoder & MAL

##################################################################
'''


USER_PATH = os.path.expanduser("~")

ROOT_DIR = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MemLoTrack", "Anti-UAV410")


import torch
import torchvision.models as models

dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
backbone = dinov2_vitb14_reg


class DINOv2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = backbone
        for param in backbone.parameters():
            param.requires_grad = False  # Backbone Freezing

    def forward(self, x):
        feats = self.backbone.forward_features(x)
        tokens=feats['x_norm_patchtokens'] # PatchEmbedding, [B, N, C]

        return tokens
    
        
class MemoryBank: # Only During Inference
    def __init__(self, max_size=7):
        self.max_size = max_size

    def add(self, tokens):
        memory = []

        # Dual Gate 필요

        memory.append(tokens)
        # remove 로 제일 첫 번째 요소를 없애기 
        # pop() : 제일 마지막꺼


class MemorySampling(nn.Module): # Only During Training
    def __init__(self, max_size=7):
        super().__init__()
        self.max_size = max_size

    def forward(self, template_idx, search_idx, frame):
        candidates = frame[template_idx + 1 : search_idx]
        num_candidatees = len(candidates)

        sampled_memory = []

        if num_candidatees >= self.max_size:
            sampled_memory = random.sample(candidates, self.max_size)

        elif num_candidatees > 0:
            sampled_memory = list(candidates)
            empty_size = self.max_size - len(sampled_memory)
            last_frame = candidates[-1]
            sampled_memory.extend([last_frame] * empty_size)
        
        else:  # No Interval : Full with Template frame
            template_frame = frame[template_idx]
            sampled_memory = [template_frame] * self.max_size

        return sampled_memory


'''
- Positional Embedding  (CLS Token X)
- Type Embedding (Template / Search / Memory)
- memory token = token + memory_type_embedding
- **Final Memory Token = Patch_embedding (DINOv2) + Positional Embedding (DINOv2)
- Save into Memory Bank


'''
class TypeEmbedding(nn.Module):
    # Input Type and shape of memory tokens in current step.
    def __init__(self, embed_dim=768):
    # Template / Search / Memory
        super().__init__()

        self.template_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Template -> Target + Background 두 개로 나뉨
        self.search_token = nn.Parameter(torch. randn(1, 1, embed_dim)* 0.02)
        self.memory_token = nn.Parameter(torch. randn(1, 1, embed_dim) * 0.02)

    def forward(self, x, token_type, target_mask = None):
        # x : [B, N, 768] (DINO 통과함)
        if token_type == 'search':
            return x + self.search_token
        elif token_type == "memory":
            return x + self.memory_token
        elif token_type == "template":
            return x + self.template_token
            # 수정 필요함



#class MAL:


class KalmanGate(object):


class TrackerDINOv2(object):
    def __init__(self, backbone):
        super().__init__()

        self.model = backbone
        self.template = None
        self.memory_bank = []


    def parse_args(self):
        pass
    
    def initialize_tracking(self):
        pass

    def init(self, image, model):
        pass

    def init_with_bbox(self, frame, bbox):
        self.template = self.extract_template(frame, bbox)
        self.prev_bbox = bbox

    def update(self, frame):
        search = self.extract_search(frame, self.prev_bbox)
        
        # Feature extraction 
        # transformer
        # bbox prediction  


        self.prev_bbox = pred_bbox

        return pred_bbox

    # def _crop_and_resize(self):
    #     pass

    def extract_search(self):
        pass

    def SiamFC_crop(self, img, cx, cy, size, out_size):
        # Tracking에서 사용할 template / Search patch를 이미지에서 추출
        # cv2.getRectSubPix() : 중심 좌표 기준으로 sub-pixel crop을 수행하는 함수
        # 일반 crop은 center 기준 crop이 어려움. -> image boundary 처리 필요함

        patch = cv2.getRectSubPix(
            img,
            (int(size), int(size)),
            (cx, cy)
        ) # bbox 중심 기준으로 crop
        patch = cv2.resize(patch, (out_size, out_size)) # 모델 입력 크기로 resize

        return patch

    def extract_template(self, frame, bbox):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        s_z = SiamFC_crop_size(w, h)
        template = self.SiamFC_crop(frame, cx, cy, s_z, 112)

        return template













def SiamFC_crop_size(w, h, context=0.5):
    p = (w + h) / 2 * context # contect padding : 배경 정보를 얻기 위함이며, appearance와 scale 변화에 강해지기 위해
    s = np.sqrt((w+p) * (h+p))

    ## gotta prevent not to let crop area leave the image area.


    return s

template_size_siam = SiamFC_crop_size(w, h)
search_region_siam = template_size_siam * 4

def SiamFC_PatchExtract(frame, bbox):
    
    SiamFC_crop_size()
    pass



def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or (len(pred) == 0)

def get_bbox(label_res):
    measure_per_frame = []
    for _gt, _exist in zip(label_res['gt_rect'], label_res['exist']):
        # Target 존재 x : _exist==False | Target 존재 O : _exist == True
        measure_per_frame.append(not_exist(_pred) if not _exist else)
