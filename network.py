import cv2
import numpy as np
import torch
import random
import math
import json
import os
import torch.nn as nn


'''
Search Region = Template * search_factor (generally, 4) -> 4배를 하는게 면적이 4배, 가로 / 세로는 각 두 배씩 
s_search = s x search_factor

def temp_and_search와 같이 단순 crop하면 
object aspect ratio가 변하면서 context가 불안정해짐.
'''

'''
##################################################################

`MemEffAttention` Module in DINOv2 : An optimized implementation of self-attention mechanism designed for memory efficiency.
particularly beneficial for precessing high-resolution images or large batch sizes in CV Tasks.

Key Features and Requirements
- Memory Optimization 
: The primary purpose of `MemEffAttention` is to reduce the memory footprint compared to standard attention moduels. It achieves this by not materializing the full attention matrix during computation, instead calculating 
the output more efficiently. This makes DINOv2 models capable of handling larger inputs, like those in depth estimation or semantic segmentation tasks, with less VRAM usage.


- xFormers Dependency

- Attnetion Map Visualization Limitation 
: As `MemEffAttention` avoids explicitly computing the full attention matrix, It doesn't easily allow for the extraction and visualization of attention heatmaps, which is a common technique for interpreting model behavior (e.g., saliency maps)
To Visualize attention, developers must switch to the standard Attnetion module implementation within the DINOv2 framework, which does produce the intermediate attnetino matrix.

##################################################################
'''


USER_PATH = os.path.expanduser("~")

ROOT_DIR = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MemLoTrack", "Anti-UAV410")


import torch
import torchvision.models as models

dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
backbone = dinov2_vitb14_reg

LoRAConfig={
    "r" : 64,
    "lora_alpha" : 8,
    "lora_dropout":0.06,
    "bias" : "none",
    "target_modules":["qkv", "fc1", "fc2"]
}

class DINOv2(nn.Module):
    def __init__(self, type_embedder, config):
        super().__init__()
        
        self.backbone = backbone
        for param in backbone.parameters():
            param.requires_grad = False  # Backbone Freezing

        self.type_embedder = type_embedder
        self.config = config

        for block in self.backbone.blocks:

            original_qkv = block.attn.qkv
            block.attn.qkv = LoRALinear(original_qkv, config=self.config)

            original_fc1 = block.mlp.fc1
            block.mlp.fc1 = LoRALinear(original_fc1, config=self.config)

            original_fc2 = block.mlp.fc2
            block.mlp.fc2 = LoRALinear(original_fc2, config=self.config)

    def get_initial_tokens(self, x):
        # Patch embedding
        tokens = self.backbone.patch_embed(x)
        
        # Positional Embeding
        pos_embed = self.backbone.pos_embed[:, 1:, :]
        return tokens + pos_embed

    def forward(self, z_img, x_img, target_mask):
        z_tokens = self.get_initial_tokens(z_img) # [B, 64, 768]
        x_tokens = self.get_initial_tokens(x_img) # [B, 256, 768] 

        z_tokens = self.type_embedder(z_tokens, token_type = 'template', target_mask = target_mask) 
        x_tokens = self.type_embedder(x_tokens, token_type = "search")

        tokens_concaten = torch.concat([z_tokens, x_tokens], dim=1)

        for block in self.backbone.blocks: # Transformer Encoder (12 layers)
            tokens_concaten = block(tokens_concaten)
        tokens_concaten = self.backbone.norm(tokens_concaten)
        
        search_token_enc = tokens_concaten[:, 64:, :] # Search만 빼오기

        return search_token_enc


class LoRALinear(nn.Module):
    # LORA Adapter는 Trainable하게 !! 
    def __init__(self, original_layer, config):
        super().__init__()
        #self.original_qkv = original_qkv
        self.config = config

        self.original_layer = original_layer
        self.rank = config['r']
        self.lora_alpha = config['lora_alpha']
        self.lora_dropout = nn.Dropout(config['lora_dropout'])
        self.bias = config['bias']
        self.target_modules = config['target_modules']
        self.scaling = self.alpha / self.rank

        in_features = self.original_layer.in_features
        out_features = self.original_layer.out_features

        self.LoRA_A = nn.Parameter(torch.zeros(in_features, self.rank))
        self.LoRA_B = nn.Parameter(torch.randn(self.rank, out_features))

        nn.init.kaiming_uniform_(self.LoRA_A, a=math.sqrt(5))
        nn.init.zeros_(self.LoRA_B)

    def forward(self, original_layer, x):
        if original_layer == "original_qkv":
            # LORA Adapter 적용된 qkv 계산
            self.original_qkv = original_layer
        elif original_layer == "original_fc1":
            self.original_fc1 = original_layer
        elif original_layer == "original_fc2":
            self.original_fc2 = original_layer

        original_output = self.original_layer(x)
        LoRA_output = torch.matmul(torch.matmul(x, self.LoRA_A), self.LoRA_B) * self.scaling
        
        return original_output + LoRA_output




'''
Model Head
|
|___ Classification MLP
|           |____ Role : 각 셀에 타겟 있을 확률 confidence 계산
|           |____ Role : 어떤 상자가 진짜 타겟 상자인가 (최댓값)
|           |____ Role : 최댓값의 셀에 Regression MLP의 거리 값을 이용해 BB 생성
|___ Regression MLP
|           |____ Role : 각 셀의 중심을 기준으로 4방향 거리 계산 
|           |____ Role : Kalman Filter 예측으로 자연스러운지 Consistency 계산


'''

def generate_gaussian_target(size=16, center=(7, 3), sigma=2.0):
    cx, cy = center

    result = torch.zeros((size, size), dtype=torch.float32)

    for i in range(size):
        for j in range(size):
            
            dist_sq = (i - cx) ** 2 + (j - cy) ** 2
            result[i, j] = math.exp(-dist_sq / (2 * 1.5**2))

    return result

class TargetGenerateor(object):
    def __init__(self, json_file):
        super().__init__()
        with open(json_file, "r") as f:
            label = json.load(f)
        
        self.exist = label['exist']
        self.gt_rect = label['gt_rect']
        self.total_frames = len(label['exist'])

    def get_target(self, frame):
        if self.exist[frame] == 0:
            return torch.zeors((16, 16))
        x, y, w, h = self.gt_rect[frame]
        cx = x + w/2
        cy = y + h/2

        grid_cx = int(cx/14)
        grid_cy = int(cy/14)
        target_heatmap = generate_gaussian_target(heatmap_size=16, center=(grid_cx, grid_cy), sigma=1.5)
        return target_heatmap



class PredictionHead(nn.Module): # [Batch, 768, 16, 16] -> [Batch, 192, 16, 16]
    def __init__(self, in_channels=768, hidden_channels=192): # 256으로도 해보기
        super().__init__()
        self.ClassificationMLP = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1), # [B, 192, 16, 16] => [B, 1, 16, 16] 1채널만 필요함 (점수값)
            nn.Sigmoid()
        )

        self.RegressionMLP = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, 256, 768] -> [B, 768, 256]
        B, N, C=x.shape
        x = x.contiguous().view(B, 768, 16, 16) # [B, 768, 256] -> [B, 768, 16, 16]

        score_map = self.ClassificationMLP(x) # Output
        size_map = self.RegressionMLP(x) 


        return score_map, size_map

class KalmanGate(object):
    # 1. 드론의 현재 상태 벡터로 정의 ->6차원 벡터 
    # 2. Prediction Head가 측정한 z_t


class MemoryBank: # Only During Inference
    def __init__(self, max_size=7):
        self.max_size = max_size
        self.memory= []

    def add(self, tokens, confidence_score):
        # Dual Gate 필요
        # (1) Confidence Score : using maximum score 


        # (2) Motion consistency : Kalman-based Mahalanobis test

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

class MAL(nn.Module):
    def __init__(self, embed_dim = 768, num_heads=12):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.MLP = nn.Sequential( # Ordinary Transformer's MLP (FFN) : embed_dim * 4
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    def forward(self, search_token, memory_token):
        if memory_token is None or memory_token.size(1) == 0: # First frame : just return
            return search_token
        
        attention_output, _ = self.cross_attention(query = search_token, key=memory_token, value = memory_token)
        x = self.norm(search_token + attention_output) # Serach token이 Encoder에서 나온거
        MLP_output = self.MLP(x)
        final_ouptut = self.norm2(x + MLP_output)

        return final_ouptut



class TrackerDINOv2(object):
    def __init__(self):
        super().__init__()

        self.net = DINOv2()
        self.template = None
        self.memory_bank = []

    def initialize_tracking(self):
        pass

    def init(self, image, model):
        pass

    def init_with_bbox(self, frame, bbox):
        self.template = self.extract_template(frame, bbox)
        self.prev_bbox = bbox

    def extract_template(self, frame, bbox):
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        s_z = SiamFC_crop_size(w, h)
        template = self.SiamFC_crop(frame, cx, cy, s_z, 112)

        template = self.transform(template).unsqueeze(0)

        return template

    def extract_search(self, frame, prev_bbox):
        x, y, w, h = self.prev_bbox
        cx = x + w / 2
        cy = y + h / 2

        s_z = SiamFC_crop_size(w, h, context=0.5)
        s_x = s_z * 2.0 # 2배 해서 면적 4배

        patch = self.SiamFC_crop(frame, cx, cy, s_x, out_size=224)
        search_tensor = self.transform(patch).unsqueeze(0)
        return search_tensor

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


def SiamFC_crop_size(w, h, context=0.5):
    p = (w + h) / 2 * context # contect padding : 배경 정보를 얻기 위함이며, appearance와 scale 변화에 강해지기 위해
    s = np.sqrt((w+p) * (h+p))

    ## gotta prevent not to let crop area leave the image area.


    return s



def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or (len(pred) == 0)

def get_bbox(label_res):
    measure_per_frame = []
    for _gt, _exist in zip(label_res['gt_rect'], label_res['exist']):
        # Target 존재 x : _exist==False | Target 존재 O : _exist == True
        measure_per_frame.append(not_exist(_pred) if not _exist else)

def get_initial_bbox(json_file_path):
    with open(json_file_path, "r") as f:
        label = json.load(f)
    

    first_bbox = label['gt_rect'][0]
    return first_bbox