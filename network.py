import scipy
import cv2
import numpy as np
import torch
import random
import math
import json
import os
import torch.nn as nn
import torchvision.transforms as T
from filterpy.kalman import KalmanFilter

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
#ROOT_DIR = "/content/drive/MyDrive/Colab Notebooks/CVIP_LAB/UAV/MemLoTrack"


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
        self.type_embedding = TypeEmbedding()
        self.config = config

        for block in self.backbone.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, config = self.config)
            block.mlp.fc1 = LoRALinear(block.mlp.fc1, config = self.config)
            block.mlp.fc2 = LoRALinear(block.mlp.fc2, config = self.config)
        self.mal = MAL(embed_dim=768, num_heads=12)
        self.head = PredictionHead(in_channels=768, hidden_channels=192)

    def get_initial_tokens(self, x):
        # Patch embedding
        tokens = self.backbone.patch_embed(x)
        
        # Positional Embeding
        pos_embed = self.backbone.pos_embed[:, 1:, :]
        output = tokens + pos_embed
        
        return output
        '''
        DINOv2는 CLS Token과 4개의 Register Token이 필요함
        논문 상세 읽어보기
        + Positional Embedding에 대해서도 다시 분석하기
        '''

    def forward(self, z_img, x_img, target_mask, memory_kv = None):
        '''
        z_img : [B, 3, 112, 112]
        x_img : [B, 3, 224, 224]
        target_mask : template grid mask (foreground & background 구분)
        memory_kv = [B, N, 768] (Encoder bypass 된 memory token, 없으면 None)
        '''
        z_tokens = self.get_initial_tokens(z_img) # [B, 64, 768]
        x_tokens = self.get_initial_tokens(x_img) # [B, 256, 768] 

        z_tokens = self.type_embedder(z_tokens, token_type = 'template', target_mask = target_mask) 
        x_tokens = self.type_embedder(x_tokens, token_type = "search")

        tokens_concaten = torch.concat([z_tokens, x_tokens], dim=1)

        B = tokens_concaten.shape[0]
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        reg_tokens = self.backbone.register_tokens.expand(B, -1, -1)

        tokens_final = torch.cat([cls_token, reg_tokens, tokens_concaten], dim=1)

        for block in self.backbone.blocks: # Transformer Encoder (12 layers)
            tokens_final = block(tokens_final)
        tokens_final = self.backbone.norm(tokens_final)
        
        #search_token_enc = tokens_concaten[:, 64:, :] # Search만 빼오기
        # tokens_final 앞에는 cls_token + register_tokens 가 붙어야 함.
        # Head는 search grid 토큰 (N_x) 위에서만 동작해야 함.
        R = self.backbone.register_tokens.shape[1] # Number of Register Tokens of DINOv2
        Nz = z_tokens.shape[1]
        Nx = x_tokens.shape[1]

        start = 1 + R + Nz
        end = start + Nx
        search_token_enc = tokens_final[:, start:end, :]
        
        search_token_enc = self.mal(search_token_enc, memory_kv)
        score_map, size_map = self.head(search_token_enc)

        return score_map, size_map

class LoRALinear(nn.Module):
    # LORA Adapter는 Trainable하게 !! 
    def __init__(self, original_layer, config):
        super().__init__()

        if isinstance(original_layer, LoRALinear):
            original_layer = original_layer.original_layer

        #self.original_qkv = original_qkv
        self.config = config

        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = config['r']
        self.lora_alpha = config['lora_alpha']
        self.lora_dropout = nn.Dropout(config['lora_dropout'])
        self.bias = config['bias']
        self.target_modules = config['target_modules']
        self.scaling = self.lora_alpha / self.rank

        in_features = self.original_layer.in_features
        out_features = self.original_layer.out_features

        self.LoRA_A = nn.Parameter(torch.zeros(in_features, self.rank))
        self.LoRA_B = nn.Parameter(torch.randn(self.rank, out_features))

        nn.init.kaiming_uniform_(self.LoRA_A, a=math.sqrt(5))
        nn.init.zeros_(self.LoRA_B)

    def forward(self, x):
        # if self.original_layer == "original_qkv":
        #     # LORA Adapter 적용된 qkv 계산
        #     self.original_qkv = self.original_layer
        # elif self.original_layer == "original_fc1":
        #     self.original_fc1 = self.original_layer
        # elif self.original_layer == "original_fc2":
        #     self.original_fc2 = self.original_layer

        '''
        [수정 후보]
        아래 두 줄 외 조건문 없애도 될듯? -> 고민해보기
        original_output = self.original_layer(x)
        LoRA_ouptut = torch.matmul(torch.matmul(x, self.LoRA_A), self.LoRA_B) * self.scaling
        '''

        original_output = self.original_layer(x)
        #LoRA_output = torch.matmul(torch.matmul(x, self.LoRA_A), self.LoRA_B) * self.scaling
        LoRA_output = self.lora_dropout(torch.matmul(torch.matmul(x, self.LoRA_A), self.LoRA_B) * self.scaling)
        
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
            return torch.zeros((16, 16))
        x, y, w, h = self.gt_rect[frame]
        cx = x + w/2
        cy = y + h/2

        grid_cx = int(cx/14)
        grid_cy = int(cy/14)
        target_heatmap = generate_gaussian_target(size=16, center=(grid_cx, grid_cy), sigma=1.5)
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

            nn.Conv2d(hidden_channels, 4, kernel_size=3, padding=1), # 4방향 Output
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        #x = x.permute(0, 2, 1) # [B, 256, 768] -> [B, 768, 256]
        #B, N, C=x.shape
        #x = x.contiguous().view(B, 768, 16, 16) # [B, 768, 256] -> [B, 768, 16, 16]
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Search tokens must be a square grid"

        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        score_map = self.ClassificationMLP(x) # Output
        size_map = self.RegressionMLP(x) 

        # score_map = self.ClassificatoinMLP(x) > threshold (.8)
        return score_map, size_map

class KalmanGate(object):
    '''
    https://www.geeksforgeeks.org/python/kalman-filter-in-python/
    참고하였음.
    '''
    # 1. 드론의 현재 상태 벡터로 정의 ->6차원 벡터 
    # 2. Prediction Head가 측정한 z_t
    def __init__(self, initial_bbox):
        #kf = KalmanFilter(dim_x=3, dim_z=1)
        self.kf = KalmanFilter(dim_x = 8, dim_z = 4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0], # cx_new = cx + vx
            [0, 1, 0, 0, 0, 1, 0, 0], # cy_new = cy + vy
            [0, 0, 1, 0, 0, 0, 1, 0], # w_new = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1], # h_new = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0], # vx_new = vx (등속도 운동 가정)
            [0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        # x(x0) : initial state estimate
        self.kf.x[:4, 0] = initial_bbox

        # Initial Error Covariance (Pos)
        self.kf.P *= 10.0
        
        # Initial Error Covariance (Velocity)
        self.kf.P[4:, 4:] *= 1000.0

        # Measurement Noise Covariance
        self.kf.R *= 10.0

        # Process NOise Covariance
        self.kf.Q *= 0.01

        # self.F = F   # State transition matrix (system Model)  
        # self.B = B   # Control matrix (effect of control input)
        # self.H = H   # Observation matrix (how we measure the state)
        # self.Q = Q   # Process Noise Covariance (Uncertainty in the process)
        # self.R = R   # Measurement noise covariance (uncertainty in the measurements)
        # self.x = x0  # Initial state estimate
        # self.P = P0  # Initial error covariance

    def predict(self):
        self.kf.predict()
        # self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        # self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        #return self.x
        return self.kf.x
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    
    def gate(self, z_t, threshold=9.49):
        '''
        Threshold 값 다시 정해야 함. 
        마할라노비스 거리 기준이니까 신뢰구간 구하기
        '''
        self.kf.predict() # kalman으로 업데이트

        z = np.array(z_t).reshape(4,1)
        y = z - np.dot(self.kf.H, self.kf.x)

        S = np.dot(self.kf.H, np.dot(self.kf.P, self.kf.H.T)) + self.kf.R
        S_inv = scipy.linalg.inv(S)
        mahalanovis_dist = np.dot(y.T, np.dot(S_inv, y))[0, 0]

        is_passed = (mahalanovis_dist < threshold)

        if is_passed: # 칼만 내부 상태 업데이트로 속도 학습하기
            self.kf.update(z)

        return is_passed, mahalanovis_dist

    


class customKalman(object):
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F   # State transition matrix (system Model)  
        self.B = B   # Control matrix (effect of control input)
        self.H = H   # Observation matrix (how we measure the state)
        self.Q = Q   # Process Noise Covariance (Uncertainty in the process)
        self.R = R   # Measurement noise covariance (uncertainty in the measurements)
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial error covariance

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    
    def gate(self, z_t, threshold=0.8):
        self.kf.predict() # kalman으로 업데이트

        z = np.array(z_t).reshape(4,1)
        y = z - np.dot(self.kf.H, self.kf.x)

        S = np.dot(self.kf.H, np.dot(self.kf.P, self.kf.H.T)) + self.kf.R
        S_inv = scipy.linalg.inv(S)
        mahalanovis_dist = np.dot(y.T, np.dot(S_inv, y))[0, 0]

        is_passed = (mahalanovis_dist < threshold)

        if is_passed: # 칼만 내부 상태 업데이트로 속도 학습하기
            self.kf.update(z)

        return is_passed, mahalanovis_dist

class MemoryBank: # Only During Inference
    def __init__(self, max_size=7, kalman_gate=None):
        self.max_size = max_size
        self.memory= []
        self.kalman = kalman_gate

    def get_affine_from_bbox(self, old_bbox, new_bbox):
        cx_o, cy_o, w_o, h_o = old_bbox
        cx_n, cy_n, w_n, h_n = new_bbox

        sx = w_n / (w_o + 1e-6)
        sy = h_n / (h_o + 1e-6)
        tx = cx_n - cx_o
        ty = cy_n - cy_o

        M = np.array([
            [sx, 0, tx],
            [0, sy, ty]
        ], dtype=np.float32)

        return M
    
    def warp_token(self, token, M):
        token = token.permute(1, 2, 0).cpu().numpy()
        warped = cv2.warpAffine(token, M, (token.shape[1], token.shape[0]))
        return torch.tensor(warped).permute(2, 0, 1)
    
        
    #def add(self, curr_token, curr_score_map, curr_regression_map, prev_bbox, tracker, tau=0.8):
    def add(self, curr_token, prev_bbox, curr_score_map, curr_regression_map, tracker, tau=0.8):
        # Dual Gate 필요
        # (1) Confidence Score : using maximum score 

        # (2) Motion consistency : Kalman-based Mahalanobis test
        # Kalman Filter 대체 : Optical Flow
        
        # 현재 프레임에서 BBox, Conf Score 추출하기
        curr_conf, curr_bbox = tracker.prediction_2_Box(curr_score_map, curr_regression_map)

        if curr_conf < tau:
            return False
        
        motion_ok, _ = self.kalman.gate(curr_bbox)
        if not motion_ok:
            return False
        
        M = self.get_affine_from_bbox(prev_bbox, curr_bbox)
        aligned_token = self.warp_token(curr_token, M)

        self.memory.append(aligned_token)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

        return True

'''
Affine Transform 역산을 통해 도출해 낸 원본 이미지 상의 최종 Bounding Box 좌표 [cx,cy,w,h]
'''


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

        #self.template_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Template -> Target + Background 두 개로 나뉨
        self.template_target = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.template_background = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.search_token = nn.Parameter(torch. randn(1, 1, embed_dim)* 0.02)
        self.memory_token = nn.Parameter(torch. randn(1, 1, embed_dim) * 0.02)

    def forward(self, x, token_type, target_mask = None):
        # x : [B, N, 768] (DINO 통과함)
        if token_type == 'search':
            return x + self.search_token
        elif token_type == "memory":
            return x + self.memory_token
        elif token_type == "template":
            #return x + self.template_target
            if target_mask is None:
                return x + self.template_target
            target_mask = target_mask.unsqueeze(-1)
            x = x + target_mask * self.template_target \
          + (1 - target_mask) * self.template_background
            return x
            # 수정 필요함
        # elif token_type == "template_bg":
        #     return x + self.template_background

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
        x = self.norm1(search_token + attention_output) # Serach token이 Encoder에서 나온거
        MLP_output = self.MLP(x)
        final_ouptut = self.norm2(x + MLP_output)

        return final_ouptut



class TrackerDINOv2():
    def __init__(self, initial_bbox):
        super().__init__()

        # Model initialization
        self.net = DINOv2(TypeEmbedding(), LoRAConfig)
        self.type_embedder = TypeEmbedding()
        self.template = None
        self.kalman = KalmanGate(initial_bbox)
        self.memory_bank = MemoryBank(kalman_gate=self.kalman)
        self.transform = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
    
    def prediction_2_Box(self, score_map, regression_map):
        B, _, H, W = score_map.shape
        assert B == 1, "PredictionHead의 output은 반드시 1개의 배치만 있어야 함."
        #flatten_scoreMap = score_map.reshape(B, -1)
        flatten_scoreMap = score_map.view(B, -1)
        conf, idx = flatten_scoreMap.max(dim=1)
        
        #max_score_pos = torch.argmax(flatten_scoreMap, dim=1).item() # return Position of Max Score
        #max_score_pos = flatten_scoreMap.argmax(dim=1) # B = 1 일 경우에는 .item() 해도 좋지만, 1보다 클 경우 첫 배치만 강제로 스칼라로 만들기 때문에 나머지 배치 정보 손실됨.
        max_score_pos = int(idx[0].item())
        confidence_score = float(conf[0].item())

        grid_x = max_score_pos % W
        grid_y = max_score_pos // W
        

        '''
        16 x 16에서 생각해보면 0~15 idx까지 row 인것 생각하기  
         xxxxxxxxxxxxxxxxx
        + 패치가 16x16 사이즈인 것이랑, 실제 원본 이미지 224x224 차이 생각하기
        '''
        grid_x =  (max_score_pos % 16) 
        grid_y =  (max_score_pos // 16)
        
        # Regression Map에서 4방향 거리를 뽑아오기
        # Regression_map 의 shape : [B, 4, 16, 16]
        l = float(regression_map[0, 0, grid_y, grid_x].item())
        t = float(regression_map[0, 1, grid_y, grid_x].item())
        r = float(regression_map[0, 2, grid_y, grid_x].item())
        b = float(regression_map[0, 3, grid_y, grid_x].item())

        stride = 14.0 # Patch Size
        w = (l + r) * 14
        h = (t + b) * 14

        cx = grid_x * 14    
        cy = grid_y * 14

        #org_bbox = [grid_x, grid_y, w, h]
        org_bbox = [cx, cy, w, h]
        return confidence_score, org_bbox

def SiamFC_crop_size(w, h, context=0.5):
    p = (w + h) / 2 * context # contect padding : 배경 정보를 얻기 위함이며, appearance와 scale 변화에 강해지기 위해
    s = np.sqrt((w+p) * (h+p))

    ## gotta prevent not to let crop area leave the image area.
    return s