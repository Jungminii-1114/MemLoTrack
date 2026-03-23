import os
import cv2
import torch
import numpy as np
import json
import torchvision.transforms as T
from network_colab import DINOv2, TypeEmbedding, LoRAConfig

import os
import json
import random
import cv2
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import wandb

from network_colab import DINOv2, TypeEmbedding, LoRAConfig, MemorySampling, generate_gaussian_target

class TrainConfig:
    def __init__(self):
        self.dataset_root : str="/content/data/Anti-UAV410"
        self.split:str='train'
        self.epochs:int = 20
        self.batch_size : int=1
        self.lr : float = 1e-4
        self.weight_decay : float = 1e-4
        self.device : str="cuda"
        self.seed : int=42
        self.l_mem : int = 7
        self.tau : float=0.8
        self.save_dir : str = "/content/ckpt"

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AntiUAVDataset(Dataset):
    def __init__(self, root_dir : str, split : str = "train"):
        self.split_dir = os.path.join(root_dir, split)
        self.video_seqs = []

        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"경로 찾을 수 없음 : {self.split_dir}")

        for video_name in sorted(os.listdir(self.split_dir)):
            seq_dir = os.path.join(self.split_dir, video_name)
            if not os.path.exists(seq_dir):
                continue

            label_path = os.path.join(seq_dir, "IR_label.json")
            if not os.path.exists(label_path):
                continue

            with open(label_path, "r") as f:
                label_data = json.load(f)

            img_paths = sorted([os.path.join(seq_dir, p) for p in os.listdir(seq_dir) if p.lower().endswith((".jpg", ".png", ".jpeg"))])

            if len(img_paths) > 0:
                self.video_seqs.append({
                    "video_name" : video_name,
                    "img_paths" : img_paths,
                    "exist" : label_data['exist'],
                    "gt_rect" : label_data['gt_rect']
                })
    def __len__(self):
        return len(self.video_seqs)

    def __getitem__(self, idx):
        return self.video_seqs[idx]

def collate_fn(batch):
    return batch[0]

def frame_choose(exist_list : List[int], l_mem : int = 7):
    valid_frames = [i for i in range(len(exist_list)) if exist_list[i] == 1]

    if len(valid_frames) < 2:
        raise ValueError("Invalid Video that consists with less than 2 drone frames")
    search_idx = random.choice(valid_frames[1:])

    template_candidates = [i for i in valid_frames if i <search_idx]
    template_idx = random.choice(template_candidates)

    mem_candidates = [i for i in valid_frames if template_idx < i < search_idx]

    mem_indices = []
    if len(mem_candidates) >= l_mem:
        mem_indices = sorted(random.sample(mem_candidates, l_mem))
    elif len(mem_candidates) > 0:
        mem_indices = mem_candidates + [mem_candidates[-1]] * (l_mem - len(mem_candidates))
    else:
        mem_indices = [template_idx] * l_mem

    return template_idx, search_idx, mem_indices

def bbox_crop(bbox : List[float]):
    x, y, w, h = bbox
    return (x+w/2.0, y+h/2.0, w, h)

def siamfc_crop(img: np.ndarray, cx : float, cy : float, size : float, out_size:int):
    size = max(16.0, float(size))
    patch = cv2.getRectSubPix(img, (int(size), int(size)), (float(cx), float(cy)))
    patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return patch



def get_cropped_target(bbox: List[float]) -> List[float]:
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    
    context = 0.5
    p = (w + h) / 2.0 * context
    
    val = max(0.0, (w + p) * (h + p))
    s_z = float(np.sqrt(val))
    s_x = s_z * 2.0
    
    s_x = max(16.0, float(s_x))
    
    scale = 224.0 / s_x
    
    new_w = w * scale
    new_h = h * scale
    new_x = 112.0 - new_w / 2.0
    new_y = 112.0 - new_h / 2.0
    
    return [new_x, new_y, new_w, new_h]

def make_template_and_search(frame : np.ndarray, bbox : List[float], transform, is_search : bool = False):
    cx, cy, w, h = bbox_crop(bbox)

    context = 0.5
    p = (w + h) / 2.0 * context
    s_z = float(np.sqrt((w+p) * (h+p)))

    if is_search:
        s_x = s_z * 2.0
        patch = siamfc_crop(frame, cx, cy, s_x, out_size=224)
    else:
        patch = siamfc_crop(frame, cx, cy, s_z, out_size=112)

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    tensor_img = transform(patch_rgb).unsqueeze(0) 

    return tensor_img

def make_target(size:int, center:Tuple[int, int], sigma:float = 1.5):
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    cx, cy = center
    heatmap = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return heatmap

def cls_target(gt_bbox : List[float], stride : int=14, grid:int = 16):
    x, y, w, h = gt_bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    grid_x = int(cx / stride)
    grid_y = int(cy / stride)

    grid_x = max(0, min(grid-1, grid_x))
    grid_y = max(0, min(grid - 1, grid_y))

    target = make_target(size=grid, center=(grid_x, grid_y), sigma=1.5)
    return target.unsqueeze(0).unsqueeze(0)

def reg_target(gt_bbox: List[float], stride: int=14, grid: int=16):
    x, y, w, h = gt_bbox
    x1, y1 = x, y               
    x2, y2 = x + w, y + h       

    gy, gx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    
    grid_cx = gx * stride + (stride / 2.0)
    grid_cy = gy * stride + (stride / 2.0)

    l = grid_cx - x1
    t = grid_cy - y1
    r = x2 - grid_cx
    b = y2 - grid_cy

    target = torch.stack([l, t, r, b], dim=0) / stride
    return target.unsqueeze(0)

def train(model:DINOv2, loader:DataLoader, optimizer:torch.optim.Optimizer, device:torch.device, cfg:TrainConfig):
    model.train()
    bce = nn.BCELoss()
    # [deactivated] reduction='mean' 이라 스칼라가 되어 cls_tgt와 위치별 가중이 깨짐 → F.smooth_l1_loss(..., reduction="none") 사용
    # l1 = nn.SmoothL1Loss()

    total_loss = 0.0
    steps = 0

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pbar = tqdm(loader, desc="Train", leave=False)
    for seq in pbar:
        img_paths = seq['img_paths']
        exist = seq['exist']
        gt_rect = seq['gt_rect']

        try:
            t_idx, s_idx, mem_idx = frame_choose(exist, cfg.l_mem)
        except ValueError:
            continue

        t_frame = cv2.imread(img_paths[t_idx])
        s_frame = cv2.imread(img_paths[s_idx])

        z_img = make_template_and_search(t_frame, gt_rect[t_idx], transform, is_search=False).to(device)
        x_img = make_template_and_search(s_frame, gt_rect[s_idx], transform, is_search=True).to(device)

        mem_imgs = []
        for mi in mem_idx:
            memory_frame = cv2.imread(img_paths[mi])
            memory_patch = make_template_and_search(memory_frame, gt_rect[mi], transform, is_search=True)
            mem_imgs.append(memory_patch.squeeze(0)) 

        mem_imgs = torch.stack(mem_imgs, dim=0).unsqueeze(0).to(device) 

        B, L, C, H, W = mem_imgs.shape
        mem_flat = mem_imgs.view(B*L, C, H, W)
        mem_tokens = model.get_initial_tokens(mem_flat)
        mem_tokens = model.type_embedder(mem_tokens, token_type="memory")
        mem_tokens = mem_tokens.reshape(B, L * mem_tokens.shape[1], mem_tokens.shape[2])

        target_mask = torch.ones((1, 64), dtype=torch.float32).to(device)
        
        gt_in_crop = get_cropped_target(gt_rect[s_idx])
        cls_tgt = cls_target(gt_in_crop).to(device)
        reg_tgt = reg_target(gt_in_crop).to(device)

        score_map, reg_map = model(z_img, x_img, target_mask, memory_kv=mem_tokens)

        loss_cls = bce(score_map, cls_tgt)
        reg_err = F.smooth_l1_loss(reg_map, reg_tgt, reduction="none", beta=1.0)
        loss_reg = (reg_err * cls_tgt).sum() / (cls_tgt.sum() * reg_map.shape[1] + 1e-6)
        # [deactivated] loss_reg = (l1(reg_map, reg_tgt) * cls_tgt).mean()
        loss = loss_cls + 5.0 * loss_reg 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        loss = loss_cls + 5.0 * loss_reg
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, steps)

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

def decode_bbox_in_search_crop(
    reg_map: torch.Tensor,
    max_y: int,
    max_x: int,
    stride: float = 14.0,
    search_size: float = 224.0,
    min_wh: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    argmax 셀의 l,t,r,b (stride 정규화) → 서치 크롭 좌표계 (x, y, w, h).
    회귀 헤드가 음수를 낼 수 있으므로 w/h는 크롭 안에서만 클램프.
    """
    pred_l, pred_t, pred_r, pred_b = reg_map[0, :, max_y, max_x] * stride
    px = max_x * stride + stride / 2.0
    py = max_y * stride + stride / 2.0
    pred_w = torch.clamp(pred_l + pred_r, min=min_wh, max=search_size).item()
    pred_h = torch.clamp(pred_t + pred_b, min=min_wh, max=search_size).item()
    pred_x = (px - pred_l).item()
    pred_y = (py - pred_t).item()
    return pred_x, pred_y, pred_w, pred_h

def validate(model: DINOv2, loader: DataLoader, device: torch.device, cfg: TrainConfig):
    model.eval() 
    bce = nn.BCELoss()
    # [deactivated] train과 동일 이유 — 위치별 가중 regression 은 reduction='none' 경로 사용
    # l1 = nn.SmoothL1Loss()

    val_loss = 0.0
    val_iou = 0.0
    steps = 0
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad(): 
        pbar = tqdm(loader, desc="Valid", leave=False)

        for seq in pbar:
            img_paths = seq['img_paths']
            exist = seq['exist']
            gt_rect = seq['gt_rect']

            try:
                t_idx, s_idx, mem_idx = frame_choose(exist, cfg.l_mem)
            except ValueError:
                continue

            t_frame = cv2.imread(img_paths[t_idx])
            s_frame = cv2.imread(img_paths[s_idx])

            z_img = make_template_and_search(t_frame, gt_rect[t_idx], transform, is_search=False).to(device)
            x_img = make_template_and_search(s_frame, gt_rect[s_idx], transform, is_search=True).to(device)

            mem_imgs = []
            for mi in mem_idx:
                memory_frame = cv2.imread(img_paths[mi])
                memory_patch = make_template_and_search(memory_frame, gt_rect[mi], transform, is_search=True)
                mem_imgs.append(memory_patch.squeeze(0))

            mem_imgs = torch.stack(mem_imgs, dim=0).unsqueeze(0).to(device)

            B, L, C, H, W = mem_imgs.shape
            mem_flat = mem_imgs.view(B*L, C, H, W)
            mem_tokens = model.get_initial_tokens(mem_flat)
            mem_tokens = model.type_embedder(mem_tokens, token_type="memory")
            mem_tokens = mem_tokens.reshape(B, L * mem_tokens.shape[1], mem_tokens.shape[2])

            target_mask = torch.ones((1, 64), dtype=torch.float32).to(device)
            
            gt_in_crop = get_cropped_target(gt_rect[s_idx])
            cls_tgt = cls_target(gt_in_crop).to(device)
            reg_tgt = reg_target(gt_in_crop).to(device)

            score_map, reg_map = model(z_img, x_img, target_mask, memory_kv=mem_tokens)
            
            # Loss 계산
            loss_cls = bce(score_map, cls_tgt)
            reg_err = F.smooth_l1_loss(reg_map, reg_tgt, reduction="none", beta=1.0)
            loss_reg = (reg_err * cls_tgt).sum() / (cls_tgt.sum() * reg_map.shape[1] + 1e-6)
            # [deactivated] loss_reg = (l1(reg_map, reg_tgt) * cls_tgt).mean()
            loss = loss_cls + 5.0 * loss_reg 
            val_loss += loss.item()
            
            b, c, h, w = score_map.shape
            scores_flat = score_map.view(b, -1)
            max_idx = torch.argmax(scores_flat, dim=1)
            max_y = (max_idx // w).item()
            max_x = (max_idx % w).item()

            pred_x, pred_y, pred_w, pred_h = decode_bbox_in_search_crop(reg_map, max_y, max_x)
            pred_box = [pred_x, pred_y, pred_w, pred_h]
            
            # IoU 계산하기 +
            current_iou = calculate_iou(pred_box, gt_in_crop)
            val_iou += current_iou
            
            steps += 1
            pbar.set_postfix(iou=f"{current_iou:.4f}")

    return val_loss / max(1, steps), val_iou / max(1, steps)

def main():
    cfg = TrainConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)
    set_seed(cfg.seed)

    wandb.init(
        project = "MemLoTrack-DINOv2_train",
        name="Experiment_01",
        config=cfg.__dict__
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(f"현재 사용 장치 :{device}")

    model = DINOv2(TypeEmbedding(), LoRAConfig)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_dataset = AntiUAVDataset(cfg.dataset_root, split=cfg.split)
    val_dataset = AntiUAVDataset(cfg.dataset_root, split='val')

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    best_loss = float('inf')
    best_iou = 0.0

    for epoch in range(cfg.epochs):
        print(f"\n---[Epoch {epoch+1}/{cfg.epochs}] Train ---")
        train_loss = train(model, train_loader, optimizer, device, cfg)
        
        print(f"---[Epoch {epoch+1}/{cfg.epochs}] Validation ---")
        val_loss, val_iou = validate(model, val_loader, device, cfg)
        
        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Val IoU": val_iou
        })

        if val_iou >= best_iou:
            print(f"New Best IoU Recorded : {best_iou:.4f} -> {val_iou:.4f}")
            best_iou = val_iou

            ckpt_path = os.path.join(cfg.save_dir, "best_MemLoTrack.pt")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou
            }, ckpt_path)
            
    wandb.finish()

if __name__ == "__main__":
    main()













def map_crop_to_original(pred_box_crop: List[float], orig_prev_bbox: List[float]) -> List[float]:
    px, py, pw, ph = pred_box_crop
    x, y, w, h = orig_prev_bbox
    
    orig_cx = x + w / 2.0
    orig_cy = y + h / 2.0
    
    context = 0.5
    p = (w + h) / 2.0 * context
    s_z = float(np.sqrt(max(0.0, (w + p) * (h + p))))
    s_x = max(16.0, s_z * 2.0)
    
    scale = s_x / 224.0
    
    # 112 112 기준으로 원본 좌표 복원하기 
    orig_pred_x = orig_cx + (px - 112.0) * scale
    orig_pred_y = orig_cy + (py - 112.0) * scale
    orig_pred_w = pw * scale
    orig_pred_h = ph * scale
    
    return [orig_pred_x, orig_pred_y, orig_pred_w, orig_pred_h]

def run_real_inference():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    ckpt_path = "/content/ckpt/best_MemLoTrack.pt"
    
    print("가중치 불러오는 중..")
    model = DINOv2(TypeEmbedding(), LoRAConfig)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() 

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_base_dir = "/content/data/Anti-UAV410/test"
    video_folders = sorted([f for f in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, f))])
    if not video_folders:
        raise FileNotFoundError("테스트 비디오 폴더가 없습니다!")
        
    test_video_dir = os.path.join(test_base_dir, video_folders[0])
    print(f"Inference 과정 투입 비디오: {test_video_dir}")
    
    img_names = sorted([p for p in os.listdir(test_video_dir) if p.endswith(('.jpg', '.png'))])
    img_paths = [os.path.join(test_video_dir, p) for p in img_names]
    
    with open(os.path.join(test_video_dir, "IR_label.json"), "r") as f:
        labels = json.load(f)
    gt_rects = labels['gt_rect']
    exists = labels['exist']

    sample_frame = cv2.imread(img_paths[0])
    h, w, _ = sample_frame.shape
    out_video_path = "/content/tracking_result_SUCCESS.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (w, h))

    print(f"Start Tracking.. 총 길이: {len(img_paths)}")
  
    template_tensor = None
    mem_imgs = []
    prev_bbox = None
    l_mem = 7 

    with torch.no_grad():
        for i, img_path in enumerate(img_paths):
            frame = cv2.imread(img_path)
            
            if i == 0:
                prev_bbox = gt_rects[0]
                # Template (112x112) 고정
                template_tensor = make_template_and_search(frame, prev_bbox, transform, is_search=False).to(device)
                
                # 메모리 초기화 (첫 프레임의 Search 영역으로 7칸 채움)
                first_mem = make_template_and_search(frame, prev_bbox, transform, is_search=True).squeeze(0)
                mem_imgs = [first_mem] * l_mem
                
                curr_bbox = prev_bbox
                print(f"[{i:04d}] 타겟 락온: {curr_bbox}")
                
            
            else:
                search_tensor = make_template_and_search(frame, prev_bbox, transform, is_search=True).to(device)
                
                # 메모리 토큰 준비
                mem_stack = torch.stack(mem_imgs, dim=0).unsqueeze(0).to(device) # [1, 7, 3, 224, 224]
                B, L, C, H, W = mem_stack.shape
                mem_tokens = model.type_embedder(model.get_initial_tokens(mem_stack.view(B*L, C, H, W)), token_type="memory")
                mem_tokens = mem_tokens.reshape(B, L * mem_tokens.shape[1], mem_tokens.shape[2])
                
                target_mask = torch.ones((1, 64), dtype=torch.float32).to(device)
                
                score_map, reg_map = model(template_tensor, search_tensor, target_mask, memory_kv=mem_tokens)
                
                b_sz, c_sz, h_sz, w_sz = score_map.shape
                scores_flat = score_map.view(b_sz, -1)
                max_idx = torch.argmax(scores_flat, dim=1)
                max_y = (max_idx // w_sz).item()
                max_x = (max_idx % w_sz).item()

                pred_x_crop, pred_y_crop, pred_w, pred_h = decode_bbox_in_search_crop(
                    reg_map, max_y, max_x
                )
                pred_box_crop = [pred_x_crop, pred_y_crop, pred_w, pred_h]
                
                # 224x224 좌표를 원본 영상 좌표로 변환
                curr_bbox = map_crop_to_original(pred_box_crop, prev_bbox)
                
                # 다음 프레임을 위해 이전 위치 갱신
                prev_bbox = curr_bbox
                
                
                curr_mem_patch = make_template_and_search(frame, curr_bbox, transform, is_search=True).squeeze(0)
                mem_imgs.pop(0)
                mem_imgs.append(curr_mem_patch)

            
            if curr_bbox is not None and sum(curr_bbox) > 0:
                bx, by, bw, bh = [int(v) for v in curr_bbox]
                # 모델의 예측 (빨간색)
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            
            # 정답 (초록색)
            if exists[i] == 1:
                gx, gy, gw, gh = [int(v) for v in gt_rects[i]]
                cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            
            out.write(frame)
            if i % 50 == 0 and i > 0:
                print(f"[{i:04d}/{len(img_paths)}] 추적 중...")

    out.release()
    print(f"Tracking finished.. Video Saved in : {out_video_path}")

# [deactivated] 학습만 돌릴 때는 아래 블록 비활성화 — __main__ 이 여러 개면 train 직후 추론이 연속 실행됨
# if __name__ == "__main__":
#     run_real_inference()









# --- [deactivated] 파일 상단 map_crop_to_original / run_real_inference 와 중복.
# 로드 시 뒤쪽 정의가 앞쪽을 덮어써 혼란·이중 실행 유발 → 전체 주석 처리 (필요 시 # 제거)
# import os
# import cv2
# import torch
# import numpy as np
# import json
# import torchvision.transforms as T
# from tqdm import tqdm
# from network_colab import DINOv2, TypeEmbedding, LoRAConfig
#
# def map_crop_to_original(pred_box_crop: List[float], orig_prev_bbox: List[float]) -> List[float]:
#     px, py, pw, ph = pred_box_crop
#     x, y, w, h = orig_prev_bbox
#     orig_cx = x + w / 2.0
#     orig_cy = y + h / 2.0
#     context = 0.5
#     p = (w + h) / 2.0 * context
#     s_z = float(np.sqrt(max(0.0, (w + p) * (h + p))))
#     s_x = max(16.0, s_z * 2.0)
#     scale = s_x / 224.0
#     orig_pred_x = orig_cx + (px - 112.0) * scale
#     orig_pred_y = orig_cy + (py - 112.0) * scale
#     orig_pred_w = pw * scale
#     orig_pred_h = ph * scale
#     return [orig_pred_x, orig_pred_y, orig_pred_w, orig_pred_h]
#
# def run_real_inference():
#     device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
#     print(f" 현재 장치: {device} ", flush=True)
#     ckpt_path = "/content/ckpt/best_MemLoTrack.pt"
#     print("weight 불러오는 중..", flush=True)
#     model = DINOv2(TypeEmbedding(), LoRAConfig)
#     checkpoint = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.to(device)
#     model.eval()
#     print("weight 전이 완료", flush=True)
#     transform = T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     test_base_dir = "/content/data/Anti-UAV410/test"
#     video_folders = sorted([f for f in os.listdir(test_base_dir) if os.path.isdir(os.path.join(test_base_dir, f))])
#     if not video_folders:
#         raise FileNotFoundError("테스트 비디오 폴더가 없습니다!")
#     test_video_dir = os.path.join(test_base_dir, video_folders[0])
#     print(f"Inference 용 비디오: {test_video_dir}", flush=True)
#     img_names = sorted([p for p in os.listdir(test_video_dir) if p.endswith(('.jpg', '.png'))])
#     img_paths = [os.path.join(test_video_dir, p) for p in img_names]
#     with open(os.path.join(test_video_dir, "IR_label.json"), "r") as f:
#         labels = json.load(f)
#     gt_rects = labels['gt_rect']
#     exists = labels['exist']
#     sample_frame = cv2.imread(img_paths[11])
#     h, w, _ = sample_frame.shape
#     out_video_path = "/content/tracking_result_SUCCESS.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(out_video_path, fourcc, 30.0, (w, h))
#     template_tensor = None
#     mem_imgs = []
#     prev_bbox = None
#     l_mem = 7
#     pbar = tqdm(enumerate(img_paths), total=len(img_paths), desc="비디오 추적 중")
#     with torch.no_grad():
#         for i, img_path in pbar:
#             frame = cv2.imread(img_path)
#             if i == 0:
#                 prev_bbox = gt_rects[0]
#                 template_tensor = make_template_and_search(frame, prev_bbox, transform, is_search=False).to(device)
#                 first_mem = make_template_and_search(frame, prev_bbox, transform, is_search=True).squeeze(0)
#                 mem_imgs = [first_mem] * l_mem
#                 curr_bbox = prev_bbox
#             else:
#                 search_tensor = make_template_and_search(frame, prev_bbox, transform, is_search=True).to(device)
#                 mem_stack = torch.stack(mem_imgs, dim=0).unsqueeze(0).to(device)
#                 B, L, C, H, W = mem_stack.shape
#                 mem_tokens = model.type_embedder(model.get_initial_tokens(mem_stack.view(B*L, C, H, W)), token_type="memory")
#                 mem_tokens = mem_tokens.reshape(B, L * mem_tokens.shape[1], mem_tokens.shape[2])
#                 target_mask = torch.ones((1, 64), dtype=torch.float32).to(device)
#                 score_map, reg_map = model(template_tensor, search_tensor, target_mask, memory_kv=mem_tokens)
#                 b_sz, c_sz, h_sz, w_sz = score_map.shape
#                 scores_flat = score_map.view(b_sz, -1)
#                 max_idx = torch.argmax(scores_flat, dim=1)
#                 max_y = (max_idx // w_sz).item()
#                 max_x = (max_idx % w_sz).item()
#                 pred_l, pred_t, pred_r, pred_b = reg_map[0, :, max_y, max_x] * 14.0
#                 px = max_x * 14.0 + 7.0
#                 py = max_y * 14.0 + 7.0
#                 pred_w = (pred_l + pred_r).item()
#                 pred_h = (pred_t + pred_b).item()
#                 pred_x_crop = (px - pred_l).item()
#                 pred_y_crop = (py - pred_t).item()
#                 pred_box_crop = [pred_x_crop, pred_y_crop, pred_w, pred_h]
#                 curr_bbox = map_crop_to_original(pred_box_crop, prev_bbox)
#                 prev_bbox = curr_bbox
#                 curr_mem_patch = make_template_and_search(frame, curr_bbox, transform, is_search=True).squeeze(0)
#                 mem_imgs.pop(0)
#                 mem_imgs.append(curr_mem_patch)
#             if curr_bbox is not None and sum(curr_bbox) > 0:
#                 bx, by, bw, bh = [int(v) for v in curr_bbox]
#                 cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
#             if exists[i] == 1:
#                 gx, gy, gw, gh = [int(v) for v in gt_rects[i]]
#                 cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
#             out.write(frame)
#     out.release()
#     print(f"\nTracking Finished .. Video Saved in : {out_video_path}")
#
# if __name__ == "__main__":
#     run_real_inference()


    