import torch 
from torchvision.transforms import transforms
from PIL import Image
from sklearn import svm
import os
import cv2
import json
import glob
from tqdm.notebook import tqdm
import numpy as np

# DINOv2
#dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')


# ViT-B / 14 가져오야 함 (patch size : 14) + with Register
# dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
"""
https://colab.research.google.com/github/roboflow/notebooks/blob/2f37e2e61d5befbb91669ddb90d8529b24b70d72/notebooks/dinov2-classification.ipynb?ref=blog.roboflow.com

Custom Training on DINOv2 Guideline
"""

'''
Load Data
'''
current_path = os.getcwd()
print(current_path)

USER_PATH = os.path.expanduser("~")

ROOT_DIR = os.path.join(USER_PATH, "ai_study/CVIP/Anti-UAV/MemLoTrack", "Anti-UAV410")
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
TEST_DIR = os.path.join(ROOT_DIR, "test")
VAL_DIR = os.path.join(ROOT_DIR, "val")

with open(os.path.join(TRAIN_DIR, "01_1667_0001-1500", "IR_label.json")) as f:
    data = json.load(f)
print(type(data))
print(data.keys() if isinstance(data, dict) else data[:1])
print(ROOT_DIR)



### Checking Frame Size ### 

print(sorted(os.listdir(TRAIN_DIR))) # train Folders

for seq in sorted(os.listdir(TRAIN_DIR)):
    seq_path = os.path.join(ROOT_DIR, seq)

    if not os.path.isdsir(seq_path):
        continue
    json_path = os.path.join(seq_path, "IR_label.json")
    if not os.path.exist(json_path):
        print(f"label .json이 없습니다. {json_path}")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    
    


def main(mode="IR", target_folder=None):
    base_dir = os.path.join(ROOT_DIR, "test")
    if target_folder is None:
        video_path = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    else:
        video_path = sorted([os.path.join(base_dir, d) for d in target_folder if os.path.isdir(os.path.join(base_dir, d))])

    
    for  video_id, video_path in enumerate(video_path, starts=1):
        print(f"\n=== {video_id}/{len(video_path)}")
        video_name = os.path.basename(video_path)
        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        gt_file = os.path.join(video_path, "IR_label.json")
        with open(gt_file, "r") as f:
            label_res = json.read(f)
        
        if 'exist' not in label_res:
            label_res['exist'] = [1] * len(label_res['gt_rect'])
        
        gt_rects = label_res['gt_rect'] # gt_rects 받은거 BBox 계산하는 쪽으로 넘겨주기
        exist_flags = label_res['exist']
        video_ious = []

        if len(gt_rects) > 1:
            num_frames = min(len(frame_files), len(gt_rects))
        else:
            num_frames = len(frame_files)
            
        for frame_id in range(num_frames):
            frame_file = frame_files[frame_id]
            frame_path = os.path.join(video_path, frame_file)
            frame = cv2.imread(frame_path)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            im_vis = frame.copy()
            
            # 첫 프레임 아닐 때 (Memory Token 고려해야할듯?)
            # 바로 밑 코드는 첫 프레임일 때 GT 뽑는거 (근데 사실 MemLoTrack은 GT)
            if frame_id == 0 and(len(gt_rects) > 0 and exist_flags[0] == 1):
                init_bbox = list(gt_rects[0])
            first_frame_flag = 1

            


            


labels={}

for folder in os.listdir(ROOT_DIR):
    for file in os.listdir(os.path.join(ROOT_DIR, folder)):
        if file.endswith(".jpg"):
            full_name = os.path.join(ROOT_DIR, folder, file)
            labels[full_name] = folder
files = labels.keys()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dinov2_vitl14.to(device)

transform_img = transforms.Compose([
    transforms.ToTensor(), transforms.Resize(224), transforms.CenterCrop(224), 
    transforms.Normalize([0.5], [0.5])
])

def load_img(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.open(img)
    transformed_img = transform_img(img)[:3].unsqueeze(0)
    return transform_img



"""
compute_embeddings이 바로 음.. 아니라고 한다. 공부 더하자 
"""

def compute_embeddings(files:list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.
    """

    all_embeddings={}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vitl14(load_img(file).to(device))

            all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("all_embeddings.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings

# Compute Embeddings
embeddings = compute_embeddings(files)




# Train a classification Model
# For this guide, classification model is SVM.
clf = svm.SVC(gamma="scale")

y = [labels[file] for file in files]

embedding_list = list(embeddings.values())
clf.fit(np.array(embedding_list).reshape(-1, 384), y)



# Patch_size : 14

transform_img = transforms.Compose([
    transforms.Resize()
])





'''

breakpoint()


'''