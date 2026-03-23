# MemLoTrack: Enhancing TIR Anti-UAV Tracking with Memory-Integrated Low-Rank Adaptation
Inspired by the architecture of the paper *"MemLoTrack: Enhancing TIR Anti-UAV Tracking with Memory-Integrated Low-Rank Adaptation"*, this repository shows the overall process of implementing MemLoTrack with DINOv2.

## Key Features
* **Dual Gating Process of Memory Bank(MB)**
* **Memory Attention Layer (MAL)**
* **Inference Pipeline**

## Repository Structure


## Project Results (Being Aborted...)

https://github.com/user-attachments/assets/cdbdf280-6250-49f0-95e6-2a5dadf5c2f6




### `MemEffAttention` Module in DINOv2
An optimized implementatino of self-attention mechanism designed for memory efficiency.
particularly beneficial for processing high-resolution images or large batch-size in Computer Vision Tasks.

>### Key Features and Requiremenets
>- **Memroy Optimization** : The primary purpose of `MemEffAttention` is to reduce the memory footprint compared to standard attention moduels. It achieves this by not materializing the full attention matrix during computation, instead calculating the output more efficiently. ** This makes DINOv2 models capable of handling larger input data, like those in depth estimation or semantic segmentation tasks, with less VRAM usage.**
>- **xFormers Dependency** : The module relies on the highly optimized xFormers library to function. The DINOv2 codebase and projects that inherit from it, such as Depth Anything V2, integrate this module conditionall
>- **Attentio Map Visualization Limitation** : As `MemEffAttention` avoids explicitly computing the full attention matrix, It doesn't easily allow for the extraction and visualization of attnetion heatmaps, which is a common technique for interpreting model behavior (e.g., saliency maps)<br> To visualize attention, developers must switch to the standard Attention module implementation within the DINOv2 framework, which does produce the intermediate attention matrix.

### LoRA Hyperparameters & Recommendations:

| Hyperparameter | Function | Recommended Settings |
|--- |--- | --- |
| **LoRA Rank(`r`)** | Controls the number of trainable parameters in the LoRA adapter matrices. A higher rank increases model capacity but also memory usage. | 8, 16, 32, 64, 128, <br> Choose 16 or 32 |
| **LoRA Alpha<br>(`lora_alpha`)** | Scales the strength of the fine-tuned adjustments in relation to the rank(`r`). | `r` (standard) or `r * 2` (common heuristic) |
| **LoRA Dropout** | A regularization technique that randomly sets a fraction of LoRA activations to zero during training to prevent overfitting. **Not that useful**, so we default set it to 0. | 0 (default) to 0.1 |
| **Weight Decay** | A regularization term that penalizes large weights to prevent overfitting and improve generalization. **Don't use too large numbers!** | 0.01 (recommended) - 0.1 |
| **Warmup Steps** | Gradually increases the learning rate at the start of training. | 5-10% of total steps |
| **Scheduler Type** | Adjust the learning rate dynamically during training | `linear` or `cosine` |
| **Seed (`random_state`)** | A fixed number to ensure reproducibility of results. | Any integer (e.g., `42`, `3407`) |
| **__Target Modules__** | Specify which parts(layer) of the model you want to apply LoRA adapters to -- either the attention, the MLP, or both. <br> Attention : `q_proj, k_proj, v_proj. o_proj`<br>MLP : `gate_proj, up_proj, down_proj` | Recommended to target all major linear layers : `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. <br><br>(⚠️CAUTION⚠️ `q_proj`, `v_proj`, `gate_proj`를 넣는게 아닌, 직접 모델에서 추출해서 layer이름을 확인하고, 그에 맞춰서 넣어야 합니다.)|

# References
* [MemLoTrack : Enhancing TIR Anti-UAV Tracking with Memory-Integrated Low-Rank Adaptation](https://www.mdpi.com/1424-8220/25/23/7359)
* [LoRA : Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
* [Tracking Meets LoRA : Faster Training, Larger Model, Stronger Performance](https://arxiv.org/pdf/2403.05231)


# Related Works
* [LoRA fine-tuning Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
* [DINOv2 official Github](https://github.com/facebookresearch/dinov2)
* [Implementing LoRA from Scratch](https://medium.com/data-science/implementing-lora-from-scratch-20f838b046f1)
