## LoRA Hyperparameters & Recommendations:

| Hyperparameter | Function | Recommended Settings |
|--- |--- | --- |
| **LoRA Rank(`r`)** | Controls the number of trainable parameters in the LoRA adapter matrices. A higher rank increases model capacity but also memory usage. | 8, 16, 32, 64, 128, <br> Choose 16 or 32 |
| **LoRA Alpha<br>(`lora_alpha`)** | Scales the strength of the fine-tuned adjustments in relation to the rank(`r`). | `r` (standard) or `r * 2` (common heuristic) |
| **LoRA Dropout** | A regularization technique that randomly sets a fraction of LoRA activations to zero during training to prevent overfitting. **Not that useful**, so we default set it to 0. | 0 (default) to 0.1 |
| **Weight Decay** | A regularization term that penalizes large weights to prevent overfitting and improve generalization. **Don't use too large numbers!** | 0.01 (recommended) - 0.1 |
| **Warmup Steps** | Gradually increases the learning rate at the start of training. | 5-10% of total steps |
| **Scheduler Type** | Adjust the learning rate dynamically during training | `linear` or `cosine` |
| **Seed (`random_state`)** | A fixed number to ensure reproducibility of results. | Any integer (e.g., `42`, `3407`) |
| **__Target Modules__** | Specify which parts of the model you want to apply LoRA adapters to -- either the attention, the MLP, or both. <br> Attention : `q_proj, k_proj, v_proj. o_proj`<br>MLP : `gate_proj, up_proj, down_proj` | Recommended to target all major linear layers : `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. <br><br>(⚠️CAUTION⚠️ `q_proj`, `v_proj`, `gate_proj`를 넣는게 아닌, 직접 모델에서 추출해서 layer이름을 확인하고, 그에 맞춰서 넣어야 합니다.)|
