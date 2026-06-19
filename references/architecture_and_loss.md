# Architecture and Loss Function Selection
# 架构与损失函数选择

Decision reference for model architecture and loss function choices in medical imaging AI.
医学影像 AI 中模型架构与损失函数选择的决策参考。

---

## 1. Architecture Selection / 架构选择

### 1.1 Dimensionality Strategy / 维度策略

| Modality / Slice Thickness (模态/层厚) | Recommended Approach (推荐方案) | Rationale (理由) |
|---|---|---|
| X-Ray, Fundus, Pathology | 2D CNN (ResNet / EfficientNet) | Native 2D; ImageNet pretraining transfers well (天然 2D，ImageNet 预训练迁移效果好) |
| CT / MRI — thick slice / anisotropic | 2.5D or slice-by-slice 2D | Poor Z-resolution makes true 3D unreliable (Z 轴分辨率低，3D 卷积不可靠) |
| CT / MRI — thin slice / isotropic | 3D patch-based (UNet3D / SwinUNETR) | Full 3D context required; patch training manages memory (需要全 3D 上下文，分块训练控制显存) |
| Multimodal (MRI sequences, PET-CT) | Late fusion or shared encoder | Align to common space first; fuse at feature or decision level (先对齐，再在特征或决策层融合) |

### 1.2 Encoder Selection / 编码器选择

**Always establish a standard UNet (segmentation) or ResNet (classification) baseline first. Report this baseline before introducing any complexity.**
**永远先建立标准 UNet（分割）或 ResNet（分类）基线，在引入任何复杂结构前先报告该基线。**

- **Transformers (SwinUNETR / UNETR):** Only when dataset ≥ 1000 volumes or long-range dependencies are critical (e.g., multi-organ). Transformers lack CNN inductive bias and overfit easily on small datasets. (仅在数据集 ≥ 1000 个体积或需要长程依赖时使用，小数据集易过拟合)
- **Pretraining (预训练):** Mandatory for 2D tasks (ImageNet). For 3D, consider self-supervised pretraining (masked volume modeling) when unlabeled data is available. (2D 任务必须使用 ImageNet 预训练；3D 任务在有无标签数据时考虑自监督预训练)
- **Generative / Denoising (生成/去噪):** Use LDM or DDPM for synthesis; flow matching for continuous-time longitudinal modeling; VAE-GAN when latent structure matters. Always validate generated images clinically, not just by FID. (生成任务用 LDM/DDPM；纵向建模用 flow matching；始终进行临床验证，不只看 FID)

---

## 2. Loss Function Selection / 损失函数选择

### 2.1 Segmentation / 分割

| Scenario (场景) | Loss (损失) | Notes (说明) |
|---|---|---|
| Balanced classes (类别均衡) | Cross-Entropy (CE) | Stable baseline (稳定基线) |
| Mild imbalance (轻度不平衡) | Dice | Region-level; handles foreground sparsity (区域级，处理前景稀疏) |
| Severe imbalance (严重不平衡) | Dice + CE | Combine region and pixel objectives (结合区域与像素目标) |
| Extremely small lesions (极小病灶) | Dice + Focal | Focal down-weights easy negatives (Focal 降低简单负样本权重) |
| Boundary precision required (边界精度要求高) | + Boundary Loss / clDice | Add as auxiliary term, not replacement (作为辅助项添加，不替换主损失) |

**Compound loss defaults (复合损失默认值):** Start with equal weights (`λ_dice=1.0, λ_ce=1.0`). Consider CE-only warm-up for the first 10 epochs — Dice gradients are unstable when predictions are random. (从等权重开始；前 10 个 epoch 考虑仅用 CE 热身，随机预测时 Dice 梯度不稳定)

**Background class (背景类):** Include in CE computation, exclude from Dice metric (`include_background=False`). (CE 中包含背景，Dice 指标中排除背景)

### 2.2 Classification / 分类

Use BCE for binary, CE for multi-class. Add focal weighting (`gamma=2`) for severe imbalance. Always report AUC alongside F1 — AUC alone is misleading on imbalanced data.
二分类用 BCE，多分类用 CE。严重不平衡时加 Focal 权重（`gamma=2`）。始终同时报告 AUC 和 F1，单独 AUC 在不平衡数据上具有误导性。

### 2.3 Generative / Self-Supervised / 生成与自监督

| Task (任务) | Loss (损失) |
|---|---|
| Image synthesis (diffusion / flow) (图像合成) | MSE or L1 on denoising target; perceptual loss optional |
| VAE | Reconstruction (L1/L2) + KL divergence; β-VAE for disentanglement |
| Contrastive (SimCLR / MoCo) (对比学习) | NT-Xent / InfoNCE; hard negative mining for medical data |
| Denoising (去噪) | L2 on clean target; mask-weighted if anatomy is localized |

### 2.4 Empty Labels and Patch Training / 空标签与分块训练

In patch-based 3D training, many patches contain only background.
在 3D 分块训练中，许多分块只包含背景。

- Dice is undefined (0/0) on empty patches → use smoothing (`smooth=1e-5`; MONAI `DiceLoss` handles this internally). (空分块上 Dice 未定义，使用平滑处理)
- Use `RandCropByPosNegLabeld` to guarantee a fixed positive-to-negative patch ratio per batch. (使用 `RandCropByPosNegLabeld` 保证每批次固定的正负样本比例)

---

## 3. Deep Supervision / 深监督

For deep architectures (UNet3D, SwinUNETR), gradients vanish before reaching the bottleneck. Deep supervision forces intermediate decoders to learn.
对于深层架构（UNet3D、SwinUNETR），梯度在到达瓶颈层前消失。深监督强制中间解码器学习。

1. Extract predictions at multiple decoder resolutions. (在多个解码器分辨率提取预测)
2. Downsample ground truth to match each resolution. (将 GT 下采样以匹配各分辨率)
3. Compute loss at each scale with decaying weights: `1.0 → 0.5 → 0.25`. (在每个尺度用衰减权重计算损失)

**Disable deep supervision at validation and inference. (验证和推理时禁用深监督)**
