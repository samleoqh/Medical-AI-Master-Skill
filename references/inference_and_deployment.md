# Inference, Testing, and Clinical Deployment
# 推理、测试与临床部署

Standards for rigorous evaluation, post-processing, and clinical deployment readiness.
严格评估、后处理与临床部署就绪性规范。

---

## 1. Test Set Discipline / 测试集纪律

**The test set is evaluated exactly once, at the very end of the project.**
**测试集在项目最终阶段仅评估一次。**

- Never evaluate on the test set during hyperparameter tuning or architecture search — use validation set or cross-validation for all decisions. (超参数调整或架构搜索期间绝不评估测试集，所有决策使用验证集或交叉验证)
- Enforce patient-level separation across train / val / test. A patient appearing in train must never appear in test, even across timepoints or modalities. (强制执行患者级分离，即使跨时间点或模态也不例外)
- Threshold selection and post-processing parameters must come from the validation set, not the test set. (阈值选择和后处理参数必须来自验证集，不得来自测试集)

---

## 2. Statistical Rigor / 统计严谨性

A higher Dice or AUC on a single test set is insufficient for publication. Required:
单个测试集上更高的 Dice 或 AUC 不足以发表。必须包含：

| Requirement (要求) | Method (方法) |
|---|---|
| Variance estimate (方差估计) | 5-fold cross-validation; report mean ± std (5 折交叉验证，报告均值 ± 标准差) |
| Continuous metric comparison (连续指标比较) | Wilcoxon signed-rank test (Wilcoxon 符号秩检验) |
| AUC comparison (AUC 比较) | DeLong's test (DeLong 检验) |
| Confidence intervals (置信区间) | Bootstrapping (1000 resamples), report 95% CI (自举法 1000 次重采样，报告 95% CI) |

Report all three together: point estimate, CI, and p-value vs. baseline.
同时报告三项：点估计、置信区间和与基线的 p 值。

---

## 3. Post-Processing / 后处理

Raw model outputs often contain anatomical impossibilities. Post-processing is mandatory before reporting final metrics.
原始模型输出通常包含解剖学上不可能的结果，报告最终指标前后处理是必须的。

| Issue (问题) | Method (方法) | When to Use (使用时机) |
|---|---|---|
| Scattered false positives (散在假阳性) | Keep largest connected component (保留最大连通组件) | Single-organ segmentation (单器官分割) |
| Internal holes (内部空洞) | Fill holes morphologically (形态学填充空洞) | Solid organ masks (实质器官掩码) |
| Noisy boundary (噪声边界) | Binary threshold with rounding (二值阈值+取整) | Always (始终使用) |
| Multi-lesion cases (多病灶情况) | Size-filtered component analysis (尺寸过滤连通分析) | Do NOT use largest-only for multi-lesion (多病灶不能只保留最大) |

**Validate that post-processing improves val metrics before applying to test.**
**在应用到测试集之前，验证后处理确实改善了验证集指标。**

---

## 4. Threshold Optimization / 阈值优化

The default threshold of 0.5 is rarely optimal on imbalanced medical data.
默认阈值 0.5 在不平衡医学数据上很少是最优的。

- Use the **validation set** PR curve to find the threshold maximizing F1. (使用**验证集** PR 曲线找到最大化 F1 的阈值)
- Alternatively, use Youden's J statistic (`sensitivity + specificity − 1`) on the ROC curve. (或在 ROC 曲线上使用 Youden's J 统计量)
- Apply the selected threshold to the test set inference — do not re-optimize on test. (将选定阈值应用于测试集推理，不在测试集上重新优化)

---

## 5. Test-Time Augmentation (TTA) / 测试时增强

TTA improves robustness by averaging predictions across augmented views.
TTA 通过对增强视图的预测取平均来提高鲁棒性。

- Typical TTA set: original + horizontal flip + vertical flip → average probabilities → reverse spatial transforms. (典型 TTA：原始 + 水平翻转 + 垂直翻转 → 平均概率 → 反转空间变换)
- Only use augmentations that preserve anatomical validity (e.g., do not apply random rotations to brain MRI if orientation is fixed in training). (只使用保持解剖有效性的增强)
- TTA adds inference time proportional to augmentation count; document latency impact. (TTA 增加与增强数量成比例的推理时间，记录延迟影响)

---

## 6. Calibration and Uncertainty / 校准与不确定性

Clinicians need to know when the model is uncertain. Required before clinical deployment:
临床医生需要知道模型何时不确定，临床部署前必须完成：

- **Calibration (校准):** Plot a reliability diagram (ECE). If the model predicts 90% probability, it should be correct 90% of the time. Overconfident models are dangerous. (绘制可靠性图，过度自信的模型是危险的)
- **Uncertainty estimation (不确定性估计):** Use Monte Carlo Dropout or deep ensembles (3–5 models) to generate uncertainty maps. Flag highly uncertain cases for manual radiologist review. (使用 MC Dropout 或深度集成生成不确定性图，标记高不确定性病例供放射科医生审查)
- **Calibration fix (校准修正):** Apply temperature scaling on the validation set if ECE > 0.05. (若 ECE > 0.05，在验证集上应用温度缩放)

---

## 7. Export and Optimization / 导出与优化

| Step (步骤) | Recommended Approach (推荐方案) |
|---|---|
| Export (导出) | ONNX (cross-platform) or TorchScript (PyTorch-native) |
| GPU optimization (GPU 优化) | TensorRT (NVIDIA) or OpenVINO (Intel / CPU) |
| Benchmarking (基准测试) | Document: inference latency (s/volume), peak VRAM (GB), throughput (记录推理延迟、峰值显存、吞吐量) |

**Document minimum hardware requirements. A model requiring 24 GB VRAM cannot be deployed on standard hospital workstations (typically 8–16 GB).**
**记录最低硬件要求。需要 24 GB 显存的模型无法部署在标准医院工作站（通常 8-16 GB）上。**

---

## 8. Input Validation (Guardrails) / 输入验证（护栏）

Clinical data is messy. Reject invalid inputs before they reach the model:
临床数据是混乱的，在无效输入到达模型之前拒绝它们：

- Verify DICOM modality tag matches training modality (reject MRI if trained on CT). (验证 DICOM 模态标签与训练模态匹配)
- Check voxel spacing and image dimensions are within training distribution bounds. (检查体素间距和图像尺寸在训练分布范围内)
- Detect corrupted, truncated, or zero-padded volumes. (检测损坏、截断或零填充的体积)
- Log all rejections with reason — silent failures are unacceptable in clinical pipelines. (记录所有拒绝及原因，临床管线中静默失败是不可接受的)
