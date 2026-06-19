# TensorBoard Logging Reference
# TensorBoard 日志规范

Standards for professional medical AI experiment monitoring.
医学影像 AI 专业实验监控规范。

---

## 1. Namespace Convention / 命名空间约定

Use `/` separators for logical grouping. Three-level structure: `{split}/{category}/{metric}`.
使用 `/` 分隔符进行逻辑分组，三层结构：`{split}/{category}/{metric}`。

| Namespace (命名空间) | Contents (内容) | Frequency (频率) |
|---|---|---|
| `train/loss/` | Total loss and per-term breakdown (总损失与各项分解) | Every step |
| `train/optim/` | LR, gradient norm (学习率、梯度范数) | Every step |
| `val/seg/` | Dice, HD95, IoU, NSD | Every epoch |
| `val/cls/` | AUC, F1, precision, recall | Every epoch |
| `val/calibration/` | ECE, reliability curve (ECE、可靠性曲线) | Every epoch |
| `val/vis/` | Segmentation overlays: input / GT / pred (分割叠加图) | Every N epochs |
| `diagnostics/` | Confusion matrix, PR curve (混淆矩阵、PR 曲线) | Every epoch |
| `parameters/` | Weight histograms (权重直方图) | Every N epochs |
| `gradients/` | Gradient histograms (梯度直方图) | Every N epochs |
| `system/` | GPU memory, throughput (显存、吞吐量) | Every step |
| `test/` | Final metrics — written once at test time only (最终指标，仅在测试时写入一次) | Once |

**`test/` namespace must never appear inside the training loop.**
**`test/` 命名空间严禁出现在训练循环内部。**

---

## 2. Required Metrics Suite / 必要指标集

Medical datasets are imbalanced. AUC alone is misleading. Justify any omission in your Methods section.
医学数据集存在不平衡。单独 AUC 具有误导性，任何省略都必须在方法部分说明。

| Metric (指标) | Why It Matters (重要性) | Task (任务) |
|---|---|---|
| Dice | Overlap quality (重叠质量) | Segmentation |
| HD95 | Boundary accuracy, 95th percentile Hausdorff (边界精度) | Segmentation |
| IoU | Jaccard index; complements Dice (Jaccard 指数) | Segmentation |
| NSD | Normalized surface distance for thin structures (薄结构归一化表面距离) | Segmentation |
| AUC | Threshold-independent discrimination (阈值无关判别力) | Classification |
| F1 | Harmonic mean of precision / recall (精确率/召回率调和均值) | Classification |
| ECE | Expected Calibration Error — critical for clinical use (期望校准误差，临床使用关键) | All |
| Grad norm | Gradient explosion / vanishing detection (梯度爆炸/消失检测) | All |

---

## 3. Implementation Standards / 实现规范

### 3.1 Metric Computation / 指标计算

- Use `torchmetrics` or MONAI metrics objects — do not accumulate manually. (使用 `torchmetrics` 或 MONAI 指标对象，不要手动累积)
- Cast outputs to `float32` before metric update: `probs = logits.sigmoid().float()`. Mixed precision (fp16) reduces precision on small lesions. (更新指标前将输出转为 `float32`，混合精度会降低小病灶精度)
- For spatial metrics (Dice, HD95), use MONAI's `include_background=False` and call `.aggregate()` at epoch end. (空间指标使用 `include_background=False`，在 epoch 结束时调用 `.aggregate()`)
- Reset all metric objects at the start of each epoch (`metric.reset()`). (每个 epoch 开始时重置所有指标对象)

### 3.2 Image Logging / 图像日志

- Log segmentation overlays as `[Input | Ground Truth | Prediction]` side-by-side grids. (将分割叠加图记录为并排网格)
- For 3D volumes, log three orthogonal planes (axial, coronal, sagittal) to capture full anatomical context. (3D 体积记录三个正交平面：轴位、冠状位、矢状位)
- Normalize each slice to `[0, 1]` before logging: `(x - x.min()) / (x.max() - x.min() + 1e-8)`. (记录前将每个切片归一化到 `[0, 1]`)
- Limit to first N batches (e.g., 4) and log every 5–10 epochs. Image logging is the main cause of TensorBoard disk bloat. (限制为前 N 个批次，每 5-10 个 epoch 记录一次，图像日志是 TensorBoard 磁盘膨胀的主要原因)
- Detach and move to CPU before stashing: prevents VRAM accumulation. (记录前 detach 并移至 CPU，防止显存累积)

### 3.3 Calibration Curve / 校准曲线

Plot a reliability diagram every epoch: bin predicted probabilities into 15 bins, compare mean predicted probability vs. fraction of positives. A well-calibrated model lies on the diagonal. **Overconfident models are dangerous in clinical settings.**
每个 epoch 绘制可靠性图：将预测概率分为 15 个区间，比较平均预测概率与正样本比例。校准良好的模型应落在对角线上。**过度自信的模型在临床环境中是危险的。**

### 3.4 Histograms / 直方图

Log weight and gradient histograms every 10 epochs (not every epoch — disk cost is high). Use to diagnose dead layers, gradient vanishing, or weight collapse.
每 10 个 epoch 记录权重和梯度直方图（不是每个 epoch，磁盘成本高）。用于诊断死亡层、梯度消失或权重坍塌。

### 3.5 Run Metadata / 运行元数据 (Log Once at Start)

Log at run initialization, not during training:
在运行初始化时记录，不在训练过程中记录：
- Full config YAML (完整配置 YAML)
- Git commit hash (Git 提交哈希)
- Split file checksum (MD5 of the splits JSON) (切分文件校验和)
- Environment: Python version, MONAI/PyTorch versions, GPU model (环境信息)

---

## 4. Paper-Ready Export / 论文就绪导出

Maintain a `metrics.json` per experiment with final val and test values.
每个实验维护一个 `metrics.json`，包含最终验证集和测试集指标。

```json
{
  "experiment_id": "seg_brats2024_swinunetr_baseline_20250601",
  "val":  { "dice": 0.8821, "hd95": 4.32, "auc": 0.9410, "ece": 0.031 },
  "test": { "dice": 0.8743, "hd95": 4.71, "auc": 0.9380, "ece": 0.034 },
  "ci95": { "dice": [0.861, 0.887], "auc": [0.921, 0.954] }
}
```

**Generate LaTeX table from this file programmatically — never copy numbers manually.**
**从此文件程序化生成 LaTeX 表格，永远不要手动复制数字。**
