---
name: medical-ai-master
description: "Core skill for medical imaging AI projects. Priority trigger for medical image task design, data pipelines, training implementation, experimental standards, model evaluation, inference deployment, or test set boundaries. High relevance keywords: segmentation, classification, detection, registration, multimodal, MONAI, nnU-Net, PyTorch Lightning. 医学影像 AI 项目的主力 skill。涉及医学影像任务设计、数据管线、训练实现、实验规范、模型评估、推理部署或测试集边界时优先触发。高相关词：分割、分类、检测、配准、多模态、MONAI、nnU-Net。"
---

# Medical AI Master (医学影像 AI 专家)

**Summary Box / 摘要**
This skill establishes the universal engineering and research standards for medical imaging AI projects. It balances clinical risk, scientific rigor, and engineering maintainability to deliver reproducible, interpretable, and scalable results.
本 skill 确立医学影像 AI 项目的通用工程与研究规范，旨在临床风险、科研严谨性与工程可维护性之间取得平衡，输出可复现、可解释、可交付的结果。

---

## 1. Applicability Boundaries / 适用边界

**Use this skill when / 优先在以下场景使用:**
1. Building medical AI projects from scratch. (从零搭建医学影像 AI 项目)
2. Upgrading existing `PyTorch`, `MONAI`, `nnU-Net`, or `PyTorch Lightning` repositories. (升级现有项目)
3. Designing or reviewing data splits, training logic, evaluation pipelines, and deployment checks. (设计或审查数据切分、训练、评估与部署流程)
4. Addressing issues related to data leakage, patient-level splitting, logging standards, test set boundaries, or statistical testing. (处理数据泄漏、患者级切分、日志规范、测试集边界等问题)

**Simplify usage when / 以下场景可简化使用:**
1. Pure terminology explanation without engineering decisions. (纯术语解释)
2. Minor syntax fixes unrelated to data splits, training logic, or test set usage. (极小型语法修补)

---

## 2. Default Stance / 默认立场

- **Data Correctness > Model Complexity:** 先保证数据正确，再谈模型复杂度。
- **Reproducibility > Single Metric Boost:** 先保证可复现和无泄漏，再追求单次指标提升。
- **Robust Baselines > Complex Structures:** 先使用稳健基线，再引入复杂结构 (大道至简 / Greatest Simplicity).
- **Verifiable Conclusions > Hype:** 先给出可验证结论，再给出性能宣传。
- **Strict Test Set Boundaries > Final Performance:** 先明确测试集边界，再讨论最终性能。

---

## 3. Task Modes / 任务模式

### Mode A: Greenfield (从零搭建)
User provides data/goals, but no mature code.
- **Workflow:** Define task/metrics -> Setup structure -> Design splits/preprocessing -> Select baseline/loss -> Implement monitoring/evaluation.

### Mode B: Brownfield Upgrade (升级现有项目)
Existing repo with messy structure, poor logging, or leakage.
- **Workflow:** Review existing structure (no full rewrite by default) -> Fix high-risk issues (leakage, missing logs, bad splits) -> Minimal necessary changes preserving business logic.

### Mode C: Targeted Support (局部专项支持)
Specific sub-problems (e.g., Loss selection, TensorBoard setup, Post-processing).
- **Workflow:** Address only the specific sub-problem without forcing a full refactor.

---

## 4. Decision Hierarchy / 任务决策顺序

1. **Task Type:** Segmentation / Classification / Detection / Registration / Multimodal
2. **Data Dimensionality:** 2D / 2.5D / 3D; Isotropic vs. Anisotropic
3. **Data Scale & Quality:** Full / Weak / Semi / Self-supervised
4. **Class Imbalance:** Foreground ratio, empty labels, small lesions
5. **Baseline & Loss:** Select robust baseline and appropriate loss function
6. **Pipeline:** Design augmentations, monitoring, thresholds, and post-processing
7. **Advanced Techniques:** Only consider Transformers, Diffusion, Flow Matching, etc., after baselines are solid.

---

## 5. Reference Document Routing / 参考文档路由

Read the following reference documents when addressing specific domains:
涉及以下领域时，必须先读取对应的参考文档：

| Domain (领域) | Document (文档) |
|---|---|
| Architecture, Loss Functions, Imbalance (架构与损失) | `references/architecture_and_loss.md` |
| Training Logs, Metrics, TensorBoard (日志与监控) | `references/tensorboard_logging.md` |
| Evaluation, Post-processing, Deployment (评估与部署) | `references/inference_and_deployment.md` |

---

## 6. Working Principles / 工作原则

- **Config-Driven (配置优先):** Experimental behavior must be driven by configuration files. (实验行为由配置文件驱动)
- **Zero Leakage (数据无泄漏):** Strict patient-level or subject-level splitting. No slice-level random splits. (必须按患者级切分，严禁切片级随机切分)
- **Evaluation Boundaries (评估有边界):** Validation set for tuning/thresholds; Test set strictly for final reporting. (验证集调参，测试集仅用于最终报告)
- **Auditable Results (结果可审计):** Save config snapshots, metrics, weights, and split info for every run. (每次实验保留配置快照、指标和切分信息)
- **Clinical Validity (临床语义有效):** Augmentations and post-processing must make anatomical sense. (增强和后处理必须符合解剖学常识)
- **Framework Agnostic (框架兼容):** Compatible with MONAI/nnU-Net/Lightning, but principles are transferable. (优先兼容主流框架，但核心原则保持可迁移)

---

## 7. Recommended Project Structure / 推荐项目结构

```text
project/
├── configs/             # YAML/JSON configs
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/          # Split files isolated from code
├── src/
│   ├── models/
│   ├── data/
│   ├── losses/
│   ├── metrics/
│   ├── inference/
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── export_model.py
├── experiments/         # Archived experiment artifacts
└── README.md
```

---

## 8. Minimal Examples / 最小示例

### Experiment Naming (实验命名)
Format: `{task}_{dataset}_{model}_{variant}_{YYYYMMDD}`
Example: `seg_brats2024_swinunetr_baseline_20250601`
*Prohibited: `test2`, `new_v3`, `final`*

### Patient-Level Split File (患者级切分文件)
Stored in `splits/split_official.json`, archived with experiments.
```json
{
  "schema_version": "1.0",
  "strategy": "stratified_k_fold",
  "unit": "subject",
  "folds": {
    "fold_0": {
      "train": ["sub-001", "sub-003"],
      "val": ["sub-002"],
      "test": ["sub-005"]
    }
  },
  "held_out_test": ["sub-099"]
}
```

### TensorBoard Namespace (TensorBoard 命名空间)
Format: `{split}/{category}/{metric}`
Example: `train/loss/seg`, `val/seg/dice_mean`, `test/calibration/ece`

---

## 9. Delivery Standards / 交付标准

When generating code or designs:
1. Explicitly state the Task Mode (A, B, or C).
2. Provide rationale for design choices, not just conclusions.
3. Limit changes to the necessary scope; do not rewrite blindly.
4. Provide a minimal verifiable path (e.g., one forward pass, one short validation loop).
5. Highlight potential clinical or engineering risks.
