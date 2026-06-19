# Medical AI Master Skill

**A universal engineering and research standard for medical imaging AI projects.**
**面向医学影像 AI 项目的通用工程与研究规范。**

This repository covers experimental design, data leakage prevention, patient-level splitting, model and loss function selection, training monitoring, evaluation standards, inference deployment, and result reproducibility.
本仓库覆盖实验设计、数据泄漏防护、患者级切分、模型与损失函数选择、训练监控、评估规范、推理部署与结果可复现性。

---

## Repository Contents / 仓库内容

```text
Medical-AI-Master-Skill/
├── SKILL.md                            # Main skill definition (主 skill 定义)
├── references/
│   ├── architecture_and_loss.md        # Architecture & loss selection (架构与损失函数)
│   ├── tensorboard_logging.md          # Logging & monitoring standards (日志与监控规范)
│   └── inference_and_deployment.md     # Evaluation, post-processing, deployment (评估与部署)
├── README.md
├── LICENSE
└── .gitignore
```

---

## Design Goals / 设计目标

This skill is not a fixed framework recipe. It distills a transferable set of medical imaging AI engineering standards with a clear priority order:
本 skill 不是某个框架的唯一写法，而是沉淀一套可迁移的医学影像 AI 工程规范，遵循明确的优先级顺序：

1. **Data Correctness > Model Complexity** — Ensure data is correct before discussing model complexity. (先确保数据正确，再讨论模型复杂度)
2. **Reproducibility > Metrics** — Ensure split integrity before reporting performance numbers. (先确保切分无泄漏，再讨论性能数字)
3. **Robust Baselines > Complex Tricks** — Establish a solid baseline before introducing advanced techniques. (先建立稳健基线，再引入复杂技巧)
4. **Auditable Results > Demos** — Ensure results are reproducible and auditable before publishing. (先保证结果可复现可审计，再做论文或演示输出)

---

## Applicable Scenarios / 适用范围

This skill is best used for:
本 skill 优先用于：

- Medical image segmentation, classification, detection, generation, denoising, registration, and multi-task learning. (医学图像分割、分类、检测、生成、去噪、配准、多任务学习)
- Projects using `MONAI`, `nnU-Net`, `PyTorch Lightning`, `PyTorch`, or `SimpleITK`. (MONAI、nnU-Net、PyTorch Lightning 相关项目)
- Building medical imaging AI codebases from scratch. (从零搭建医学影像 AI 代码库)
- Standardizing existing research code. (对已有研究代码做规范化升级)
- Unifying training logs, experiment naming, test set boundaries, and evaluation standards. (统一训练日志、实验命名、测试集边界和评估标准)

**Simplify usage for / 以下情况可以简化使用:**
- Pure terminology explanation. (纯术语解释)
- Minor syntax fixes unrelated to engineering decisions. (不涉及工程决策的极小型语法修补)

---

## Usage / 使用方式

### Option 1: Direct Copy (直接复制)
Copy `SKILL.md` and the `references/` directory into your skill directory.

### Option 2: Fork as Template (作为模板仓库)
Fork this repository and customize:
- Trigger keywords (触发词)
- Framework preferences (框架偏好)
- Minimal examples (最小示例)
- Referenced documents (引用的参考文档)

---

## Key Scenarios / 推荐使用场景

### Greenfield (从零搭建)
When you have data and task goals but no mature training code, use this skill to constrain project structure, data splits, experiment naming, monitoring, and evaluation.

### Brownfield Upgrade (升级现有项目)
When an existing repository has unclear data splits, insufficient logs, misused test sets, or scattered configurations, this skill provides structured remediation.

### Targeted Support (局部专项支持)
Use individual reference documents for specific sub-problems:
- Architecture & loss selection → `references/architecture_and_loss.md`
- TensorBoard design → `references/tensorboard_logging.md`
- Inference & post-processing → `references/inference_and_deployment.md`

---

## Open Source Release Checklist / 开源发布建议

Before publishing a public version based on this skill:
1. Remove any internal paths, hospital names, patient information, or non-public project identifiers. (删除内部路径、医院名称、患者信息)
2. Confirm that example data and split examples contain no real identity information. (确认示例不含真实身份信息)
3. Clearly state applicable and non-applicable boundaries in `README.md`. (明确适用边界与非适用边界)
4. If significantly modified, add a version difference note on the repository homepage. (若修改较多，补充版本差异说明)

---

## Disclaimer / 注意事项

This repository provides research and engineering standard recommendations. It does not constitute medical advice, clinical certification, regulatory approval, or medical device compliance conclusions.
本仓库提供的是研究与工程规范建议，不构成医疗建议，也不代表任何临床认证、注册审批或医疗器械合规结论。

If your system will enter a real clinical environment, you must separately complete:
如果你的系统将进入真实临床环境，仍需单独完成：
- Data compliance review (数据合规审查)
- Privacy and security assessment (隐私与安全评估)
- Model calibration and risk assessment (模型校准与风险评估)
- Deployment environment validation (部署环境验证)
- Clinical trial or formal approval process (临床试用或正式审批流程)

---

## License / 许可证

This repository uses the `MIT License`. You are free to use, modify, and distribute it, but please retain the original license text.
本仓库使用 `MIT License`。你可以自由使用、修改和分发，但请保留原始许可证文本。
