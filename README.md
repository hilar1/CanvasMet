# 金相智能分析系统 (CanvasMet) v1.1

AiMetallographic 是一款面向材料科学领域的端到端金相图像智能分析平台。系统基于 PySide6 构建了现代化 Canvas-First 桌面端图形界面，底层深度集成 Cellpose 实例分割算法，实现了金相显微图像的自动化晶粒识别、人工交互修正、ASTM E112 物理指标计算以及闭环的模型微调训练（数据飞轮）。

## 核心特性 (Key Features)

* **沉浸式交互界面 (Canvas-First UI):** 采用顶部工具栏与无边框画布设计，提供动态悬浮的防重叠撤销/重做（Undo/Redo）胶囊栏。
* **高精度实例分割:** 底层基于 PyTorch 加速的深度学习引擎，实现复杂晶界的高效分割。
* **丰富的预训练基座:** 内置 `CP`, `cpsam`, `cyto2`, `cyto`, `LC3`, `LC4`, `livecell`, `nucl`, `tis`, `TN1`, `TN2`, `TN3` 等十余款基础模型，适配多种显微成像场景。
* **实时物理学报告:** 支持动态修改像素换算率（μm/px），实时输出有效晶粒数、ASTM 晶粒度（G值）与平均截距。
* **增量历史状态机:** 工业级状态追溯机制，支持多步无损撤销与重做，保障人工修正过程的数据安全。
* **数据飞轮 (闭环训练):** 内置微调（Fine-tuning）引擎，允许用户将当前修正结果导出为训练样本，并在本地一键启动模型再训练，持续进化特定材质的识别准度。

## 环境依赖与安装指引

本项目包含高强度的矩阵运算与 GPU 显存调度，请严格按照以下步骤配置物理环境。

### 1. 基础环境
* 操作系统：Windows 10/11 或 Linux (Ubuntu 20.04+)
* Python 版本：Python 3.10 或更高版本

### 2. GPU 驱动与计算加速库 (关键)
为了启用模型推理与训练的硬件加速，**必须优先安装与本地显卡驱动（CUDA）版本匹配的 PyTorch**。请勿直接使用标准 pip 命令安装全局依赖。

以 CUDA 11.8 为例，在终端执行：
```bash

pip install torch torchvision --index-url \[https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
