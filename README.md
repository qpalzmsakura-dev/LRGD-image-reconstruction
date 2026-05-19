# LRGD Diffusion Model for Low-SNR Image Reconstruction
基于 LRGD 扩散模型的低信噪比鲁棒图像重建研究

---

## 项目概述
本项目在 LRGD 扩散模型的基础上，针对低信噪比环境下的图像重建任务进行改进与优化。通过调整模型引导策略与噪声调度方案，提升了模型在极端信道条件下的重建质量与鲁棒性，在公开数据集上验证了改进方案的有效性。

## 技术栈
- Python 3.x
- PyTorch
- 扩散模型（Diffusion Model）
- 低信噪比图像重建
- 指标评估（PSNR / SSIM / LPIPS / FID）

## 核心改进与亮点 ✨
1.  **自适应引导策略优化**
    针对低信噪比场景，优化模型引导强度与采样步长，提升模型对噪声的鲁棒性，减少伪影生成。
2.  **噪声调度方案调整**
    改进扩散过程的噪声调度策略，优化低信噪比阶段的重建效果，提升图像细节还原能力。
3.  **多场景对比实验**
    搭建多信噪比梯度实验环境，对比不同引导策略与噪声调度方案下的重建性能，完成消融实验与结果分析。
4.  **多维度指标评估**
    基于 PSNR/SSIM/LPIPS/FID 等指标，对重建图像的质量、感知一致性与生成效率进行综合评估。

## 项目结构
├── main.py # 项目入口，主程序与测试流程
├── train.py # 模型训练脚本，含自适应引导策略实现
├── model.py # LRGD 扩散模型结构定义
├── utils.py # 通用工具函数
├── evaluation.py # 多维度指标评估脚本
└── requirements.txt # 依赖库列表

## 运行方式
1. 安装依赖：
```bash
pip install -r requirements.txt
python main.py --mode test
python evaluation.py
