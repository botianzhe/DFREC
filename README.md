# DFREC &mdash; official PyTorch Implementation

# FaceShifter &mdash; Unofficial PyTorch Implementation
![](./assets/teaser_v8.jpg)
![](./assets/deepfake_method_stage1_v8.png)
![issueBadge](https://img.shields.io/github/issues/mindslab-ai/faceshifter)   ![starBadge](https://img.shields.io/github/stars/mindslab-ai/faceshifter)   ![repoSize](https://img.shields.io/github/repo-size/mindslab-ai/faceshifter)  ![lastCommit](https://img.shields.io/github/last-commit/mindslab-ai/faceshifter) 

Unofficial Implementation of [FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping](https://arxiv.org/abs/1912.13457) with [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
In the paper, there are two networks for full pipe-line, AEI-Net and HEAR-Net. We only implement the AEI-Net, which is main network for face swapping.







unzip segmentation_models.zip

python -m pip install -r requirements.txt

python DFREC_eval.py


checkpoint path：https://drive.google.com/drive/folders/1ZXH-7QTy5P-o1zfhY7myUeTIhXUj_JmS?usp=sharing
