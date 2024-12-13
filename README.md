# DFREC &mdash; official PyTorch Implementation

![](./assets/teaser.png)
![](./assets/framework.png)
![issueBadge](https://img.shields.io/github/issues/botianzhe/DFREC)   ![starBadge](https://img.shields.io/github/stars/botianzhe/DFREC)   ![repoSize](https://img.shields.io/github/repo-size/botianzhe/DFREC)  ![lastCommit](https://img.shields.io/github/last-commit/botianzhe/DFREC) 

Official Implementation of [DFREC:DeepFake Identity Recovery Based on Identity-aware Masked Autoencoder](https://openreview.net/pdf?id=nxVUqDXJZG)

checkpoint path：https://drive.google.com/drive/folders/1ZXH-7QTy5P-o1zfhY7myUeTIhXUj_JmS?usp=sharing

## Inference
To inference the DFREC, run this command:

```bash
#unzip the segmentation models codes
unzip segmentation_models.zip
#Prepare the environment required for the project
python -m pip install -r requirements.txt
#and then you can run the inference code
python DFREC_eval.py
```

## Implementation Author

Peipeng Yu @ Jinan University, Guangzhou, China. (ypp865@163.com)

## Paper Information

```bibtex
@inproceedings{Yu2024DFRECDI, title={DFREC: DeepFake Identity Recovery Based on Identity-aware Masked Autoencoder}, author={Peipeng Yu and Hui Gao and Zhitao Huang and Zhihua Xia and Chip-Hong Chang}, year={2024}, url={https://api.semanticscholar.org/CorpusID:274610259} }
```

