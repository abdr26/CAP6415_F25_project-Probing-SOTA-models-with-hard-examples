# Probing-SOTA-models-with-hard-examples

## Abstract

Modern deep learning models achieve impressive accuracy on benchmark datasets, but with *hard examples*(images that are ambiguous, occluded, noisy, or visually similar to other classes.) their performance often drops sharply.
This project investigates how *state-of-the-art (SOTA)* computer vision models handle such challenging inputs, with the goal of understanding *where* and *why* they fail.  

I focus on probing pretrained convolutional and transformer architectures, beginning with **ResNet-50**, later **Vision Transformer (ViT)** using the **CIFAR-10** dataset.  
My pipeline extracts deep feature representations from these models, classifies them using lightweight methods (e.g., k-Nearest Neighbors, fine-tuned heads), and identifies *hard examples* based on model confidence and misclassification patterns.  
Further analysis employs **Grad-CAM** visualizations and embedding similarity metrics to interpret model weaknesses and decision biases.

The implementation relies on the **PyTorch** deep learning framework (`torch`, `torchvision`), with supporting libraries such as **scikit-learn**, **NumPy**, **Pandas**, and **Matplotlib** for analysis and visualization.  
Baseline feature extraction builds upon open-source pretrained weights released by the **PyTorch model zoo** (ResNet-50 trained on ImageNet).  
Grad-CAM visualization methods are adapted from the open-source library *pytorch-grad-cam* (by Jacob Gildenblat et al., MIT license).  

This work does not claim originality in model architecture or training design; rather, it focuses on *systematically probing and interpreting* the behavior of existing SOTA models under difficult visual conditions.

---

## Frameworks and Libraries

- **PyTorch / Torchvision** – Model loading, feature extraction, and training  
- **scikit-learn** – kNN baseline and performance metrics  
- **pytorch-grad-cam** – Model explanation and visualization  
- **NumPy, Pandas, Matplotlib** – Data handling and plots

---

### Attribution
- **ResNet-50 architecture and pretrained weights:** *Kaiming He et al., “Deep Residual Learning for Image Recognition,” CVPR 2016.*  
- **Vision Transformer (ViT):** *Alexey Dosovitskiy et al., “An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale,” ICLR 2021.*  
- **Grad-CAM methodology:** *Ramprasaath R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” ICCV 2017.*  
- **pytorch-grad-cam implementation:** [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

---

### Run commands
```powershell

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\scripts\baseline_knn.py
