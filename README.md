# Probing SOTA Models with Hard Examples  
### Fine-Tuned ResNet-50 on CIFAR-10 with Hard Example Analysis + Grad-CAM Visualization

---

## Abstract
This project investigates the failure behavior of a fine-tuned ResNet-50 model on the CIFAR-10 dataset by identifying and analyzing *hard examples*—specifically **misclassified samples** and **low-confidence predictions**.  
Using Grad-CAM, I visualize the model’s internal attention patterns to understand where and why it fails.  
The analysis reveals systematic issues such as attention drift, background bias, and ambiguity-driven uncertainty.  
Based on these insights, several strategies are proposed to improve model robustness and interpretability.

---

## Development & Execution Environment

This project was implemented using a combination of local (windows powershell) and google colab(T4 GPU):

### **Week 1 – Local Setup (Windows PowerShell)**
- Project initialization, environment setup, and early script testing were performed locally on Windows PowerShell.
- Python virtual environment (`.venv`) and dependency installation were performed locally.

### **Week 2 – Model Training on Google Colab (T4 GPU)**
- Due to limited compute on local hardware, fine-tuning the ResNet-50 model was performed on Google Colab.
- Colab’s NVIDIA T4 GPU was used to accelerate training.
- The `train.py` script was executed in the Colab environment.

### **Week 3 – Execution on Colab**
- Hard example extraction (`evaluate.py`, `hard_examples.py`) was executed on colab.
- Grad-CAM visualization (`gradcam.py`) was executed on Colab depending on resource needs.
- Final visualizations were produced in Colab for GPU efficiency.

### **Week 4 - Final conclusions and Recommendations**

### **Code Format**
- All project code is implemented as `.py` scripts, not `.ipynb` notebooks.
- This ensures cleaner version control and reproducibility.

---

## Dataset & Model
- **Dataset:** CIFAR-10  
  - 50,000 training images  
  - 10,000 test images  
- **Model:** ResNet-50 pretrained on ImageNet and fine-tuned for 3 epochs  
- **Training Hardware:** Google Colab T4 GPU  
- **Key Outputs:**
  - `test_predictions_finetuned.csv`
  - `misclassified.csv` (406 samples)
  - `low_confidence.csv` (21 samples)
  - Grad-CAM visualizations for both categories

---

## Model Training
A ResNet-50 model was fine-tuned on CIFAR-10 using:

- Cross-entropy loss  
- Adam optimizer  
- Small number of epochs for faster experimental iteration  

This produced:

- `resnet50_finetuned.pth`

---

## Week 3 — Hard Example Extraction

Hard examples were identified using:

### **1. Misclassified Samples**
- `pred != true_label`

### **2. Low-Confidence Samples**
- `max_softmax_probability < 0.40`

Scripts used:
python scripts/evaluate.py, 
python scripts/hard_examples.py

Outputs generated:

- `misclassified.csv`
- `low_confidence.csv`

---

## Week 3 — Grad-CAM Visualization

Grad-CAM visual explanations were generated for:

- Misclassified samples  
- Low-confidence samples  

Command used:

python scripts/gradcam.py

Images saved under:
gradcam_outputs/misclassified/, 
gradcam_outputs/lowconf/

---

# **Misclassified Example Visualizations**

Below are  4 selected misclassified Grad-CAM samples:

<img width="384" height="384" alt="gradcam_470" src="https://github.com/user-attachments/assets/35e58b3f-84dc-4d54-bbc6-6064db3ce9f0" />

<img width="384" height="384" alt="gradcam_655" src="https://github.com/user-attachments/assets/592ef9d2-49dd-4827-856d-148ab28b5048" />

<img width="384" height="384" alt="gradcam_956" src="https://github.com/user-attachments/assets/d20846b8-236c-4a80-92eb-6706ccfe421f" />

<img width="384" height="384" alt="gradcam_1056" src="https://github.com/user-attachments/assets/69cded2b-04d1-4650-9735-0a6ed735d5b9" />

---

## **Misclassification Analysis**
Key patterns observed:

### **1. Attention Drift**
The heatmap often highlights background textures or irrelevant edges, showing that the model struggles to localize the correct object.

### **2. Similar-Class Confusion**
Classes with visual similarity (e.g., cat vs dog, truck vs automobile) cause overlapping activation patterns.

### **3. Weak or Partial Object Visibility**
Images where the object is small or partially visible result in incorrect predictions and scattered activations.

---

# Low-Confidence Example Visualizations

Below are 2 selected low-confidence samples:

<img width="384" height="384" alt="gradcam_lowconf_355" src="https://github.com/user-attachments/assets/e28bede9-6c0c-4650-a7c2-dfc42c397f32" />

<img width="384" height="384" alt="gradcam_lowconf_4244" src="https://github.com/user-attachments/assets/c7a3b895-5f50-4e85-bf23-6599158bebcb" />

---

## **Low-Confidence Analysis**

- Grad-CAM maps show **diffuse, weak activations**, indicating uncertainty.  
- The model fails to lock onto a specific object region.  
- Low-confidence often occurs in images with:
  - Blur  
  - Occlusion  
  - Poor contrast  
  - Ambiguous shapes  

---

# Key Insights & Failure Modes

### **1. Background Bias**
Model overattends to background structures instead of the object.

### **2. Poor Object Localization**
Misclassified and low-confidence examples show failure to focus on the actual object.

### **3. Ambiguity-Driven Errors**
Blurred or occluded objects lead to inconsistent activations and low confidence.

---

# Recommendations for Improvement

### **1. Stronger Data Augmentation**
Applying augmentations such as:
- CutMix  
- MixUp  
- ColorJitter  
- Random Erasing  

can improve generalization.

### **2. Hard Example Mining**
Re-train using misclassified and low-confidence samples to address model weaknesses.

### **3. Longer Training or Larger Batch Size**
Adds stability and improves feature extraction.

---

#  Repository Structure

```
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── hard_examples.py
│   ├── gradcam.py
│   ├── baseline_knn.py
│   ├── data.py
│   ├── utils.py
│
├── models/
│   └── resnet50_finetuned.pth
│
├── gradcam_outputs/
│   ├── misclassified/
│   │   ├── gradcam_misclassified_xxx.png
│   │   └── ...
│   ├── lowconf/
│       ├── gradcam_lowconf_xxx.png
│       └── ...
│
├── results/
│   ├── test_predictions_finetuned.csv
│   ├── test_predictions_baseline.csv
│   ├── misclassified.csv
│   ├── low_confidence.csv
│   ├── confusion_matrix_finetuned.csv
│   ├── confusion_matrix.csv
│   ├── classification_report.txt
│   ├── class_names.csv
│   ├── train_log.csv
│
├── week_logs/
│   ├── week1log.txt
│   ├── week2log.txt
│   ├── week3log.txt
│   ├── week4log.txt
│   ├── week5log.txt
│
├── requirements.txt
└── README.md
```

## Installation

clone:
```
$ git clone https://github.com/abdr26/CAP6415_F25_project-Probing-SOTA-models-with-hard-examples
$ cd CAP6415_F25_project-Probing-SOTA-models-with-hard-examples
```

If using windows powershell activate venv:
```
$ .\.venv\Scripts\Activate.ps1
```

Install dependencies:
```
$ pip install -r requirements.txt
```

---

# How to Run the Project in colab

### **Train**
```
$ !python scripts/train.py
```

### **Evaluate**
```
$ !python scripts/evaluate.py
```

### **Extract Hard Examples**
```
$ !python scripts/hard_examples.py
```

### **Generate Grad-CAM Visualizations**
```
$ !python scripts/gradcam.py
```

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

# Conclusion
This project successfully identifies and visualizes model weaknesses using misclassified and low-confidence examples.  
Grad-CAM proves highly effective in interpreting CNN behavior and diagnosing failure patterns.  
The insights gained help guide future improvements to model design and training strategy.

---
