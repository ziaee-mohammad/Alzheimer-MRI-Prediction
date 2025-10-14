# 🧠 Alzheimer's Disease Prediction from MRI (OASIS Dataset)

Early prediction of **Alzheimer’s Disease (AD)** using **MRI-based features** and a diverse set of **machine learning** and **deep learning** models.  
This repository includes end‑to‑end data analysis, model development, **rigorous evaluation** (ROC‑AUC / Precision / Recall), and **interpretability** via Grad‑CAM.

---

## 📖 Overview
This project explores multiple approaches to predict Alzheimer’s diagnosis from patient-level information with an emphasis on **clinical metrics** and **reproducibility**. We compare classical ML (Logit, LDA, CART, Random Forest, Bagging, Boosting, SVM), shallow/deep neural networks, and visualize CNN attention with **Grad‑CAM**.  
The goal is to build reliable models that support early detection and can generalize without data leakage.

**Highlights**
- 74,283 samples, 25 features (4 numeric + 20 categorical)  
- Strict train/validation/test protocol (no leakage; scalers/encoders fit on train only)  
- Model selection with cross‑validation; evaluation using ROC‑AUC, Precision, Recall  
- Clinical orientation: prioritize **Recall** when minimizing False Negatives matters  
- Model interpretability with **Grad‑CAM** for CNN variants

---

## 🗂️ Dataset
- **Source**: Alzheimer’s Prediction dataset (public/open source).  
- **Target**: Binary label — Alzheimer’s diagnosis (Yes/No).  
- **Numeric features**: `Age`, `BMI`, `Education Level (years)`, `Cognitive Test Score`.  
- **Categorical features** (examples): `Gender`, `Genetic Risk Factor`, `Family History`, `Smoking`, `Physical Activity`, `Sleep Quality`, `Diabetes`, `Hypertension`, `Income Level`, etc.  
- **Missing values**: none observed in the analyzed snapshot.  
- **Standardization**: z‑score scaling for numeric features.  
- **Encoding**: categorical variables cast to factor / one‑hot encoded in ML pipelines.

> ⚠️ **Important**: If you work with MRI images, ensure patient‑level splits (subject‑wise) to avoid leakage across train/test when multiple scans per subject exist.

---

## 🧪 Experimental Setup
- **Split**: 70% train / 30% test (stratified), fixed seeds, cross‑validated model selection.  
- **Thresholding**: thresholds swept to report both **Recall** and **Precision**; pick operating point by task (e.g., Youden‑J or Fβ for recall priority).  
- **Pipelines**: All preprocessing happens **inside** `sklearn` pipelines to prevent leakage.  
- **Calibration**: optional probability calibration (e.g., isotonic) for clinically meaningful scores.

---

## 🧠 Models
- **Statistical / Classical ML**: Logistic Regression (Logit), Additive Logit + L1 (Lasso), **LDA**, **CART**, **Bagging**, **Random Forest**, **Boosting** (GLM Boosting, XGBoost), **SVM** (Linear & RBF).  
- **Neural Networks**: MLP / CNN for MRI‑based learning.  
- **Explainability**: **Grad‑CAM** to localize salient regions for CNN predictions.

---

## 📈 Results (Test Set)
The following results summarize the comparative performance across models:

| Model | Precision | Recall | ROC‑AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.6344 | 0.7170 | 0.7146 |
| Additive Logit + Lasso | 0.6290 | 0.7650 | 0.7989 |
| LDA | 0.4122 | 0.9517 | 0.5031 |
| CART | 0.6485 | 0.7316 | 0.7514 |
| Random Forest | 0.6628 | 0.6891 | 0.7820 |
| Bagging | 0.6589 | 0.6719 | 0.7318 |
| **XGBoost** | **0.7657** | 0.6291 | 0.7982 |
| **GLM Boosting (Best AUC)** | 0.7870 | 0.6264 | **0.8039** |
| SVM (Linear) | 0.6386 | 0.7239 | 0.7894 |
| SVM (RBF) | 0.6386 | 0.7238 | 0.7894 |

**Takeaways**
- **Boosting** methods (GLM, XGBoost) reach the **highest ROC‑AUC (~0.80)**.  
- **Additive Logit + L1** offers a strong balance of **Recall** and interpretability.  
- In clinical settings, **higher Recall** can be preferred to minimize False Negatives.

> Note: Exact numbers may vary with seeds/splits. Please re‑run with your environment and document your split/seed for full reproducibility.

---

## 🧩 Repository Structure (suggested)
```
Alzheimer-MRI-Prediction/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  ├─ 02_ml_models.ipynb          # Logit/LDA/CART/RF/Boosting/SVM
│  ├─ 03_cnn_training.ipynb
│  ├─ 04_gradcam.ipynb
├─ src/
│  ├─ data.py          # loaders, patient-wise split
│  ├─ preprocess.py    # scaling/encoding inside sklearn pipelines
│  ├─ models.py        # model builders
│  ├─ train.py         # training loops & CV
│  ├─ eval.py          # metrics, threshold sweep, calibration
│  ├─ viz.py           # plots, Grad-CAM helpers
├─ reports/figures/    # ROC/PR curves, confusion matrices, CAMs
├─ models/             # saved weights (gitignored)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## ⚙️ Setup & Usage
1) **Clone**
```bash
git clone https://github.com/ziaee-mohammad/Alzheimer-MRI-Prediction.git
cd Alzheimer-MRI-Prediction
```

2) **Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3) **Run notebooks / scripts**
- Open Jupyter and run notebooks in order (`notebooks/`).  
- Or use the scripts in `src/` to train/evaluate:
```bash
python -m src.train --model xgboost
python -m src.eval  --checkpoint path/to/model --threshold auto
```

4) **Grad‑CAM**
```bash
python -m src.viz --checkpoint path/to/cnn --input path/to/mri.png --out reports/figures/cam.png
```

---

## 📦 Requirements (example)
```
numpy
pandas
scikit-learn
matplotlib
seaborn
xgboost
torch            # or tensorflow (choose one DL backend)
torchvision
opencv-python
shap             # optional (explainability)
```

> Tip: Pin exact versions for full reproducibility.

---

## ✅ Reproducibility Checklist
- Set `random_state` everywhere (NumPy, sklearn, DL backend).  
- Perform **patient‑level** splits; never mix a subject’s scans across train/test.  
- Fit scalers/encoders **only** on train; apply to val/test.  
- Report **ROC‑AUC** and **PR‑AUC**; include confusion matrices at multiple thresholds.  
- Consider **probability calibration** (`CalibratedClassifierCV`) for clinically meaningful scores.

---

## 🧑‍⚕️ Ethical Considerations
- This repository is for **research and educational** purposes; not a medical device.  
- Models should not be used for clinical decision‑making without proper validation and oversight.

---

## 🙌 Acknowledgments
- Open data contributors and the research community for public datasets and benchmarks.  
- Methodological inspiration from public articles and tutorials. Please see citation notes in the repository if you adapt parts of this work.

---

## 👤 Author
**Mohammad Ziaee** – Computer Engineer | AI & Data Science  
GitHub: https://github.com/ziaee-mohammad

