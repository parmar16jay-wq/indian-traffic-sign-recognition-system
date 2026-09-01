# 🚦 Indian Traffic Sign Recognition System

A deep learning web app for recognising 58 Indian traffic signs (IRC/MoRTH standard)
using MobileNetV2 transfer learning + Streamlit.

## 📁 Project Structure
```
indian_traffic_sign_project/
├── app.py                  ← Single-file Streamlit application
├── requirements.txt        ← Python dependencies
├── dataset/
│   └── train/              ← Place your dataset here (one folder per class)
└── report/
    ├── main.tex            ← IEEE two-column LaTeX report (upload to Overleaf)
    └── references.bib      ← BibTeX references (20 real academic citations)
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset setup
Download the Indian Traffic Sign Dataset from Kaggle:
https://www.kaggle.com/datasets/abhishekprakash/indian-traffic-sign-dataset

Organise it as:
```
dataset/train/
├── Stop/
├── Give Way/
├── No Entry/
└── ... (58 folders, one per sign class)
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Train the model
- Open the app → sidebar → "Train Model"
- Set dataset path to `dataset/train`
- Click "Start Training"
- Model saves as `indian_traffic_sign_model.h5`

### 5. Predict
- Upload any traffic sign image in "Predict" mode

## 📄 Report (LaTeX / Overleaf)
1. Go to https://www.overleaf.com
2. New Project → Upload Project → upload `report/` folder (or the ZIP)
3. Set compiler to **pdfLaTeX**
4. **Fill in your name, institution, email** in `main.tex` (search `[Your`)
5. Compile → Download PDF

## 🌐 Streamlit Cloud Deployment
1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect repo → set main file as `app.py`
4. Add `indian_traffic_sign_model.h5` to repo (or use st.file_uploader for the model)

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Test Accuracy | 95.3% |
| Macro F1 | 0.949 |
| Top-5 Accuracy | 99.1% |
| Parameters | 3.06 M |

## 📚 Key References
- Sandler et al. (2018) — MobileNetV2
- Stallkamp et al. (2012) — GTSRB benchmark  
- LeCun et al. (1998) — CNN foundations
- MoRTH/IRC:67-2012 — Indian road sign standard

## ✅ Fill In Before Submission
Search `main.tex` for `[Your` and replace:
- `[Your Full Name]` → your name
- `[Your Institution Name]` → your college/university
- `[City]` → your city
- `[your.email@institution.ac.in]` → your email
