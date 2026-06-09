<div align="center">

# ⚡ Data Science Autometer

### Automated Machine Learning Pipeline — From Raw Data to Best Model in One Click

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datascienceautometer.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> **Upload any dataset → choose Classification or Regression → get the best trained model, rich visualisations, and a downloadable `.pkl` — no code required.**

<br/>

![screenshot](https://raw.githubusercontent.com/YOUR_USERNAME/DataScienceautometer/main/docs/screenshot.png)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Supported Algorithms](#-supported-algorithms)
- [Evaluation Metrics](#-evaluation-metrics)
- [Sample Datasets](#-sample-datasets)
- [Installation](#-installation)
- [Usage](#-usage)
- [Deployment](#-deployment-streamlit-cloud)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧭 Overview

**Data Science Autometer** is a no-code, interactive AutoML dashboard that compresses the typical ML workflow — data cleaning, feature engineering, model training, hyperparameter comparison, and export — into a single, guided Streamlit interface.

Instead of writing the same boilerplate for every new dataset, you:

1. **Upload** a CSV / Excel file (or pick a built-in sample)
2. **Select** a target column and task type (**Classification** or **Regression**)
3. **Click** *Run Pipeline*
4. **Explore** an interactive comparison of multiple trained models
5. **Download** the best model as a `.pkl` file — ready to deploy

The modular `src/` backend is clean Python — each concern (ingestion, preprocessing, training, evaluation) lives in its own file, making it trivially easy to extend with new algorithms or metrics.

---

## 🌐 Live Demo

| Resource | Link |
|---|---|
| 🚀 **Streamlit Cloud App** | [datascienceautometer.streamlit.app](https://datascienceautometer.streamlit.app) |
| 📦 **GitHub Repository** | [github.com/YOUR_USERNAME/DataScienceautometer](https://github.com/YOUR_USERNAME/DataScienceautometer) |

> Try it instantly — no installation required. Use the **Iris**, **Titanic**, or **Housing** sample datasets from the sidebar.

---

## ✨ Features

### 🧠 Task Type Selection
Choose how the pipeline interprets your target column:

| Mode | Icon | Behaviour |
|---|---|---|
| **Auto-detect** | 🤖 | Infers task from data type and cardinality of the target column |
| **Classification** | 🟢 | Forces classification — trains 6 classifiers, shows Accuracy / F1 / Confusion Matrix |
| **Regression** | 🟠 | Forces regression — trains 5 regressors, shows R² / MAE / Residuals / Actual vs Predicted |

### 📊 Interactive Dashboard
- Dark-themed premium UI with custom CSS and Google Fonts
- Real-time pipeline progress with `st.status`
- Plotly charts — bar charts, confusion matrices, scatter plots, histograms
- Responsive multi-column layout

### 🔄 Full Preprocessing Control
| Option | Choices |
|---|---|
| **Missing Values** | Mean, Median, Mode, Drop |
| **Feature Scaling** | Standard (Z-score), Min-Max, None |
| **Test Split** | 10 % – 50 % (slider) |

### 🏆 Smart Model Comparison
- Side-by-side metric table with colour-highlighted best scores
- For regression: **R² highlighted green** (higher is better), **MAE / RMSE highlighted green** (lower is better)
- Training time comparison bar chart for every run

### 🔮 Live Prediction Tab
- Input feature values via an auto-generated form
- Instant prediction from the best model — no extra code
- Shows **class probabilities** as a bar chart for classifiers that support `predict_proba`

### 📥 One-Click Export
- Download the full model comparison as **CSV**
- Download the best trained model as a **`.pkl`** file ready to `pickle.load()`

---

## 🏗️ Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA SCIENCE AUTOMETER                        │
│                          Streamlit Frontend                          │
└──────────────────────┬───────────────────────────────────────────────┘
                        │
         ┌──────────────▼──────────────┐
         │      1. Data Ingestion       │  load_data()  ·  profile_data()
         │   CSV / Excel → DataFrame    │
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │       2. Preprocessing       │  handle_missing()
         │  Clean · Encode · Scale ·    │  encode_categoricals()
         │        Train/Test Split      │  scale_features()  ·  split_data()
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │      3. Task Detection       │  detect_task_type()
         │  Auto / Classification /     │  — or manual override —
         │         Regression           │
         └──────────────┬──────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
 ┌────────▼────────┐       ┌─────────▼────────┐
 │  Classification  │       │    Regression     │
 │  6 algorithms    │       │   5 algorithms    │
 └────────┬────────┘       └─────────┬────────┘
          └─────────────┬─────────────┘
                        │
         ┌──────────────▼──────────────┐
         │       4. Training            │  train_models()
         │   All models fit in parallel │  — with timing —
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │       5. Evaluation          │  evaluate_classification()
         │  Metrics · Ranking ·         │  evaluate_regression()
         │       Best Model             │  get_best_model()
         └──────────────┬──────────────┘
                        │
         ┌──────────────▼──────────────┐
         │       6. Results UI          │  Comparison Table · Charts
         │  Predict Tab · Export Tab    │  Live Prediction · .pkl Export
         └─────────────────────────────┘
```

---

## 📂 Project Structure

```
DataScienceautometer/
│
├── app.py                      # 🖥️  Main Streamlit application
├── requirements.txt            # 📦  Python dependencies
├── README.md                   # 📖  Project documentation
├── .gitignore
│
├── .streamlit/
│   └── config.toml             # 🎨  Dark theme configuration
│
├── src/                        # ⚙️  Core ML pipeline modules
│   ├── __init__.py
│   ├── data_ingestion.py       #     load_data · profile_data
│   ├── preprocessing.py        #     handle_missing · encode · scale · split
│   ├── model_selection.py      #     get_classifiers · get_regressors · detect_task_type
│   ├── training.py             #     train_models (with timing)
│   ├── evaluation.py           #     evaluate · get_best_model · confusion_matrix
│   └── pipeline.py             #     AutoMLPipeline orchestrator class
│
└── sample_data/                # 🗂️  Built-in demo datasets
    ├── iris.csv                #     150 rows · Classification
    ├── titanic.csv             #     891 rows · Classification
    └── housing.csv             #     500 rows · Regression
```

---

## 🤖 Supported Algorithms

### 🟢 Classification (6 models)

| Model | Key Strength |
|---|---|
| **Logistic Regression** | Fast, interpretable baseline for linear problems |
| **Decision Tree** | Highly interpretable; handles non-linear boundaries |
| **Random Forest** | Ensemble of 100 trees — robust, low variance |
| **SVM (SVC)** | Excellent on high-dimensional data with kernel trick |
| **K-Nearest Neighbors** | Non-parametric; effective on small datasets |
| **Gradient Boosting** | Sequential ensemble — typically top accuracy |

### 🟠 Regression (5 models)

| Model | Key Strength |
|---|---|
| **Linear Regression** | Interpretable baseline; fast on large data |
| **Decision Tree Regressor** | Captures non-linear relationships |
| **Random Forest Regressor** | Ensemble of 100 trees — resistant to outliers |
| **SVR** | Kernel-based; strong on non-linear, small datasets |
| **Gradient Boosting Regressor** | Generally highest R² on tabular data |

---

## 📊 Evaluation Metrics

### 🟢 Classification

| Metric | Formula | What It Means |
|---|---|---|
| **Accuracy** | Correct / Total | Overall correctness percentage |
| **Precision** | TP / (TP + FP) | Of positive predictions, how many were right |
| **Recall** | TP / (TP + FN) | Of actual positives, how many were found |
| **F1 Score** | 2 · (P · R) / (P + R) | Harmonic balance of Precision and Recall |

> Multi-class problems use **weighted averaging** across classes.  
> Binary problems use **binary** averaging.

### 🟠 Regression

| Metric | Formula | What It Means |
|---|---|---|
| **MAE** | mean(|y − ŷ|) | Average absolute error — same unit as target |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more than MAE |
| **R² Score** | 1 − SS_res/SS_tot | Proportion of variance explained (1.0 = perfect) |

---

## 🗂️ Sample Datasets

| Dataset | Rows | Features | Task | Target | Description |
|---|---|---|---|---|---|
| **Iris** | 150 | 4 | Classification | Species | Classify iris flowers into 3 species by petal/sepal dimensions |
| **Titanic** | 891 | 7 | Classification | Survived | Predict passenger survival from age, class, fare, etc. |
| **Housing** | 500 | 8 | Regression | Median House Value | Predict California housing prices from census features |

---

## 📦 Installation

### Prerequisites

- Python **3.9 or higher**
- `pip` package manager

### Clone & Install

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/DataScienceautometer.git
cd DataScienceautometer

# 2. (Recommended) Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.30.0 | Interactive web dashboard |
| `pandas` | ≥ 2.0.0 | Data manipulation & analysis |
| `scikit-learn` | ≥ 1.3.0 | ML algorithms, metrics, preprocessing |
| `plotly` | ≥ 5.18.0 | Interactive visualisations |
| `openpyxl` | ≥ 3.1.0 | Excel file support |

---

## ▶️ Usage

### Run Locally

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Step-by-Step Walkthrough

#### Option A — Sample Dataset (fastest)
1. Open the **sidebar → Sample** tab
2. Select **Iris (Classification)**, **Titanic (Classification)**, or **Housing (Regression)**
3. Click **Load Sample**

#### Option B — Your Own Dataset
1. Open the **sidebar → Upload** tab
2. Drag & drop any `.csv` or `.xlsx` file

#### Running the Pipeline
1. Select your **Target Column** from the dropdown
2. Choose **Task Type**:
   - `🤖 Auto-detect` — let the app infer from data
   - `🟢 Classification` — force classification models
   - `🟠 Regression` — force regression models
3. Configure **Missing Values**, **Scaling**, and **Test Split**
4. Click **🚀 Run Pipeline**

#### Exploring Results
| Tab | What You'll Find |
|---|---|
| **📋 Data Preview** | Dataset table + shape / missing / numeric metrics |
| **🏆 Results** | Best model card · comparison table · all charts |
| **🔮 Predict** | Enter feature values → instant prediction + probabilities |
| **📥 Export** | Download comparison CSV and best model `.pkl` |

### Load the Exported Model

```python
import pickle

# Load model
model = pickle.load(open("best_model_random_forest.pkl", "rb"))

# Predict
import pandas as pd
X_new = pd.DataFrame([{"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2}])
prediction = model.predict(X_new)
print(prediction)  # e.g. ['setosa']
```

---

## ☁️ Deployment — Streamlit Cloud

Deploy in under 5 minutes — **completely free**:

1. **Fork or push** this repo to your GitHub account
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
3. Click **New app** → select your repo and branch
4. Set **Main file path** to `app.py`
5. Click **Deploy** — your app will be live at `https://YOUR_APP.streamlit.app` in ~2 minutes

> The `requirements.txt` is auto-detected, so no extra config is needed.

---

## 🛠️ Tech Stack

| Technology | Version | Role |
|---|---|---|
| **Python** | 3.9+ | Core language |
| **Streamlit** | 1.30+ | Web application framework |
| **Pandas** | 2.0+ | Data loading, cleaning, transformation |
| **Scikit-learn** | 1.3+ | ML algorithms, preprocessing, metrics |
| **Plotly** | 5.18+ | Interactive charts and visualisations |
| **NumPy** | 1.24+ | Numerical computations |
| **OpenPyXL** | 3.1+ | Excel file reading |

---

## 🗺️ Roadmap

| Status | Feature |
|---|---|
| ✅ | Auto task-type detection |
| ✅ | Manual Classification / Regression selection |
| ✅ | 6 classifiers + 5 regressors |
| ✅ | Confusion matrix, residuals, Actual vs Predicted |
| ✅ | Live Predict tab with class probabilities |
| ✅ | Model & comparison export |
| 🔲 | Cross-validation (k-fold) support |
| 🔲 | Hyperparameter tuning (GridSearch / Optuna) |
| 🔲 | Feature importance charts (SHAP) |
| 🔲 | Multi-label classification support |
| 🔲 | Time-series dataset detection & handling |
| 🔲 | PDF report generation |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/DataScienceautometer.git
cd DataScienceautometer

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and commit
git add .
git commit -m "feat: add your feature description"

# 5. Push and open a Pull Request
git push origin feature/your-feature-name
```

### Contribution Guidelines
- Follow existing code style (each concern in its own `src/` module)
- Add docstrings to all new functions
- Test with at least one sample dataset before submitting a PR
- Keep PRs focused — one feature / fix per PR

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

```
MIT License

Copyright (c) 2024 Aman Khokhar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

Made with ❤️ by **Aman Khokhar**

⭐ **Star this repo** if it saved you time — it helps others find it!

[![GitHub Stars](https://img.shields.io/github/stars/YOUR_USERNAME/DataScienceautometer?style=social)](https://github.com/YOUR_USERNAME/DataScienceautometer)

</div>
