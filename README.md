# ⚡ AI-Powered Data Science Workflow Automator

An intelligent **AutoML dashboard** built with Streamlit that automates the entire machine learning pipeline — from data preprocessing to model selection and evaluation.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://datascienceautometer.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## ✨ Features

- 🔍 **Automatic Task Detection** — Intelligently identifies classification vs regression tasks
- 🤖 **Multiple Model Training** — Trains and compares Random Forest, Gradient Boosting, and Linear models
- 📊 **Smart Preprocessing** — Handles missing values, encodes categorical variables automatically
- 📈 **Interactive Visualizations** — Beautiful charts for model comparison and feature importance
- 📥 **Export Reports** — Download comprehensive AutoML reports

---

## 📌 Problem Statement

In typical data science workflows, a significant amount of time is spent on repetitive tasks:
- Loading and validating datasets
- Handling missing values and encoding categorical features
- Manually selecting, training, and comparing ML models
- Writing boilerplate code for evaluation metrics

**Data Science Autometer** eliminates this overhead by automating the entire pipeline — from raw data upload to best-model selection — in a single click.

---

## 🚀 Key Features

### 🔄 Automated End-to-End Pipeline
Upload any CSV/Excel dataset and the system automatically:
1. **Ingests** the data and generates a statistical profile
2. **Preprocesses** it (handles missing values, encodes categoricals, scales features)
3. **Detects** whether the task is Classification or Regression
4. **Trains** multiple ML algorithms simultaneously
5. **Evaluates** all models with comprehensive metrics
6. **Selects** the best-performing model automatically

### 🤖 Multi-Algorithm Auto-Selection
The pipeline trains and compares **6 classification** and **5 regression** algorithms in parallel, then automatically outputs the best performer — no manual tuning required.

### 🧩 Modular, Reusable Architecture
Each component (ingestion, preprocessing, training, evaluation) is a standalone Python module. This eliminates repetitive boilerplate and allows rapid experimentation across different datasets **without any code changes**.

### 📊 Interactive Dashboard
A premium Streamlit interface with:
- Dark-themed UI with custom CSS styling
- Interactive Plotly charts (bar charts, confusion matrices, heatmaps)
- Real-time pipeline progress tracking
- One-click model and report export

---

## 🏗️ Project Architecture

```
DataScienceautometer/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Git ignore rules
├── .streamlit/
│   └── config.toml               # Streamlit dark theme configuration
│
├── src/                          # Core pipeline modules
│   ├── __init__.py               # Package initializer
│   ├── data_ingestion.py         # Data loading & profiling
│   ├── preprocessing.py          # Cleaning, encoding, scaling, splitting
│   ├── model_selection.py        # Algorithm registry & task auto-detection
│   ├── training.py               # Model training with time tracking
│   ├── evaluation.py             # Metrics computation & model comparison
│   └── pipeline.py               # AutoMLPipeline orchestrator class
│
└── sample_data/                  # Built-in demo datasets
    ├── iris.csv                  # Iris flower classification (150 rows)
    ├── titanic.csv               # Titanic survival classification (891 rows)
    └── housing.csv               # California housing regression (500 rows)
```

---

## 🔧 How It Works — Module Breakdown

### 1. Data Ingestion (`src/data_ingestion.py`)
- **`load_data(file)`** — Accepts CSV or Excel uploads and returns a clean Pandas DataFrame
- **`profile_data(df)`** — Generates a complete data profile including: shape, column types, missing value counts/percentages, numeric vs categorical column detection

### 2. Preprocessing (`src/preprocessing.py`)
- **`handle_missing(df, strategy)`** — Fills or drops missing values using configurable strategies:
  - `mean` — fills numeric columns with mean, categorical with mode
  - `median` — fills numeric columns with median
  - `mode` — fills all columns with most frequent value
  - `drop` — removes rows with any missing values
- **`encode_categoricals(df, target_col)`** — Applies Label Encoding to convert categorical text features into numeric values
- **`scale_features(X, method)`** — Normalizes feature values using:
  - `StandardScaler` — zero mean, unit variance
  - `MinMaxScaler` — scales to [0, 1] range
  - `none` — no scaling applied
- **`split_data(X, y, test_size)`** — Splits data into training and testing sets

### 3. Model Selection (`src/model_selection.py`)
- **`detect_task_type(y)`** — Automatically detects whether the problem is classification or regression based on:
  - Target column data type (object/category → classification)
  - Number of unique values (≤20 unique or <5% unique ratio → classification)
  - Otherwise → regression
- **`get_classifiers()`** — Returns 6 classification algorithms
- **`get_regressors()`** — Returns 5 regression algorithms

### 4. Training (`src/training.py`)
- **`train_models(models, X_train, y_train)`** — Trains all models sequentially, records training time for each, and returns fitted model objects

### 5. Evaluation (`src/evaluation.py`)
- **`evaluate_classification()`** — Computes Accuracy, Precision, Recall, F1 Score for each model
- **`evaluate_regression()`** — Computes MAE, RMSE, R² Score for each model
- **`get_best_model()`** — Returns the top-performing model (by Accuracy for classification, R² for regression)
- **`get_confusion_matrix()`** — Generates confusion matrix for the best classification model

### 6. Pipeline Orchestrator (`src/pipeline.py`)
The `AutoMLPipeline` class ties everything together:
```python
pipeline = AutoMLPipeline()
pipeline.ingest(file)                    # Load & profile
pipeline.preprocess(target, strategy)    # Clean & split
pipeline.train()                         # Train all models
pipeline.evaluate()                      # Compare & select best
```

---

## 🧠 Supported Algorithms

### Classification Models
| Algorithm | Description |
|-----------|-------------|
| **Logistic Regression** | Linear model for binary/multiclass classification |
| **Decision Tree** | Tree-based model that splits data on feature thresholds |
| **Random Forest** | Ensemble of 100 decision trees with bagging |
| **SVM (SVC)** | Support Vector Machine with kernel-based separation |
| **K-Nearest Neighbors** | Instance-based learning using distance metrics |
| **Gradient Boosting** | Sequential ensemble that corrects previous errors |

### Regression Models
| Algorithm | Description |
|-----------|-------------|
| **Linear Regression** | Fits a linear relationship between features and target |
| **Decision Tree** | Tree-based model for continuous value prediction |
| **Random Forest** | Ensemble of 100 regression trees |
| **SVR** | Support Vector Regression with kernel mapping |
| **Gradient Boosting** | Sequential boosting for regression tasks |

---

## 📊 Evaluation Metrics

### Classification
| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Percentage of correct predictions overall |
| **Precision** | Of predicted positives, how many were actually positive |
| **Recall** | Of actual positives, how many were correctly predicted |
| **F1 Score** | Harmonic mean of Precision and Recall |

### Regression
| Metric | What It Measures |
|--------|-----------------|
| **MAE** | Mean Absolute Error — average magnitude of errors |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R² Score** | Proportion of variance explained (1.0 = perfect) |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/DataScienceautometer.git
cd DataScienceautometer

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.30.0 | Web application framework |
| `pandas` | ≥2.0.0 | Data manipulation and analysis |
| `scikit-learn` | ≥1.3.0 | ML algorithms and metrics |
| `plotly` | ≥5.18.0 | Interactive charts and visualizations |
| `openpyxl` | ≥3.1.0 | Excel file reading support |

---

## ▶️ Usage

### Run Locally
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Quick Demo
1. Click **"Sample"** tab in the sidebar
2. Select **"Iris (Classification)"** or **"Housing (Regression)"**
3. Click **"Load Sample"**
4. Choose your target column and preprocessing options
5. Click **"🚀 Run Pipeline"**
6. View results in the **"🏆 Results"** tab
7. Download the best model from the **"📥 Export"** tab

### Use Your Own Data
1. Click **"Upload"** tab in the sidebar
2. Drag and drop your CSV or Excel file
3. Configure target column, missing value strategy, scaling method, and test split
4. Click **"🚀 Run Pipeline"**

---

## 🌐 Deployment (Streamlit Cloud)

1. Push the project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — your app will be live in ~2 minutes!

---

## 🛠️ Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.9+** | Core programming language |
| **Pandas** | Data loading, cleaning, and manipulation |
| **Scikit-learn** | Machine learning algorithms, preprocessing, and evaluation |
| **Streamlit** | Interactive web application framework |
| **Plotly** | Interactive data visualizations and charts |

---

## 📈 Sample Datasets Included

| Dataset | Rows | Columns | Task | Description |
|---------|------|---------|------|-------------|
| **Iris** | 150 | 5 | Classification | Classify iris flowers into 3 species based on petal/sepal measurements |
| **Titanic** | 891 | 8 | Classification | Predict passenger survival based on age, class, fare, etc. |
| **Housing** | 500 | 9 | Regression | Predict California median house values from census features |

---

## 📄 License

MIT License — free to use, modify, and distribute.
