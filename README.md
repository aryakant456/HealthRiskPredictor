# 🏥 Medical Prediction System

This project is a machine learning–based medical prediction system designed to assist in the early detection and classification of common health conditions. Using structured medical datasets, the system predicts various metrics such as heart rate abnormalities, glucose levels, and tumor classification (benign or malignant). It integrates preprocessing, training, evaluation, and visualization into a complete pipeline that demonstrates the power of AI in healthcare.

---

## 📌 Features

- ✅ Predictive models for:
  - Heart rate classification
  - Glucose level prediction
  - Breast cancer (tumor) classification
- 🧹 Data cleaning and preprocessing
- 📊 Exploratory Data Analysis (EDA)
- 📈 Model training and evaluation using metrics like accuracy, precision, recall
- 🔍 Visualization using Matplotlib and Seaborn
- 🧠 Algorithms include Logistic Regression, Decision Tree, etc.
- 💾 Uses popular datasets such as the Framingham Heart Study and Breast Cancer Wisconsin dataset

---

## ⚙️ Technical Overview

This project consists of three primary machine learning pipelines implemented in Jupyter Notebooks. Each pipeline is focused on predicting a specific medical outcome using well-known health datasets.

### 📁 1. Heart Rate Prediction

**Goal:** Classify whether a person's heart rate condition is normal or abnormal based on their health metrics.

- Dataset: Based on cardiovascular data (e.g., Framingham Study)
- Features: Age, cholesterol, blood pressure, diabetes, etc.
- Model: Logistic Regression or Decision Tree
- Evaluation: Accuracy, confusion matrix
- Visuals: Heatmaps, bar charts

### 📁 2. Glucose Level Prediction

**Goal:** Predict glucose levels or classify diabetes risk.

- Dataset: Includes BMI, age, insulin, blood pressure, etc.
- Model: Regression or Classification
- Evaluation: RMSE, accuracy, F1-score
- Visuals: Histograms, scatter plots

### 📁 3. Tumor Classification (Breast Cancer Detection)

**Goal:** Classify tumor as malignant or benign.

- Dataset: Breast Cancer Wisconsin Dataset
- Features: Cell size, texture, smoothness, etc.
- Model: Logistic Regression, Decision Tree, Random Forest
- Evaluation: ROC AUC, F1-score
- Visuals: Feature correlation heatmap, ROC curve

### 🔁 Common Components

- Preprocessing: Label encoding, normalization, missing value handling
- EDA: Distribution plots, correlation analysis
- Model Training: scikit-learn pipeline with train/test split
- Evaluation: Confusion matrix, classification report, accuracy

---

## 🧠 Technologies Used

- Python 🐍
- pandas & NumPy
- scikit-learn
- matplotlib & seaborn
- Jupyter Notebook

---

## 📂 Project Structure

```
MedicalPredictionSystem/
├── Heart_Rate_Prediction.ipynb
├── Glucose_Level_Prediction.ipynb
├── Tumor_Classification.ipynb
├── dataset/
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/aryakant456/MedicalPredictionSystem.git
cd MedicalPredictionSystem
pip install -r requirements.txt
```

Run each notebook in JupyterLab or VS Code to explore the models.

---

## 📈 Sample Outputs

- Accuracy scores and confusion matrices
- Learning curves
- Feature importance and EDA charts

---

## 💡 Use Cases

- Early health risk screening
- Medical decision support prototypes
- ML practice on healthcare data

---

## 📜 License

This project is licensed under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome! Fork the repo and open a PR to add new models or improvements.

---

## 📬 Contact

For questions or feedback, contact **[Arya Kant Pathak]** at aryakant9973@gmail.com
