# ML-Use-Case-1.-Rock_vs_Mine_Prediction.ipynb
# 🪨 ML Use Case 1: Rock vs Mine Prediction

This project applies machine learning to classify sonar signals as reflections from either **rocks** or **underwater mines**. It demonstrates the use of supervised learning models on frequency-based data for defense or sonar-based underwater navigation systems.

---

## 📂 Dataset

**Source**: [UCI Machine Learning Repository - Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))

- **Features**: 60 numerical attributes representing sonar signal energy in various frequency bands  
- **Target**:  
  - `R` = Rock  
  - `M` = Mine

---

## 🎯 Objective

To build a binary classification model that accurately predicts whether an object detected by sonar is a **rock** or a **mine** based on its signal profile.

---

## 📊 Workflow Overview

1. **Data Loading & Exploration**
   - Checked data structure and statistics  
   - Analyzed class distribution  

2. **Preprocessing**
   - Encoded labels (`R` → 0, `M` → 1)  
   - Scaled features using `StandardScaler`  
   - Split data into training and test sets  

3. **Model Training**
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - Random Forest Classifier  
   - (Optional) k-NN or XGBoost  

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix  
   - ROC Curve (optional)  

---

## 🧠 Sample Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 83%      | 0.81      | 0.85   | 0.83     |
| Random Forest       | 85%      | 0.84      | 0.86   | 0.85     |
| SVM                 | 86%      | 0.85      | 0.87   | 0.86     |

---

## 🧰 Requirements

```bash
Python 3.x
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter

