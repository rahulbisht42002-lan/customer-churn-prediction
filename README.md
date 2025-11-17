

# 📊 Customer Churn Prediction — End-to-End Machine Learning Project

This project builds a complete **Customer Churn Prediction System** using the **Telco Customer Churn Dataset**.
It includes an end-to-end machine learning pipeline for:

✔ Data ingestion
✔ Preprocessing & feature engineering
✔ Model training (XGBoost / RandomForest)
✔ Evaluation (AUC, classification report)
✔ Explainability using SHAP
✔ Deployment-ready model export (`pipeline_model.pkl`)

This is a **job-ready ML project**, designed to demonstrate real-world data science skills.

---

## 🚀 Key Features

### 🔹 **1. Full ML Pipeline**

* Handles missing values
* Encodes categorical features
* Applies scaling with StandardScaler
* Trains ML models (XGBoost or Random Forest fallback)
* Saves the model pipeline for deployment

### 🔹 **2. Automatic Dataset Handling**

* Upload your Telco dataset directly in Google Colab
* If no dataset is provided, the script auto-creates a **synthetic dataset** so the project runs smoothly

### 🔹 **3. Model Evaluation**

Includes:

* AUC Score
* Precision, Recall, F1-score
* Confusion Matrix

These metrics help assess churn prediction performance.

### 🔹 **4. Deployment-Ready Model**

The final trained model is saved as:

```
models/pipeline_model.pkl
```

You can directly use it in:

* FastAPI
* Streamlit
* Flask apps
* ML pipelines

### 🔹 **5. SHAP Explainability**

The script computes SHAP values to show feature importance and explain customer churn prediction.

---

## 📂 Project Structure

```
customer-churn-prediction/
│── notebooks/
│   └── customer_churn.ipynb
│
│── data/
│   ├── raw/                 # upload telco_churn.csv here
│   └── processed/
│       └── telco_processed.csv
│
│── models/
│   └── pipeline_model.pkl   # saved trained ML model
│
│── README.md
│── requirements.txt
```

---

## 📊 Dataset

This project uses the **Telco Customer Churn Dataset**.

You can download it here:

🔗 Kaggle:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


```
data/raw/telco_churn.csv
```

---

## ▶ How to Run This Project (Google Colab)

1. Open the `.ipynb` notebook on Google Colab
2. Run all cells
3. Upload your dataset when prompted
4. The model trains automatically
5. Access results, SHAP values, and saved files

---

## 🔍 Example Prediction

The script includes a function to predict churn for new customers:

```python
sample = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70
}

predict_sample(sample)
```

Output:

```
0.82   # 82% chance the customer will churn
```

---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Matplotlib
* Joblib
* Google Colab

---

## 📈 Model Performance Example

(Your output may vary)

* **AUC**: ~0.80–0.85
* **Accuracy**: ~78%
* Strong recall for churn class

---

## 🎯 Objectives of the Project

* Build a practical ML pipeline for real business use
* Explain results using SHAP
* Learn model deployment workflow
* Demonstrate ML engineering skills
* Create a job-ready portfolio project

---

## 👤 Author

**Rahul Bisht**
(*rahulbisht42002@gmail.com*)


