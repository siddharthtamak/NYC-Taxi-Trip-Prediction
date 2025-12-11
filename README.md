# 🚕 New York Taxi Trip Prediction — Machine Learning Project

**IIT Delhi · AIL7024 — Machine Learning**

| Role | Name | Entry Number |
| :--- | :--- | :--- |
| **Author** | Yuvraj Verma | 2025AIB2568 |
| **Author** | Siddharth | 2025AIB2670 |
| **Instructor** | Prof. Tanmoy Chakraborty | |

---

## ⭐ Project Overview
This project focuses on two important tasks in intelligent transportation systems using the **NYC Taxi Trip Duration Dataset** from Kaggle (1.45M rides):

### 1️⃣ Trip Duration Prediction (Regression Task)
Predicting the travel time of NYC taxi rides using machine learning models based on geospatial and temporal features.

### 2️⃣ Ride Acceptance Prediction (Classification Task)
Predicting whether a ride request will be accepted or rejected based on engineered rules that simulate real-world driver decision-making behavior (e.g., profitability, traffic conditions).

---

## 📂 Project Structure

```text
ML-Taxi-Prediction/
│
├── data/
│   ├── taxi_data.csv
│
├── notebooks/
│   ├── 01_preprocessing_eda.ipynb
│   ├── 02_regression.ipynb
│   ├── 03_classification.ipynb
│
├── src/
│   ├── utils.py
│   ├── regression.py
│   └── classification.py
│
├── report/
│   └── Project_Report.pdf
│
├── README.md
└── requirements.txt
```

## 🧹 Preprocessing & Exploratory Data Analysis (EDA)
Extensive feature engineering was performed to prepare the data for modeling:

* **Geospatial Features:**
    * Calculated **Haversine** and **Manhattan** distances.
    * Computed distance to key landmarks (Airports, NYC Center).
    * Removed invalid coordinates and handled missing values.
* **Temporal Features:**
    * Extracted hour, weekday, month.
    * Created boolean flags: `Weekend Indicator`, `Rush-hour Indicator`.
    * Applied **Cyclical Encoding** (Sine/Cosine) to time features.
* **Target Transformation:** Applied log transformation (`log(1 + trip_duration)`) to stabilize the highly right-skewed duration target.

---

## 📈 Regression Modelling (Trip Duration)
We implemented baseline models from scratch and compared them with state-of-the-art ensemble methods.

* **Implemented from Scratch:**
    * Linear Regression
    * Polynomial Regression (Degree 2)
    * Ridge Regression
    * Lasso Regression
* **Advanced Models:**
    * Support Vector Regression (SVR)
    * XGBoost
    * **LightGBM (Best Performance)**
* **Evaluation Metrics:** RMSE, Log RMSE, R² Score.

---

## 🧠 Classification Modelling (Ride Acceptance)

### Target Creation Rules
Since the dataset lacks explicit acceptance labels, we created a custom binary label (`accepted=1`, `rejected=0`) simulating driver behavior:
1. **Reject:** Short trips (< 1 km).
2. **Reject:** Long trips (> 50 km).
3. **Reject:** Short trips (< 3 km) during Rush Hour.
4. **Reject:** Trips during Sleep Hours (00:00–05:00).
* *Distribution:* ~40% Rejected / 60% Accepted.

### Models Implemented
* **From Scratch:** Logistic Regression (Standard, L1 Regularization, L2 Regularization).
* **Advanced:** Decision Tree, Random Forest, HistGradientBoostingClassifier, SGDClassifier.
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.

---

## 📊 Results Summary

| Task | Best Model | Key Metric |
| :--- | :--- | :--- |
| **Regression** | LightGBM | Lowest Log-RMSE |
| **Classification** | Random Forest / HGBClassifier | F1 Score ≈ 1.00 |

*Both tasks demonstrated strong predictive performance due to rich feature engineering and the large dataset size.*

---

## ▶️ How to Run the Code

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Jupyter Notebooks**
   Launch the notebooks to view the step-by-step analysis
   ```bash
   jupyter notebook
   ```
   *Order of execution:* `01_preprocessing_eda.ipynb` → `02_regression.ipynb` → `03_classification.ipynb`

3. **Source Code**
   Modularized logic for training and evaluation can be found in the `src/` folder.

---

## 📝 Report
The complete academic project report, detailing the methodology, mathematical formulations, and in-depth results, is available here:
👉 [**report/Project_Report.pdf**](Project_Report.pdf)

---

## 🙌 Contributions
* **Yuvraj Verma:** EDA, Feature Engineering, Regression Models, Report Writing.
* **Siddharth:** Classification Models, Model Evaluation, Presentation, Documentation.
* *Both authors contributed equally to the overall analysis and implementation.*

---

## ⚠️ Academic Integrity Policy
> **This repository is intended for educational and portfolio purposes only.**
>
> Please do not copy, clone, or submit this code as your own work for any academic assignment or competition. If you are a student taking a similar course (e.g., AIL7024), please use this repository only as a reference for understanding the concepts. Plagiarism is a serious academic offense.
---

## 📬 Contact
For queries, feel free to reach out through GitHub Issues.


   
