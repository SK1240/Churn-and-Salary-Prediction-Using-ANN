# 🧠 ANN — Churn & Salary Prediction (Streamlit Deployments)
A lightweight **Artificial Neural Network (ANN)** repository featuring two **Streamlit applications** for real-time prediction — a dual-purpose **deep learning** project demonstrating how **ANNs** power both Customer Churn Classification and Salary Regression, seamlessly integrated into interactive apps.

## 🚀 Overview
This repository features two **ANN-based Streamlit apps** for real-time prediction:

🎯 `churn_app.py` — Predicts whether a customer will exit a service (Classification).

💰 `salary_app.py` — Estimates an employee’s salary based on given features (Regression).

Each app loads pre-trained **ANN models** and preprocessing artifacts for seamless inference.

## ⚡ Highlights
* Two pre-trained **ANN models** (`.h5`) built using **TensorFlow/Keras**
* Interactive Streamlit UI for real-time predictions
* Preprocessing artifacts stored in organized model folders
* Includes notebooks for training, tuning, and inference
* Ready for deployment on Streamlit Cloud

## 🧩 Project Structure
```
ANN-Churn-and-Salary-Prediction/
│
├── churn_prediction_models/               # Preprocessing artifacts for churn classification
│   ├── le_gender.pkl                      # LabelEncoder for 'Gender' feature
│   ├── ohe_geo.pkl                        # OneHotEncoder for 'Geography' feature
│   └── scaler.pkl                         # StandardScaler for feature scaling
│
├── salary_prediction_models/              # Preprocessing artifacts for salary regression
│   ├── le_gender_salary.pkl               # LabelEncoder for 'Gender' feature (salary model)
│   ├── ohe_geo_salary.pkl                 # OneHotEncoder for 'Geography' feature (salary model)
│   └── scaler_salary.pkl                  # StandardScaler for feature scaling (salary model)
│
├── logs/                                  # Log files for churn model
├── rglogs/                                # Log files for salary model
│
├── venv/                                  # Virtual environment (optional)
│
├── .gitignore
├── Churn_Modelling.csv                    # Dataset for churn model training
│
├── churn_app.py                           # Streamlit app — churn prediction (classification)
├── salary_app.py                          # Streamlit app — salary prediction (regression)
│
├── churn_prediction.h5                    # Trained ANN model (churn)
├── salary_prediction_model.h5             # Trained ANN model (salary)
│
├── churn_prediction.ipynb                 # Churn model training/inference notebook
├── salary_prediction.ipynb                # Salary model training/inference notebook
├── hyperparametertuning_churn.ipynb       # ANN hyperparameter tuning notebook (churn)
│
├── requirements.txt                       # Dependency list
└── README.md                              # (this file)
```

## ⚙️ Tech Stack

| Component                            |            Description |
|-----------|------------|
| 🧠 Keras / TensorFlow                |            Model building and training (ANN) |
| 📊 Scikit-learn                      |            Data preprocessing and encoding |
| 🥣 Pandas / NumPy                    |            Data wrangling |
| 🎨 Streamlit                         |            Web app deployment |
| 📦 Pickle (.pkl)                     |            Saving preprocessing models |
| 📈 Matplotlib / Seaborn              |            Data visualization |

## 🧭 Config Checks Before Running

Before running the apps, ensure the following:

* `churn_prediction.h5` and `salary_prediction_model.h5` are present in the repo root.

* Preprocessing artifacts exist in their respective folders:

     * Churn: `churn_prediction_models`/`ohe_geo.pkl`, `le_gender.pkl`, `scaler.pkl`

     * Salary: `salary_prediction_models`/`ohe_geo_salary.pkl`, `le_gender_salary.pkl`, `scaler_salary.pkl`

`requirements.txt` is installed in your Python environment.

If retraining locally, maintain the same feature order used when saving the scaler for consistent inference.

## 🔧 Setup & Installation

### 1.Clone this repository: 
```
git clone https://github.com/SK1240/Churn-and-Salary-Prediction-Using-ANN.git
cd ANN-Churn-and-Salary-Prediction
```

### 2.Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
venv\Scripts\activate       # For Windows
source venv/bin/activate    # For macOS/Linux
```

### 3.Install the dependencies:
```
pip install -r requirements.txt
```

## 🧠 Run the Apps
### 🎯 Run Churn Prediction App
```
streamlit run churn_app.py
```

### 💰 Run Salary Prediction App
```
streamlit run salary_app.py
```
Once executed, your default browser will open the Streamlit dashboard for real-time predictions.

Deployment: Upload the project repository to GitHub and deploy the application using [Streamlit Cloud](https://share.streamlit.io/).

## 🛠️ How the Apps Work

Churn App (`churn_app.py`)
* Model & Preprocessing: Loads `churn_prediction.h5` and preprocessing artifacts from `churn_prediction_models/` (`le_gender.pkl`, `ohe_geo.pkl`, `scaler.pkl`).

* **Input**: Geography, Gender, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

* **Processing**: Encodes categorical features, scales numeric features, and predicts churn probability.

* Output: Displays probability with thresholded message (`prob > 0.5 → churn likely`).

Salary App (`salary_app.py`)

* Model & Preprocessing: Loads `salary_prediction_model.h5` and artifacts from `salary_prediction_models/` (`le_gender_salary.pkl`, `ohe_geo_salary.pkl`, `scaler_salary.pkl`).

* **Input**: Same features as churn app.

* **Processing**: Encodes & scales features, predicts salary.

* **Output**: Displays the predicted salary.

## 📈 Model Training
Both models were trained on structured tabular data:

   Churn Model: Trained using `Churn_Modelling.csv`

   Salary Model: Trained on preprocessed synthetic employee datasets

Architecture:

Input Layer: 
   Encoded categorical and numerical features

Hidden Layers: 
   Fully connected layers with ReLU activation

Output Layer:

   **Sigmoid** for churn classification

   **Linear** for salary regression

Preprocessing:

Categorical features encoded with LabelEncoder and OneHotEncoder

Numeric features scaled with StandardScaler















