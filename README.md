# ❤️ Heart Risk Prediction System

## 📌 Project Overview
The **Heart Risk Prediction System** is a Machine Learning web application built using **Flask** and **Random Forest Classifier**.  
This project predicts the risk level of heart disease based on medical input parameters.

The problem comes under **Classification** because:
- The target variable contains **3 classes**
  - Low Risk
  - Medium Risk
  - High Risk

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Random Forest Algorithm
- HTML (Frontend)

---

## 📂 Dataset Information

The dataset is loaded using Pandas:

```python
df = pd.read_csv("heart_final.csv")
```

### Selected Features (Input Variables)

- age
- sex
- chest pain type
- resting bp s
- cholesterol
- max heart rate
- oldpeak

### Target Variable

- target (Heart Risk Level: Low / Medium / High)

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Feature Selection

```python
X = df[selected_features]
y = df["target"]
```

### 2️⃣ Train-Test Split

- 70% Training Data
- 30% Testing Data

```python
train_test_split(test_size=0.3, random_state=42)
```

### 3️⃣ Model Used

**Random Forest Classifier**

```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

### 4️⃣ Model Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report

```python
accuracy_score()
confusion_matrix()
classification_report()
```

---

## 🌐 Flask Application Workflow

### Home Route

```python
@app.route('/')
```

- Loads `index.html`
- Displays input form for user data

### Predict Route

```python
@app.route("/predict", methods=["POST"])
```

Steps:
1. Receives user input from form
2. Converts input values to float
3. Creates input array
4. Predicts heart risk using trained model
5. Calculates probability for each class
6. Returns prediction result to frontend

---

## 🔄 Prediction Process

```python
prediction = rf_clf.predict(input_data)
prediction_proba = rf_clf.predict_proba(input_data)
```

- `predicted_class` → Final Risk Level
- `predicted_probabilities` → Probability of each class

---

## 📊 Output

The application displays:

- Predicted Heart Risk Level
- Probability of each risk category

---

## 🚀 How to Run the Project

1. Install required libraries:

```bash
pip install flask pandas scikit-learn
```

2. Run the application:

```bash
python app.py
```

3. Open browser:

```
http://127.0.0.1:5000/
```

---

## 📈 Model Performance

- The model is trained using Random Forest.
- Accuracy is printed in the console.
- Confusion matrix and classification report help evaluate performance.

---

## 🎯 Project Objective

The main objective of this project is to:

- Predict heart disease risk using Machine Learning
- Provide a simple web interface for user interaction
- Help in early detection of heart-related issues

---

## 📌 Conclusion

This project demonstrates:

- End-to-end Machine Learning pipeline
- Integration of ML model with Flask
- Real-time prediction through web application

It is suitable for:
- Academic Projects
- Portfolio Projects
- Beginner ML Deployment Practice

---

