from flask import Flask,request,render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#our project comes under classification
# because we have 3 classes to classify
#classes -low,medium, high

#initialize the flask app
app=Flask(__name__)

#load the dataset
df=pd.read_csv(r"C:\Users\USer\Documents\Heart Risk Prediction\heart_final (1).csv")

#selection the feature
selected_fetures=["age","sex","chest pain type","resting bp s","cholesterol","max heart rate","oldpeak"]
X=df[selected_fetures]

#target variable
y=df["target"]

#spliting
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#initialize and train the RFC
rf_clf=RandomForestClassifier(n_estimators=100,random_state=42)
rf_clf.fit(X_train,y_train)

#predict on the test data and evaluate
y_pred=rf_clf.predict(X_test)

#print the accuracy and classification report
accuracy=accuracy_score(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)

print("Accuracy on test Data:",accuracy)
print("Confusion Matrix",conf_matrix)

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    
    age = float(request.form["age"])
    sex = float(request.form["sex"])
    chest_pain_type = float(request.form["chest_pain_type"])
    resting_bp_s = float(request.form["resting_bp_s"])
    cholesterol = float(request.form["cholesterol"])
    max_heart_rate = float(request.form["max_heart_rate"])
    oldpeak = float(request.form["oldpeak"])

    input_data=[[age,sex,chest_pain_type,resting_bp_s,cholesterol,max_heart_rate,oldpeak]]

    #make a prediction
    prediction = rf_clf.predict(input_data)
    prediction_proba = rf_clf.predict_proba(input_data)

    predicted_class = prediction[0]
    predicted_probabilities = dict(
        zip(rf_clf.classes_, prediction_proba[0])
    )

    return render_template(
        'index.html',
        predicted_class=predicted_class,
        predicted_probabilities=predicted_probabilities
    )


if __name__ == '__main__':
    app.run(debug=True)