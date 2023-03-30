from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request
import json
from flask_cors import CORS
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from datetime import date

app = Flask(__name__)
CORS(app)

dataframe = pd.read_csv("./data/cleaned_dataset_2.csv")
X, y = dataframe.drop("cardio", axis=1), dataframe["cardio"]

scaler = StandardScaler()
model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=2)
model.fit(X, y)


def user(
    ap_hi,
    ap_lo,
    cholesterol,
    gluc,
    smoke,
    alco,
    active,
    age,
    age_years,
    bmi,
    gender,
    id,
):
    data = {
        "id": id,
        "age": age,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age_years,
        "bmi": bmi,
        "gender_women": gender,
    }
    df = pd.DataFrame(data, index=[0])
    return df


def age_to_days(age):
    today = date.today()
    birth_date = today.replace(year=today.year - age)
    days = (today - birth_date).days
    return int(days)


@app.route("/api/predict")
def predict():
    form = request.args.get("form")
    form_dict = json.loads(form)
    gender = int(form_dict.get("gender"))
    age_years = int(form_dict.get("age"))
    height = int(form_dict.get("height"))
    weight = int(form_dict.get("weight"))
    ap_hi = int(form_dict.get("systolicBP"))
    ap_lo = int(form_dict.get("diastolicBP"))
    alco = int(form_dict.get("alco"))
    active = int(form_dict.get("active"))
    cholesterol = int(form_dict.get("cholesterol"))
    gluc = int(form_dict.get("gluc"))
    smoke = int(form_dict.get("smoke"))

    age = age_to_days(age_years)
    bmi = weight / (height * height)

    df = user(
        id=59000,
        ap_hi=ap_hi,
        ap_lo=ap_lo,
        cholesterol=cholesterol,
        gluc=gluc,
        smoke=smoke,
        alco=alco,
        active=active,
        age=age,
        bmi=bmi,
        gender=gender,
        age_years=age_years,
    )

    prediction = model.predict(df)
    result = prediction[0].item()
    
    if result == 0:
        conclusion = 'The algorithm suggests you MAY NOT have any cardiovascular disease'
        warning = 'If you believe you could be sick, please contact the specialist!'
    elif result == 1:
        conclusion = 'The algorithm suggests you MAY have a cardiovascular disease'
        warning = 'This is a machine learning algoritm. The diagnosis are not done by a licensed prefessional and could well be wrong!'
    return jsonify((conclusion, warning))


if __name__ == "__main__":
    app.run(debug=True, port=5200, host="0.0.0.0")
