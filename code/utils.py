from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.pipeline import Pipeline


dataframe =  pd.read_csv('../data/cleaned_dataset_2.csv')
dataframe = dataframe.drop(['age', 'id'], axis=1)

explainatory, response = dataframe.drop('cardio', axis=1), dataframe['cardio']

def classification_evaluation(x_test, y_test, model):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()

def user_input(ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age, bmi, gender):
    data = {
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age,
        "bmi": bmi,
        "gender_women": gender,
        
    }
    df = pd.DataFrame(data)
    return df

def pipe_model(scaler, model):
    pipe = Pipeline([scaler, model])
    pipe.fit(explainatory, response)
    return pipe