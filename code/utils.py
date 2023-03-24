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
    ConfusionMatrixDisplay(cm, display_labels=['absent', 'present']).plot()

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



def patient_diagnostic_for_cardiovascular_disease(scaler, model):
    pipe = pipe_model(scaler=scaler, model=model)
    prompts = ['Enter value for systolic blood pressure (ap_hi): ',
               'Enter value for diastolic blood presure (ap_lo): ',
               'Between 1 to 3 enter value for cholesterol: ',
               'Between 1 to 3 enter value for glucose: ',
               'Between 0 or 1, enter value for smoker: ',
               'Between 0 or 1, enter value for alcohol consumption: ',
               'Between 0 or 1 enter value for active: ',
               'Enter value for age: ',
               'Enter value for bmi: ',
               'Between 1 or 2, enter value for gender: ']
    
    values = []
    for prompt in prompts:
        value = input(prompt)
        values.append(value)
    
    df = pd.DataFrame({'ap_hi': [values[0]],
                       'ap_lo': [values[1]],
                       'cholesterol': [values[2]],
                       'gluc': [values[3]],
                       'smoke': [values[4]],
                       'alco': [values[5]],
                       'active': [values[6]],
                       'age_years': [values[7]],
                       'bmi': [values[8]],
                       'gender_women': [values[9]]})
    
    prediction = pipe.predict(df)
    
    if prediction[0] == 0:
        print(f"The diagnosis suggest you dont have any cardiovascular disease. But if you have any concerns about your health contact a professional.")
    else:
        print(f"The diagnosis suggest you MIGHT have a cardiovascular disease. These results are not DEFINITIVE! Contact a professional!!!.")
