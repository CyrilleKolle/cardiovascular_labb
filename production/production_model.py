import pandas as pd
import joblib



test_samples = pd.read_csv("./data/test_samples.csv")
logistic_regression_model = joblib.load("./models/model.pkl")
explanatory_variables, respose_variable = test_samples.drop('cardio', axis=1), test_samples['cardio']
prediction = logistic_regression_model.predict(explanatory_variables)

print(prediction[:10], respose_variable[:10].to_list())

#Courtesy of chatGPT
# Create a DataFrame with the prediction indices and predicted probabilities of class 1

dataframe = pd.DataFrame({'prediction': explanatory_variables['id'].apply(lambda x: f'Id: {x:03}'), 'result': prediction})

# Map the predicted probabilities to class labels (0 or 1) based on a threshold
threshold = 0.5
dataframe['predicted_result'] = dataframe['result'].apply(lambda x: 'probability class 1' if x > threshold else 'probability class 0')

# Insert the actual results into the DataFrame
dataframe['actual_result'] = respose_variable.apply(lambda x: 'positive 1' if x > threshold else 'negative 0')

dataframe.drop('result', axis=1, inplace=True)

# Print the first few rows of the DataFrame
print(dataframe.head())
dataframe.to_csv('./data/prediction.csv', index=False)
